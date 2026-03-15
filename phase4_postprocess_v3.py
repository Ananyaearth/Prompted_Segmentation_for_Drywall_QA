#!/usr/bin/env python3
"""
Phase 4 (v3) — Post-process SAM pseudo-masks from v2 (NO re-running SAM).

Why:
- Even after v2 (bbox clipping + multimask selection), some masks are still too "thick"
  (e.g., triangle/wedge-like leakage inside the bbox).
- Here we tighten masks using simple geometry rules that match the expected target:
  seams/joints are typically long + thin.

What v3 does (per image):
1) Load the v2 pseudo-mask (binary PNG)
2) Split into connected components
3) For each component compute simple shape stats:
   - area
   - bbox fill ratio = area / (bbox_area)   [rejects blob-like regions]
   - elongation = max(w,h) / min(w,h)      [prefers long thin components]
   - thickness proxy = area / perimeter    [prefers thin components]
4) Keep only the best component(s) that pass thresholds
5) Save new v3 mask + a few overlay debug images

Inputs:
  outputs/phase4_v2/pseudo_masks/{train,valid}/*.png

Outputs:
  outputs/phase4_v3/pseudo_masks/{train,valid}/*.png
  outputs/phase4_v3/overlays/{train,valid}/*.png   (subset)
  outputs/phase4_v3/logs.jsonl                     (per-image decisions)

Usage:
  python phase4_postprocess_v3.py /path/to/project_root
"""

from __future__ import annotations

import argparse
import json
import os
import random
from typing import Dict, List, Tuple


from tqdm import tqdm


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def overlay_image(img_rgb, mask_bool, boxes_xywh: List[List[float]]):
    # Simple visual: green overlay for mask + red bbox outlines.
    import numpy as np
    from PIL import Image, ImageDraw

    img = np.array(img_rgb, dtype="uint8")
    mask = mask_bool.astype(bool)

    overlay = img.copy()
    overlay[mask] = (0, 255, 0)
    out = (img * 0.55 + overlay * 0.45).astype("uint8")

    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    for b in boxes_xywh:
        if not (isinstance(b, list) and len(b) == 4):
            continue
        x, y, w, h = b
        d.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
    return pil


def component_stats(mask_u8):
    """Return connected components and their geometry stats."""
    import numpy as np
    import cv2  # type: ignore

    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    comps = []
    for cid in range(1, n):
        x = int(stats[cid, cv2.CC_STAT_LEFT])
        y = int(stats[cid, cv2.CC_STAT_TOP])
        w = int(stats[cid, cv2.CC_STAT_WIDTH])
        h = int(stats[cid, cv2.CC_STAT_HEIGHT])
        area = int(stats[cid, cv2.CC_STAT_AREA])
        bbox_area = max(1, w * h)
        fill_ratio = float(area / bbox_area)
        elongation = float(max(w, h) / max(1, min(w, h)))

        comp_mask = (labels == cid).astype("uint8") * 255
        contours, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        per = 0.0
        for c in contours:
            per += float(cv2.arcLength(c, True))
        per = max(1.0, per)
        thickness = float(area / per)  # px; triangles/blobs tend to be larger

        comps.append(
            {
                "label_id": cid,
                "area": area,
                "x": x,
                "y": y,
                "w": w,
                "h": h,
                "fill_ratio": fill_ratio,
                "elongation": elongation,
                "perimeter": per,
                "thickness": thickness,
            }
        )
    return comps, labels


def select_components(
    comps: List[dict],
    *,
    min_area: int,
    max_fill_ratio: float,
    min_elongation: float,
    max_thickness: float,
):
    """
    Filters components by simple geometry constraints and selects the best one.
    Returns: (selected_label_ids, kept_components, rejected_components)
    """
    kept = []
    rejected = []
    for c in comps:
        reason = []
        if c["area"] < min_area:
            reason.append("too_small")
        if c["fill_ratio"] > max_fill_ratio:
            reason.append("too_filled")
        if c["elongation"] < min_elongation:
            reason.append("not_elongated")
        if c["thickness"] > max_thickness:
            reason.append("too_thick")

        if reason:
            rc = dict(c)
            rc["reject_reason"] = ",".join(reason)
            rejected.append(rc)
        else:
            kept.append(c)

    if not kept:
        return [], kept, rejected

    # score: prefer higher elongation and lower thickness
    kept_sorted = sorted(kept, key=lambda c: (c["elongation"] / (c["thickness"] + 1e-6)), reverse=True)
    best = kept_sorted[0]
    return [best["label_id"]], kept_sorted, rejected


def run_split(
    *,
    split_name: str,
    project_root: str,
    v2_root: str,
    v3_root: str,
    overlay_limit: int,
    seed: int,
    min_area: int,
    max_fill_ratio: float,
    min_elongation: float,
    max_thickness: float,
):
    import numpy as np
    from PIL import Image

    tqdm_local = tqdm
    random.seed(seed)

    coco_path = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", split_name, "_annotations.coco.json")
    images_dir = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", split_name)

    coco = load_json(coco_path)
    img_info = {im["id"]: im for im in coco.get("images", [])}
    ann_map: Dict[int, List[dict]] = {}
    for a in coco.get("annotations", []):
        ann_map.setdefault(a["image_id"], []).append(a)

    v2_mask_dir = os.path.join(v2_root, "pseudo_masks", split_name)
    v3_mask_dir = os.path.join(v3_root, "pseudo_masks", split_name)
    v3_overlay_dir = os.path.join(v3_root, "overlays", split_name)
    ensure_dir(v3_mask_dir)
    ensure_dir(v3_overlay_dir)

    image_ids = list(img_info.keys())
    overlay_ids = set(random.sample(image_ids, min(overlay_limit, len(image_ids))))

    logs = []
    changed = 0
    empty_after = 0

    for image_id in tqdm_local(image_ids, desc=f"[v3] {split_name}"):
        info = img_info[image_id]
        file_name = info["file_name"]
        w = int(info.get("width", 640))
        h = int(info.get("height", 640))

        img_path = os.path.join(images_dir, file_name)
        v2_mask_path = os.path.join(v2_mask_dir, os.path.splitext(file_name)[0] + ".png")
        v3_mask_path = os.path.join(v3_mask_dir, os.path.splitext(file_name)[0] + ".png")

        # Load v2 mask (0/255)
        m2 = Image.open(v2_mask_path).convert("L")
        m2_np = (np.array(m2, dtype=np.uint8) > 0).astype(np.uint8) * 255

        comps, labels = component_stats(m2_np)

        if not comps:
            Image.fromarray(m2_np, mode="L").save(v3_mask_path)
            logs.append(
                {
                    "split": split_name,
                    "file_name": file_name,
                    "status": "no_components",
                    "kept": 0,
                    "rejected": 0,
                }
            )
            continue

        sel, kept, rejected = select_components(
            comps,
            min_area=min_area,
            max_fill_ratio=max_fill_ratio,
            min_elongation=min_elongation,
            max_thickness=max_thickness,
        )

        # Build v3 mask
        m3 = np.zeros((h, w), dtype=np.uint8)
        for cid in sel:
            m3[labels == cid] = 255

        Image.fromarray(m3, mode="L").save(v3_mask_path)

        if (m3 > 0).sum() == 0:
            empty_after += 1
        if not np.array_equal(m2_np, m3):
            changed += 1

        logs.append(
            {
                "split": split_name,
                "file_name": file_name,
                "status": "ok" if sel else "no_component_passed_filters",
                "selected_components": sel,
                "num_components": len(comps),
                "num_kept": len(kept),
                "num_rejected": len(rejected),
                "kept_top1": kept[0] if kept else None,
                "rejected_sample": rejected[:3],
            }
        )

        if image_id in overlay_ids:
            img = Image.open(img_path).convert("RGB")
            boxes = [a.get("bbox") for a in ann_map.get(image_id, [])]
            ov = overlay_image(img, (m3 > 0), boxes_xywh=boxes)
            ov.save(os.path.join(v3_overlay_dir, os.path.splitext(file_name)[0] + ".png"))

    return logs, {"split": split_name, "changed": changed, "empty_after": empty_after, "images": len(image_ids)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_root", type=str)
    ap.add_argument("--v2_root", type=str, default=None, help="Path to outputs/phase4_v2 (default: <root>/outputs/phase4_v2)")
    ap.add_argument("--v3_root", type=str, default=None, help="Path to outputs/phase4_v3 (default: <root>/outputs/phase4_v3)")
    ap.add_argument("--overlay_limit", type=int, default=60)
    ap.add_argument("--seed", type=int, default=7)

    # Simple geometry thresholds (tune based on what you see)
    ap.add_argument("--min_area", type=int, default=150)
    ap.add_argument("--max_fill_ratio", type=float, default=0.35, help="area / (component bbox area)")
    ap.add_argument("--min_elongation", type=float, default=6.0, help="max(w,h)/min(w,h)")
    ap.add_argument("--max_thickness", type=float, default=8.0, help="area / perimeter (px)")

    args = ap.parse_args()

    project_root = args.project_root
    v2_root = args.v2_root or os.path.join(project_root, "outputs", "phase4_v2")
    v3_root = args.v3_root or os.path.join(project_root, "outputs", "phase4_v3")
    ensure_dir(v3_root)

    print("[phase4 v3] Post-processing v2 pseudo-masks (no SAM inference).")
    print(f"[phase4 v3] v2_root: {v2_root}")
    print(f"[phase4 v3] v3_root: {v3_root}")
    print(
        f"[phase4 v3] thresholds: min_area={args.min_area}, max_fill_ratio={args.max_fill_ratio}, "
        f"min_elongation={args.min_elongation}, max_thickness={args.max_thickness}"
    )

    all_logs = []
    summaries = []
    for split in ["train", "valid"]:
        logs, summary = run_split(
            split_name=split,
            project_root=project_root,
            v2_root=v2_root,
            v3_root=v3_root,
            overlay_limit=args.overlay_limit,
            seed=args.seed,
            min_area=args.min_area,
            max_fill_ratio=args.max_fill_ratio,
            min_elongation=args.min_elongation,
            max_thickness=args.max_thickness,
        )
        all_logs.extend(logs)
        summaries.append(summary)
        print(f"[phase4 v3] {split}: images={summary['images']} changed={summary['changed']} empty_after={summary['empty_after']}")

    save_jsonl(os.path.join(v3_root, "logs.jsonl"), all_logs)
    with open(os.path.join(v3_root, "stats.json"), "w", encoding="utf-8") as f:
        json.dump({"version": 3, "summaries": summaries, "thresholds": vars(args)}, f, indent=2)
    print("[phase4 v3] Saved logs + stats. Check outputs/phase4_v3/ (pseudo_masks + overlays).")


if __name__ == "__main__":
    main()

