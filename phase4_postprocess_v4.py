#!/usr/bin/env python3
"""
Phase 4 (v4) — Post-process SAM pseudo-masks from v2 with tunable strictness.

Context:
- v2 creates pseudo-masks using SAM.
- v3 added strict geometry filtering, but it was too aggressive (many empty masks).
- v4 adds:
  1) A simple "tuning" mode to visualize 5 samples across a few presets
  2) A safer selection rule: if nothing passes hard thresholds, keep the best-scoring component
     instead of returning an empty mask.

This script NEVER re-runs SAM. It only reads v2 masks and writes new masks.

Inputs:
  outputs/phase4_v2/pseudo_masks/{train,valid}/*.png

Outputs:
  Tune mode:
    outputs/phase4_v4/tuning/<split>/*.png  (comparisons)
  Apply mode:
    outputs/phase4_v4/pseudo_masks/{train,valid}/*.png
    outputs/phase4_v4/overlays/{train,valid}/*.png (subset)
    outputs/phase4_v4/stats.json

Usage:
  # 1) Tune on 5 images (valid split):
  python phase4_postprocess_v4.py /path/to/project_root tune --split valid

  # 2) Apply chosen preset to whole dataset:
  python phase4_postprocess_v4.py /path/to/project_root apply --preset medium
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


def overlay_image(img_rgb, mask_bool, boxes_xywh: List[List[float]]):
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
    """
    Connected components + simple stats.
    Returns (components, labels) where:
      - components: list of dicts with area/fill_ratio/elongation/thickness
      - labels: HxW int labels
    """
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
        thickness = float(area / per)  # px (larger => thicker/blobbier)

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


def score_component(c: dict) -> float:
    return float(c["elongation"] / (c["thickness"] + 1e-6))


def select_component_v4(
    comps: List[dict],
    *,
    min_area: int,
    max_fill_ratio: float,
    min_elongation: float,
    max_thickness: float,
):
    """Pick the best component, and keep a fallback if all filters fail."""
    if not comps:
        return None, 0, False

    passed = []
    for c in comps:
        if c["area"] < min_area:
            continue
        if c["fill_ratio"] > max_fill_ratio:
            continue
        if c["elongation"] < min_elongation:
            continue
        if c["thickness"] > max_thickness:
            continue
        passed.append(c)

    best_all = sorted(comps, key=score_component, reverse=True)[0]
    if passed:
        best = sorted(passed, key=score_component, reverse=True)[0]
        return best["label_id"], len(passed), False

    return best_all["label_id"], 0, True


PRESETS = {
    # Very strict (similar to v3)
    "strict": {"min_area": 150, "max_fill_ratio": 0.35, "min_elongation": 6.0, "max_thickness": 8.0},
    # Medium (recommended starting point)
    "medium": {"min_area": 80, "max_fill_ratio": 0.55, "min_elongation": 3.5, "max_thickness": 14.0},
    # Loose (keeps more, good if masks disappear too often)
    "loose": {"min_area": 40, "max_fill_ratio": 0.75, "min_elongation": 2.5, "max_thickness": 22.0},
}


def list_mask_bases(mask_dir: str) -> List[str]:
    if not os.path.exists(mask_dir):
        return []
    out = []
    for f in os.listdir(mask_dir):
        if f.lower().endswith(".png"):
            out.append(os.path.splitext(f)[0])
    return sorted(out)


def load_original_image(project_root: str, split: str, base: str):
    from PIL import Image

    img_dir = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", split)
    # masks are named using the original image file stem
    for ext in [".jpg", ".jpeg", ".png"]:
        p = os.path.join(img_dir, base + ext)
        if os.path.exists(p):
            return Image.open(p).convert("RGB")
    # fallback scan
    for f in os.listdir(img_dir):
        if os.path.splitext(f)[0] == base:
            return Image.open(os.path.join(img_dir, f)).convert("RGB")
    return None


def load_boxes_for_base(project_root: str, split: str, base: str) -> List[List[float]]:
    coco_path = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", split, "_annotations.coco.json")
    coco = load_json(coco_path)
    # Find image_id by file_name stem match
    image_id = None
    for im in coco.get("images", []):
        if os.path.splitext(im.get("file_name", ""))[0] == base:
            image_id = im["id"]
            break
    if image_id is None:
        return []
    boxes = []
    for a in coco.get("annotations", []):
        if a.get("image_id") == image_id:
            b = a.get("bbox")
            if isinstance(b, list) and len(b) == 4:
                boxes.append(b)
    return boxes


def apply_postprocess_to_mask(mask_u8, preset: dict):
    import numpy as np

    comps, labels = component_stats(mask_u8)
    sel, pass_count, used_fallback = select_component_v4(comps, **preset)

    if sel is None:
        return np.zeros_like(mask_u8), {"num_components": 0, "pass_count": 0, "fallback": False}

    out = np.zeros_like(mask_u8)
    out[labels == sel] = 255
    return out, {"num_components": len(comps), "pass_count": pass_count, "fallback": used_fallback}


def cmd_tune(project_root: str, split: str, seed: int, n: int):
    import numpy as np
    from PIL import Image, ImageDraw

    random.seed(seed)
    v2_dir = os.path.join(project_root, "outputs", "phase4_v2", "pseudo_masks", split)
    bases = list_mask_bases(v2_dir)
    if not bases:
        raise FileNotFoundError(f"No v2 masks found at: {v2_dir}")

    sample = random.sample(bases, min(n, len(bases)))
    out_dir = os.path.join(project_root, "outputs", "phase4_v4", "tuning", split)
    ensure_dir(out_dir)

    print("[phase4 v4 tune] Presets:", ", ".join(PRESETS.keys()))
    print("[phase4 v4 tune] Showing bases:", sample)

    for base in sample:
        img = load_original_image(project_root, split, base)
        if img is None:
            continue

        v2_mask = Image.open(os.path.join(v2_dir, base + ".png")).convert("L")
        v2_np = (np.array(v2_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255

        # Build panel list: original | v2 | preset outputs...
        titles = ["original", "v2"]
        panels = [img, v2_mask]
        meta = {"base": base, "presets": {}}

        for name, preset in PRESETS.items():
            out_u8, info = apply_postprocess_to_mask(v2_np, preset)
            panels.append(Image.fromarray(out_u8, mode="L"))
            titles.append(name)
            meta["presets"][name] = info

        # Save a simple side-by-side image using PIL (no matplotlib dependency).
        # Layout: one row -> [original | v2 | strict | medium | loose]
        w, h = panels[0].size
        header_h = 26
        out = Image.new("RGB", (w * len(panels), h + header_h), (15, 15, 15))
        draw = ImageDraw.Draw(out)

        for i, (t, p) in enumerate(zip(titles, panels)):
            if i == 0:
                tile = p.convert("RGB")
            else:
                tile = p.convert("L").convert("RGB")
            out.paste(tile, (i * w, header_h))
            draw.text((i * w + 6, 6), t, fill=(255, 255, 255))

        fig_path = os.path.join(out_dir, base + ".png")
        out.save(fig_path)

        # Save per-image metadata so you can see if fallback happened.
        with open(os.path.join(out_dir, base + ".json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    print("[phase4 v4 tune] Saved tuning comparisons to:", out_dir)


def cmd_apply(project_root: str, preset_name: str, overlay_limit: int, seed: int):
    import numpy as np
    from PIL import Image

    if preset_name not in PRESETS:
        raise ValueError(f"Unknown preset '{preset_name}'. Choose from: {list(PRESETS.keys())}")

    preset = PRESETS[preset_name]
    tqdm_local = tqdm
    random.seed(seed)

    v2_root = os.path.join(project_root, "outputs", "phase4_v2")
    v4_root = os.path.join(project_root, "outputs", "phase4_v4")
    ensure_dir(v4_root)

    summaries = []
    for split in ["train", "valid"]:
        v2_dir = os.path.join(v2_root, "pseudo_masks", split)
        bases = list_mask_bases(v2_dir)

        out_mask_dir = os.path.join(v4_root, "pseudo_masks", split)
        out_ov_dir = os.path.join(v4_root, "overlays", split)
        ensure_dir(out_mask_dir)
        ensure_dir(out_ov_dir)

        overlay_ids = set(random.sample(bases, min(overlay_limit, len(bases))))

        changed = 0
        fallback_ct = 0
        total = 0

        for base in tqdm_local(bases, desc=f"[phase4 v4 apply] {split}"):
            total += 1
            v2_mask = Image.open(os.path.join(v2_dir, base + ".png")).convert("L")
            v2_np = (np.array(v2_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255

            out_u8, info = apply_postprocess_to_mask(v2_np, preset)
            if info.get("fallback"):
                fallback_ct += 1

            if not np.array_equal(v2_np, out_u8):
                changed += 1

            Image.fromarray(out_u8, mode="L").save(os.path.join(out_mask_dir, base + ".png"))

            if base in overlay_ids:
                img = load_original_image(project_root, split, base)
                if img is not None:
                    boxes = load_boxes_for_base(project_root, split, base)
                    ov = overlay_image(img, (out_u8 > 0), boxes_xywh=boxes)
                    ov.save(os.path.join(out_ov_dir, base + ".png"))

        summaries.append(
            {
                "split": split,
                "images": total,
                "changed_vs_v2": changed,
                "fallback_used": fallback_ct,
            }
        )

    stats = {"version": 4, "preset": preset_name, "preset_params": preset, "summaries": summaries}
    with open(os.path.join(v4_root, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("[phase4 v4 apply] Saved outputs to:", v4_root)
    print("[phase4 v4 apply] Wrote stats.json with summary counts.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_root", type=str)
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_tune = sub.add_parser("tune")
    ap_tune.add_argument("--split", type=str, default="valid", choices=["train", "valid"])
    ap_tune.add_argument("--seed", type=int, default=7)
    ap_tune.add_argument("--n", type=int, default=5)

    ap_apply = sub.add_parser("apply")
    ap_apply.add_argument("--preset", type=str, default="medium", choices=list(PRESETS.keys()))
    ap_apply.add_argument("--overlay_limit", type=int, default=60)
    ap_apply.add_argument("--seed", type=int, default=7)

    args = ap.parse_args()

    import cv2  

    if args.cmd == "tune":
        cmd_tune(args.project_root, split=args.split, seed=args.seed, n=args.n)
    elif args.cmd == "apply":
        cmd_apply(args.project_root, preset_name=args.preset, overlay_limit=args.overlay_limit, seed=args.seed)


if __name__ == "__main__":
    main()

