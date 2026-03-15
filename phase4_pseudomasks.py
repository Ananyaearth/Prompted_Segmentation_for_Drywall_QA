#!/usr/bin/env python3
"""
Phase 4 — Pseudo-mask generation for drywall (box-only labels) using SAM (v2).

Goal:
- Drywall dataset only has bounding boxes. We need pixel masks to train a segmenter.
- Use SAM with box prompts: bbox -> mask, then union masks per image.

What this script saves (under outputs/phase4_v2/):
- pseudo_masks/{train,valid}/<file_name>.png          (single-channel {0,255})
- overlays/{train,valid}/<file_name>.png              (image with bbox + mask overlay)
- ignored/ignored_boxes.jsonl                         (degenerate/invalid boxes)
- ignored/images/<split>/*.png                        (visuals for ignored/failed boxes)
- stats.json                                          (counts + settings used)

Usage:
  python phase4_pseudomasks.py /path/to/project_root --sam_type vit_b --sam_ckpt /path/to/sam_vit_b.pth

"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, List, Tuple


from tqdm import tqdm


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def pad_box_xywh(b: List[float], w: int, h: int, pad_frac: float) -> List[float]:
    """Pad bbox by a fraction of its size, then clamp to image."""
    x, y, bw, bh = b
    px = bw * pad_frac
    py = bh * pad_frac
    x2 = x + bw
    y2 = y + bh
    x = clamp(x - px, 0, w - 1)
    y = clamp(y - py, 0, h - 1)
    x2 = clamp(x2 + px, 0, w - 1)
    y2 = clamp(y2 + py, 0, h - 1)
    return [x, y, x2, y2]  # xyxy


def box_xywh_to_xyxy(b: List[float]) -> List[float]:
    x, y, bw, bh = b
    return [x, y, x + bw, y + bh]


def _clip_mask_to_box(mask_bool, box_xyxy, h: int, w: int):
    """Keep the SAM mask inside the box."""
    import numpy as np

    x1, y1, x2, y2 = box_xyxy
    x1 = int(clamp(round(x1), 0, w - 1))
    y1 = int(clamp(round(y1), 0, h - 1))
    x2 = int(clamp(round(x2), 0, w - 1))
    y2 = int(clamp(round(y2), 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((h, w), dtype=bool)

    clipped = np.zeros((h, w), dtype=bool)
    clipped[y1:y2, x1:x2] = mask_bool[y1:y2, x1:x2]
    return clipped


def _choose_mask(masks, scores, box_xyxy, h: int, w: int, max_area_frac: float):
    """Pick the tightest mask that still looks reasonable inside the box."""
    import numpy as np

    x1, y1, x2, y2 = box_xyxy
    bbox_area = max(1.0, float((x2 - x1) * (y2 - y1)))

    cand = []
    for i in range(masks.shape[0]):
        m = _clip_mask_to_box(masks[i].astype(bool), box_xyxy, h=h, w=w)
        area_frac = float(m.sum() / bbox_area)
        cand.append((float(scores[i]), area_frac, m))

    # Prefer masks that don't fill most of the bbox.
    ok = [c for c in cand if c[1] <= max_area_frac]
    picked_from_ok = True
    if not ok:
        ok = cand
        picked_from_ok = False

    ok.sort(key=lambda x: x[0], reverse=True)  # by score
    best_score, best_area_frac, best_mask = ok[0]
    return best_mask, best_score, best_area_frac, picked_from_ok, [(c[0], c[1]) for c in cand]


def maybe_cleanup_mask(mask_bool, min_area: int, do_morph: bool):
    """Remove very small pieces and optionally smooth the mask a bit."""
    if min_area <= 0 and not do_morph:
        return mask_bool

    import cv2  # type: ignore
    import numpy as np

    m = (mask_bool.astype("uint8") * 255)

    if min_area > 0:
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        keep = np.zeros_like(m)
        for cid in range(1, n):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area >= min_area:
                keep[labels == cid] = 255
        m = keep

    if do_morph:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k, iterations=1)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, k, iterations=1)

    return (m > 0)


def draw_overlay(image_rgb, boxes_xyxy: List[List[float]], mask_bool):
    """Return a PIL image with bbox outlines + translucent mask overlay."""
    from PIL import Image, ImageDraw
    import numpy as np

    img = np.array(image_rgb, dtype="uint8")
    m = mask_bool.astype(bool)

    # simple translucent green overlay
    overlay = img.copy()
    overlay[m] = (0, 255, 0)
    out = (img * 0.55 + overlay * 0.45).astype("uint8")

    pil = Image.fromarray(out)
    d = ImageDraw.Draw(pil)
    for (x1, y1, x2, y2) in boxes_xyxy:
        d.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
    return pil


def save_jsonl(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def run_split(
    *,
    split_name: str,
    coco_path: str,
    images_dir: str,
    out_root: str,
    predictor,
    pad_frac: float,
    max_area_frac: float,
    min_area: int,
    do_morph: bool,
    overlay_limit: int,
    seed: int,
):
    from PIL import Image
    import numpy as np

    tqdm_local = tqdm
    random.seed(seed)

    coco = load_json(coco_path)
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])

    # image_id -> image info
    img_map: Dict[int, dict] = {im["id"]: im for im in images}
    # image_id -> list of annotations
    ann_map: Dict[int, List[dict]] = {}
    for a in annotations:
        ann_map.setdefault(a["image_id"], []).append(a)

    pseudo_dir = os.path.join(out_root, "pseudo_masks", split_name)
    overlay_dir = os.path.join(out_root, "overlays", split_name)
    ignored_img_dir = os.path.join(out_root, "ignored", "images", split_name)
    ensure_dir(pseudo_dir)
    ensure_dir(overlay_dir)
    ensure_dir(ignored_img_dir)

    ignored_rows: List[dict] = []
    suspicious_rows: List[dict] = []

    total_boxes = 0
    used_boxes = 0
    total_images = 0

    # Choose a random subset of images for overlays (so we don't save thousands).
    image_ids = list(img_map.keys())
    overlay_ids = set(random.sample(image_ids, min(overlay_limit, len(image_ids))))

    for image_id in tqdm_local(image_ids, desc=f"[phase4] {split_name} images"):
        info = img_map[image_id]
        file_name = info["file_name"]
        w = int(info.get("width", 640))
        h = int(info.get("height", 640))
        image_path = os.path.join(images_dir, file_name)

        anns = ann_map.get(image_id, [])
        total_images += 1

        # Load and set the image for SAM once per image.
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)
        predictor.set_image(img_np)

        union = np.zeros((h, w), dtype=bool)
        boxes_xyxy: List[List[float]] = []

        for a in anns:
            bbox = a.get("bbox")
            if not (isinstance(bbox, list) and len(bbox) == 4):
                continue
            total_boxes += 1
            x, y, bw, bh = bbox
            if bw <= 1 or bh <= 1:
                ignored_rows.append(
                    {"split": split_name, "image_id": image_id, "file_name": file_name, "bbox": bbox, "reason": "degenerate_box"}
                )
                continue

            # v2: default is no padding. (Padding can be re-enabled via --pad_frac if needed.)
            raw_box_xyxy = box_xywh_to_xyxy(bbox)
            if pad_frac > 0:
                prompt_box_xyxy = pad_box_xywh(bbox, w, h, pad_frac)
            else:
                prompt_box_xyxy = raw_box_xyxy

            boxes_xyxy.append(raw_box_xyxy)

            masks, scores, _ = predictor.predict(
                box=np.array(prompt_box_xyxy, dtype=np.float32),
                multimask_output=True,
            )
            m, best_score, best_area_frac, picked_from_ok, cand_summary = _choose_mask(
                masks, scores, raw_box_xyxy, h=h, w=w, max_area_frac=max_area_frac
            )

            m = maybe_cleanup_mask(m, min_area=min_area, do_morph=do_morph)
            union |= m
            used_boxes += 1

            if not picked_from_ok:
                suspicious_rows.append(
                    {
                        "split": split_name,
                        "image_id": image_id,
                        "file_name": file_name,
                        "bbox": bbox,
                        "reason": "no_candidate_under_max_area_frac",
                        "max_area_frac": max_area_frac,
                        "chosen_score": best_score,
                        "chosen_area_frac": best_area_frac,
                        "candidates_(score,area_frac)": cand_summary,
                    }
                )

        # Save pseudo mask PNG (0/255)
        out_mask = (union.astype("uint8") * 255)
        mask_path = os.path.join(pseudo_dir, os.path.splitext(file_name)[0] + ".png")
        Image.fromarray(out_mask, mode="L").save(mask_path)

        # Save overlay for a subset
        if image_id in overlay_ids:
            overlay = draw_overlay(img, boxes_xyxy, union)
            overlay_path = os.path.join(overlay_dir, os.path.splitext(file_name)[0] + ".png")
            overlay.save(overlay_path)

        if any(r["image_id"] == image_id for r in ignored_rows):
            from PIL import ImageDraw

            vis = img.copy()
            d = ImageDraw.Draw(vis)
            for a in anns:
                bbox = a.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                x, y, bw, bh = bbox
                color = (255, 0, 0) if (bw <= 1 or bh <= 1) else (255, 200, 0)
                d.rectangle([x, y, x + bw, y + bh], outline=color, width=2)
            d.text((5, 5), "has ignored boxes", fill=(255, 0, 0))
            vis_path = os.path.join(ignored_img_dir, os.path.splitext(file_name)[0] + ".png")
            vis.save(vis_path)

    ignored_path = os.path.join(out_root, "ignored", "ignored_boxes.jsonl")
    suspicious_path = os.path.join(out_root, "ignored", "suspicious_masks.jsonl")
    ensure_dir(os.path.join(out_root, "ignored"))
    save_jsonl(ignored_path, ignored_rows)
    save_jsonl(suspicious_path, suspicious_rows)

    return {
        "split": split_name,
        "images": total_images,
        "total_boxes": total_boxes,
        "used_boxes": used_boxes,
        "ignored_boxes": len(ignored_rows),
        "suspicious_masks": len(suspicious_rows),
        "pseudo_dir": pseudo_dir,
        "overlay_dir": overlay_dir,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("project_root", type=str)
    ap.add_argument("--sam_type", type=str, default="vit_b", help="SAM encoder type (vit_h/vit_l/vit_b)")
    ap.add_argument("--sam_ckpt", type=str, required=True, help="Path to SAM checkpoint .pth")
    # v2 defaults:
    ap.add_argument("--pad_frac", type=float, default=0.0, help="Pad fraction applied to bbox before SAM (0 disables)")
    ap.add_argument(
        "--max_area_frac",
        type=float,
        default=0.60,
        help="For multimask selection: prefer masks with (mask_area / bbox_area) <= this value",
    )
    ap.add_argument("--min_area", type=int, default=100, help="Remove connected components smaller than this (0 disables)")
    ap.add_argument("--morph", action="store_true", help="Apply small morphology close/open (requires opencv)")
    ap.add_argument("--overlay_limit", type=int, default=40, help="How many overlay images to save per split")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    project_root = args.project_root
    out_root = os.path.join(project_root, "outputs", "phase4_v2")
    ensure_dir(out_root)

    from segment_anything import SamPredictor, sam_model_registry  # type: ignore  # noqa: PLC0415
    import torch  # noqa: PLC0415

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("[phase4 v2] Using GPU:", torch.cuda.get_device_name(0))
    else:
        print("[phase4 v2] WARNING: CUDA not available; SAM will be slow on CPU.")

    print("[phase4 v2] Loading SAM model...")
    sam = sam_model_registry[args.sam_type](checkpoint=args.sam_ckpt)
    sam.to(device=device)
    predictor = SamPredictor(sam)

    # Drywall dataset paths
    base = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco")
    train_coco = os.path.join(base, "train", "_annotations.coco.json")
    valid_coco = os.path.join(base, "valid", "_annotations.coco.json")
    train_dir = os.path.join(base, "train")
    valid_dir = os.path.join(base, "valid")

    print("[phase4 v2] Generating pseudo-masks for drywall (train + valid)...")
    print(
        f"[phase4 v2] pad_frac={args.pad_frac}  max_area_frac={args.max_area_frac}  "
        f"min_area={args.min_area}  morph={args.morph}  overlay_limit={args.overlay_limit}"
    )

    stats = {
        "version": 2,
        "sam_type": args.sam_type,
        "sam_ckpt": args.sam_ckpt,
        "device": device,
        "pad_frac": args.pad_frac,
        "max_area_frac": args.max_area_frac,
        "min_area": args.min_area,
        "morph": bool(args.morph),
        "overlay_limit": args.overlay_limit,
        "seed": args.seed,
        "splits": [],
    }

    for split_name, coco_path, images_dir in [
        ("train", train_coco, train_dir),
        ("valid", valid_coco, valid_dir),
    ]:
        s = run_split(
            split_name=split_name,
            coco_path=coco_path,
            images_dir=images_dir,
            out_root=out_root,
            predictor=predictor,
            pad_frac=args.pad_frac,
            max_area_frac=args.max_area_frac,
            min_area=args.min_area,
            do_morph=args.morph,
            overlay_limit=args.overlay_limit,
            seed=args.seed,
        )
        stats["splits"].append(s)
        print(
            f"[phase4 v2] {split_name}: images={s['images']} boxes_used={s['used_boxes']}/{s['total_boxes']} "
            f"ignored={s['ignored_boxes']} suspicious={s['suspicious_masks']}"
        )

    stats_path = os.path.join(out_root, "stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("[phase4 v2] Saved stats:", stats_path)
    print("[phase4 v2] Done. Check outputs/phase4_v2/ (pseudo_masks, overlays, ignored logs).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

