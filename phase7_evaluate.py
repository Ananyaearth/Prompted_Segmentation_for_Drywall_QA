#!/usr/bin/env python3
"""
Phase 7 — Evaluate prompted segmentation outputs and save report assets.

What this script does:
1) Reads the masks produced by inference.py
2) Computes Dice and mIoU
   - cracks: against true polygon GT
   - drywall: against v2 pseudo-GT
3) Saves a few report triptychs: original | GT (or pseudo-GT) | prediction
4) Writes a summary json with metrics and basic footprint info

Usage:
  python phase7_evaluate.py /path/to/project_root
"""

from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Tuple


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def file_size_mb(path: str) -> float:
    if not os.path.exists(path):
        return -1.0
    return os.path.getsize(path) / (1024.0 * 1024.0)


def polygons_to_mask(polygons: List[List[float]], w: int, h: int):
    from PIL import Image, ImageDraw

    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in polygons:
        if not (isinstance(poly, list) and len(poly) >= 6):
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        d.polygon(pts, outline=1, fill=1)
    return m


def pil_mask_to_tensor(mask):
    import torch

    w, h = mask.size
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    t = raw.view(h, w).float()
    return (t > 0).float()


def dice_iou_binary(pred_2d, gt_2d, eps: float = 1e-6) -> Tuple[float, float, float, float]:
    inter = float((pred_2d * gt_2d).sum().item())
    pred_sum = float(pred_2d.sum().item())
    gt_sum = float(gt_2d.sum().item())
    union = float((pred_2d + gt_2d - pred_2d * gt_2d).sum().item())
    return inter, pred_sum, gt_sum, union


def global_metrics(total_inter: float, total_pred: float, total_gt: float, total_union: float) -> Tuple[float, float]:
    eps = 1e-6
    dice = (2 * total_inter + eps) / (total_pred + total_gt + eps)
    iou = (total_inter + eps) / (total_union + eps)
    return float(dice), float(iou)


def save_triptych(orig_path: str, gt_mask, pred_mask, out_path: str) -> None:
    from PIL import Image

    orig = Image.open(orig_path).convert("RGB")
    gt_vis = gt_mask.convert("L").point(lambda px: 255 if px > 0 else 0).convert("RGB")
    pred_vis = pred_mask.convert("L").point(lambda px: 255 if px > 0 else 0).convert("RGB")

    out = Image.new("RGB", (orig.width * 3, orig.height))
    out.paste(orig, (0, 0))
    out.paste(gt_vis, (orig.width, 0))
    out.paste(pred_vis, (orig.width * 2, 0))
    out.save(out_path)


def find_image_by_stem(folder: str, stem: str) -> str:
    for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
        p = os.path.join(folder, stem + ext)
        if os.path.exists(p):
            return p
    return ""


def evaluate_cracks(project_root: str, out_dir: str, n_triptychs: int = 4) -> Dict[str, object]:
    from PIL import Image

    image_dir = os.path.join(project_root, "data", "cracks.v1i.coco", "valid")
    coco_path = os.path.join(image_dir, "_annotations.coco.json")
    pred_dir = os.path.join(project_root, "outputs", "inference", "cracks_valid")
    trip_dir = os.path.join(out_dir, "cracks_triptychs")
    ensure_dir(trip_dir)

    coco = load_json(coco_path)
    img_map = {im["id"]: im for im in coco.get("images", [])}
    ann_map: Dict[int, List[dict]] = {}
    for ann in coco.get("annotations", []):
        ann_map.setdefault(ann["image_id"], []).append(ann)

    total_inter = 0.0
    total_pred = 0.0
    total_gt = 0.0
    total_union = 0.0
    per_image: List[Dict[str, object]] = []
    missing_preds = []
    saved = 0

    for image_id, im in img_map.items():
        stem = os.path.splitext(im["file_name"])[0]
        orig_path = os.path.join(image_dir, im["file_name"])
        pred_path = os.path.join(pred_dir, f"{stem}__segment_crack.png")
        if not os.path.exists(pred_path):
            missing_preds.append(stem)
            continue

        w = int(im.get("width", 640))
        h = int(im.get("height", 640))
        polygons: List[List[float]] = []
        for ann in ann_map.get(image_id, []):
            seg = ann.get("segmentation")
            if isinstance(seg, list):
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:
                        polygons.append(poly)

        gt_mask = polygons_to_mask(polygons, w, h)
        pred_mask = Image.open(pred_path).convert("L")

        gt_t = pil_mask_to_tensor(gt_mask)
        pred_t = pil_mask_to_tensor(pred_mask)
        inter, pred_sum, gt_sum, union = dice_iou_binary(pred_t, gt_t)

        total_inter += inter
        total_pred += pred_sum
        total_gt += gt_sum
        total_union += union

        dice, iou = global_metrics(inter, pred_sum, gt_sum, union)
        per_image.append({"image_id": stem, "dice": dice, "mIoU": iou})

        if saved < n_triptychs:
            save_triptych(orig_path, gt_mask, pred_mask, os.path.join(trip_dir, f"{saved:02d}_{stem}.png"))
            saved += 1

    dice, iou = global_metrics(total_inter, total_pred, total_gt, total_union)
    return {
        "prompt": "segment crack",
        "gt_type": "true_polygon_gt",
        "images_scored": len(per_image),
        "missing_predictions": len(missing_preds),
        "val_dice": dice,
        "val_mIoU": iou,
        "triptychs_dir": trip_dir,
    }


def evaluate_drywall(project_root: str, out_dir: str, n_triptychs: int = 4) -> Dict[str, object]:
    from PIL import Image

    image_dir = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", "valid")
    gt_dir = os.path.join(project_root, "outputs", "phase4_v2", "pseudo_masks", "valid")
    pred_dir = os.path.join(project_root, "outputs", "inference", "drywall_valid")
    trip_dir = os.path.join(out_dir, "drywall_triptychs")
    ensure_dir(trip_dir)

    total_inter = 0.0
    total_pred = 0.0
    total_gt = 0.0
    total_union = 0.0
    per_image: List[Dict[str, object]] = []
    missing_preds = []
    saved = 0

    gt_files = sorted([f for f in os.listdir(gt_dir) if f.lower().endswith(".png")])
    for gt_name in gt_files:
        stem = os.path.splitext(gt_name)[0]
        orig_path = find_image_by_stem(image_dir, stem)
        pred_path = os.path.join(pred_dir, f"{stem}__segment_taping_area.png")
        gt_path = os.path.join(gt_dir, gt_name)

        if not orig_path or not os.path.exists(pred_path):
            missing_preds.append(stem)
            continue

        gt_mask = Image.open(gt_path).convert("L")
        pred_mask = Image.open(pred_path).convert("L")

        gt_t = pil_mask_to_tensor(gt_mask)
        pred_t = pil_mask_to_tensor(pred_mask)
        inter, pred_sum, gt_sum, union = dice_iou_binary(pred_t, gt_t)

        total_inter += inter
        total_pred += pred_sum
        total_gt += gt_sum
        total_union += union

        dice, iou = global_metrics(inter, pred_sum, gt_sum, union)
        per_image.append({"image_id": stem, "dice": dice, "mIoU": iou})

        if saved < n_triptychs:
            save_triptych(orig_path, gt_mask, pred_mask, os.path.join(trip_dir, f"{saved:02d}_{stem}.png"))
            saved += 1

    dice, iou = global_metrics(total_inter, total_pred, total_gt, total_union)
    return {
        "prompt": "segment taping area",
        "gt_type": "pseudo_gt_v2",
        "images_scored": len(per_image),
        "missing_predictions": len(missing_preds),
        "val_dice": dice,
        "val_mIoU": iou,
        "triptychs_dir": trip_dir,
    }


def main() -> int:
    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    if not project_root:
        print("Usage: python phase7_evaluate.py /path/to/project_root", file=sys.stderr)
        return 2

    out_dir = os.path.join(project_root, "outputs", "phase7")
    ensure_dir(out_dir)

    print("[phase7] Evaluating cracks inference outputs...")
    cracks = evaluate_cracks(project_root, out_dir)
    print(f"[phase7] cracks: Dice={cracks['val_dice']:.4f}  mIoU={cracks['val_mIoU']:.4f}")

    print("[phase7] Evaluating drywall inference outputs...")
    drywall = evaluate_drywall(project_root, out_dir)
    print(f"[phase7] drywall: Dice={drywall['val_dice']:.4f}  mIoU={drywall['val_mIoU']:.4f}  (vs pseudo-GT)")

    summary = {
        "cracks": cracks,
        "drywall": drywall,
        "footprint": {
            "cracks_model_mb": file_size_mb(os.path.join(project_root, "outputs", "phase5_resnet", "cracks_best.pt")),
            "drywall_model_mb": file_size_mb(os.path.join(project_root, "outputs", "phase5_resnet", "drywall_best.pt")),
        },
    }

    out_path = os.path.join(out_dir, "summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[phase7] Saved summary: {out_path}")
    print(f"[phase7] Saved triptychs under: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
