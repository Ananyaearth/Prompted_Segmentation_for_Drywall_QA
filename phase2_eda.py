#!/usr/bin/env python3
"""
Phase 2 EDA:
1) Visual sanity checks: montages with boxes/polygons overlaid
2) Geometry stats: a couple of quick histograms + instance/image stats

Usage:
  python phase2_eda.py /path/to/project_root

Expected folders (under project_root):
  data/Drywall-Join-Detect.v2i.coco/train/_annotations.coco.json
  data/cracks.v1i.coco/train/_annotations.coco.json

Outputs:
  outputs/eda/*.png (montages + plots)
"""

from __future__ import annotations

import json
import math
import os
import random
import sys


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_index(coco: dict, images_dir: str):
    # image_id -> image_path
    img_map = {im["id"]: os.path.join(images_dir, im["file_name"]) for im in coco.get("images", [])}
    # image_id -> annotations
    ann_map = {}
    for a in coco.get("annotations", []):
        ann_map.setdefault(a["image_id"], []).append(a)
    return img_map, ann_map


def polygon_area(poly):
    pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
    s = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        s += x1 * y2 - x2 * y1
    return abs(s) / 2.0


def polygon_perimeter(poly):
    pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
    per = 0.0
    for i in range(len(pts)):
        x1, y1 = pts[i]
        x2, y2 = pts[(i + 1) % len(pts)]
        per += math.hypot(x2 - x1, y2 - y1)
    return per


def make_montage(images, cols=6, bg=(30, 30, 30)):
    if not images:
        return None
    w, h = images[0].size
    rows = (len(images) + cols - 1) // cols
    out = Image.new("RGB", (cols * w, rows * h), bg)
    for i, im in enumerate(images):
        r = i // cols
        c = i % cols
        out.paste(im, (c * w, r * h))
    return out


def quick_percentiles(xs):
    xs = sorted(xs)
    if not xs:
        return {}

    def at(p):
        i = int(round((len(xs) - 1) * p))
        return xs[i]

    return {"p50": at(0.50), "p90": at(0.90), "p99": at(0.99), "max": xs[-1]}


def main():
    if len(sys.argv) != 2:
        print("Usage: python phase2_eda.py /path/to/project_root", file=sys.stderr)
        return 2

    project_root = sys.argv[1]

    # Keep EDA repeatable.
    random.seed(7)

    drywall_ann = os.path.join(
        project_root, "data", "Drywall-Join-Detect.v2i.coco", "train", "_annotations.coco.json"
    )
    cracks_ann = os.path.join(project_root, "data", "cracks.v1i.coco", "train", "_annotations.coco.json")

    drywall_images_dir = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", "train")
    cracks_images_dir = os.path.join(project_root, "data", "cracks.v1i.coco", "train")

    out_dir = os.path.join(project_root, "outputs", "eda")
    os.makedirs(out_dir, exist_ok=True)

    # Dependencies are intentionally light (PIL + matplotlib).
    global Image, ImageDraw, plt
    from PIL import Image, ImageDraw  # noqa: PLC0415
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # --------------------
    # 2A) Visual sanity checks (montages)
    # --------------------
    def montage_boxes(coco_path, images_dir, out_path, n=30):
        coco = load_json(coco_path)
        img_map, ann_map = build_index(coco, images_dir)
        ids = list(img_map.keys())
        if not ids:
            return
        sample = random.sample(ids, min(n, len(ids)))

        imgs = []
        for iid in sample:
            im = Image.open(img_map[iid]).convert("RGB")
            d = ImageDraw.Draw(im)
            for a in ann_map.get(iid, []):
                bbox = a.get("bbox")
                if not (isinstance(bbox, list) and len(bbox) == 4):
                    continue
                x, y, w, h = bbox
                if w <= 1 or h <= 1:
                    continue
                d.rectangle([x, y, x + w, y + h], outline=(0, 255, 0), width=3)
            imgs.append(im)

        m = make_montage(imgs, cols=6)
        if m:
            m.save(out_path)

    def montage_polygons(coco_path, images_dir, out_path, n=30):
        coco = load_json(coco_path)
        img_map, ann_map = build_index(coco, images_dir)
        ids = list(img_map.keys())
        if not ids:
            return
        sample = random.sample(ids, min(n, len(ids)))

        imgs = []
        for iid in sample:
            im = Image.open(img_map[iid]).convert("RGB")
            d = ImageDraw.Draw(im)
            for a in ann_map.get(iid, []):
                seg = a.get("segmentation")
                if not (isinstance(seg, list) and len(seg) > 0):
                    continue
                for poly in seg:
                    if not (isinstance(poly, list) and len(poly) >= 6):
                        continue
                    pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
                    d.line(pts + [pts[0]], fill=(255, 0, 0), width=3)
            imgs.append(im)

        m = make_montage(imgs, cols=6)
        if m:
            m.save(out_path)

    drywall_montage = os.path.join(out_dir, "drywall_train_boxes_montage.png")
    cracks_montage = os.path.join(out_dir, "cracks_train_polygons_montage.png")
    montage_boxes(drywall_ann, drywall_images_dir, drywall_montage, n=30)
    montage_polygons(cracks_ann, cracks_images_dir, cracks_montage, n=30)
    print("Saved montage:", drywall_montage)
    print("Saved montage:", cracks_montage)

    # --------------------
    # 2B) Geometry stats (a couple of plots + quick prints)
    # --------------------
    # Drywall: bbox area % and aspect ratio
    coco = load_json(drywall_ann)
    sizes = {im["id"]: (im.get("width", 640), im.get("height", 640)) for im in coco.get("images", [])}

    area_pct = []
    aspect = []
    inst_per_img = {}

    for a in coco.get("annotations", []):
        iid = a["image_id"]
        inst_per_img[iid] = inst_per_img.get(iid, 0) + 1
        x, y, w, h = a.get("bbox", [0, 0, 0, 0])
        if w <= 1 or h <= 1:
            continue
        W, H = sizes.get(iid, (640, 640))
        area_pct.append((w * h) / (W * H))
        aspect.append(w / h)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(area_pct, bins=30)
    plt.title("Drywall bbox area %")
    plt.xlabel("bbox area / image area")

    plt.subplot(1, 2, 2)
    plt.hist(aspect, bins=30)
    plt.title("Drywall bbox aspect")
    plt.xlabel("w / h")

    plt.tight_layout()
    p = os.path.join(out_dir, "drywall_train_bbox_hist.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved plot:", p)
    print("Drywall instances/image percentiles:", quick_percentiles(list(inst_per_img.values())))

    # Cracks: polygon area %, perimeter, thickness proxy
    coco = load_json(cracks_ann)
    sizes = {im["id"]: (im.get("width", 640), im.get("height", 640)) for im in coco.get("images", [])}

    poly_area_pct = []
    poly_per = []
    thickness = []
    inst_per_img = {}

    for a in coco.get("annotations", []):
        iid = a["image_id"]
        inst_per_img[iid] = inst_per_img.get(iid, 0) + 1
        seg = a.get("segmentation")
        if not (isinstance(seg, list) and len(seg) > 0):
            continue

        total_area = 0.0
        total_per = 0.0
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                total_area += polygon_area(poly)
                total_per += polygon_perimeter(poly)
        if total_area <= 0 or total_per <= 0:
            continue

        W, H = sizes.get(iid, (640, 640))
        poly_area_pct.append(total_area / (W * H))
        poly_per.append(total_per)
        thickness.append(total_area / total_per)  # rough proxy: larger => thicker

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.hist(poly_area_pct, bins=30)
    plt.title("Crack polygon area %")
    plt.xlabel("area / image area")

    plt.subplot(1, 3, 2)
    plt.hist(poly_per, bins=30)
    plt.title("Crack polygon perimeter")
    plt.xlabel("perimeter (px)")

    plt.subplot(1, 3, 3)
    plt.hist(thickness, bins=30)
    plt.title("Thickness proxy (area/perimeter)")
    plt.xlabel("px")

    plt.tight_layout()
    p = os.path.join(out_dir, "cracks_train_polygon_hist.png")
    plt.savefig(p, dpi=150)
    plt.close()
    print("Saved plot:", p)
    print("Cracks instances/image percentiles:", quick_percentiles(list(inst_per_img.values())))

    print("\nOpen the montage images and jot down 2–3 observations for your report.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

