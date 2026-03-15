#!/usr/bin/env python3
"""
Quick COCO audit (boxes vs masks + basic geometry stats).

Usage:
  python coco_audit.py /path/to/dataset_dir
  python coco_audit.py /path/to/split/_annotations.coco.json
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def _median(xs: Sequence[float]) -> Optional[float]:
    # Small helper used for quick "typical size" summaries in the console output.
    if not xs:
        return None
    xs_sorted = sorted(xs)
    n = len(xs_sorted)
    mid = n // 2
    if n % 2 == 1:
        return xs_sorted[mid]
    return (xs_sorted[mid - 1] + xs_sorted[mid]) / 2.0


def _percentile(xs: Sequence[float], p: float) -> Optional[float]:
    if not xs:
        return None
    if p <= 0:
        return min(xs)
    if p >= 100:
        return max(xs)
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * (p / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return xs_sorted[int(k)]
    d0 = xs_sorted[f] * (c - k)
    d1 = xs_sorted[c] * (k - f)
    return d0 + d1


@dataclass
class CocoSplitStats:
    path: str
    num_images: int
    num_annotations: int
    categories: List[Tuple[int, str]]
    seg_present: bool
    seg_nonempty_count: int
    bbox_count: int
    bbox_degenerate_count: int
    bbox_area_px: List[float]
    bbox_aspect: List[float]


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _as_categories(coco: Dict[str, Any]) -> List[Tuple[int, str]]:
    cats = coco.get("categories", []) or []
    out: List[Tuple[int, str]] = []
    for c in cats:
        out.append((int(c.get("id")), str(c.get("name"))))
    return sorted(out, key=lambda x: x[0])


def audit_coco_json(path: str) -> CocoSplitStats:
    coco = _load_json(path)

    images = coco.get("images", []) or []
    annotations = coco.get("annotations", []) or []
    categories = _as_categories(coco)

    seg_present = False
    seg_nonempty_count = 0

    bbox_count = 0
    bbox_degenerate_count = 0
    bbox_area_px: List[float] = []
    bbox_aspect: List[float] = []

    for ann in annotations:
        if "segmentation" in ann:
            seg_present = True
            seg = ann.get("segmentation")
            # COCO segmentation can be:
            # - []                -> no mask provided (common in detection-only exports)
            # - [[x1,y1,...], ..] -> polygon(s)
            # - { ... }           -> RLE (used for masks in some COCO variants)
            if isinstance(seg, dict):
                # If it's a dict, it's almost certainly an RLE mask.
                seg_nonempty_count += 1
            elif isinstance(seg, list) and len(seg) > 0:
                # Non-empty list usually means polygon annotations exist.
                seg_nonempty_count += 1

        bbox = ann.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            bbox_count += 1
            x, y, w, h = bbox
            w = float(w)
            h = float(h)
            if w <= 1.0 or h <= 1.0:
                bbox_degenerate_count += 1
                continue
            bbox_area_px.append(w * h)
            bbox_aspect.append(w / h)

    return CocoSplitStats(
        path=path,
        num_images=len(images),
        num_annotations=len(annotations),
        categories=categories,
        seg_present=seg_present,
        seg_nonempty_count=seg_nonempty_count,
        bbox_count=bbox_count,
        bbox_degenerate_count=bbox_degenerate_count,
        bbox_area_px=bbox_area_px,
        bbox_aspect=bbox_aspect,
    )


def _fmt_num(x: Optional[float]) -> str:
    if x is None:
        return "n/a"
    if abs(x) >= 1000:
        return f"{x:,.1f}"
    return f"{x:.3f}"


def print_stats(s: CocoSplitStats) -> None:
    print(f"\n=== {s.path} ===")
    print(f"images: {s.num_images}")
    print(f"annotations: {s.num_annotations}")
    print("categories:")
    for cid, name in s.categories:
        print(f"  - {cid}: {name}")

    if s.seg_present:
        print(f"segmentation field: present (non-empty: {s.seg_nonempty_count}/{s.num_annotations})")
    else:
        print("segmentation field: not present")

    print(f"bbox field: {s.bbox_count}/{s.num_annotations} annotations")
    if s.bbox_count:
        print(f"degenerate bboxes (w<=1 or h<=1): {s.bbox_degenerate_count}")
        # Area/aspect summaries are useful to understand object geometry:
        # long thin seams -> extreme aspect ratios; tiny cracks -> small areas.
        print("bbox area (px^2): "
              f"min={_fmt_num(min(s.bbox_area_px) if s.bbox_area_px else None)}, "
              f"med={_fmt_num(_median(s.bbox_area_px))}, "
              f"p95={_fmt_num(_percentile(s.bbox_area_px, 95))}, "
              f"max={_fmt_num(max(s.bbox_area_px) if s.bbox_area_px else None)}")
        print("bbox aspect (w/h): "
              f"min={_fmt_num(min(s.bbox_aspect) if s.bbox_aspect else None)}, "
              f"med={_fmt_num(_median(s.bbox_aspect))}, "
              f"p95={_fmt_num(_percentile(s.bbox_aspect, 95))}, "
              f"max={_fmt_num(max(s.bbox_aspect) if s.bbox_aspect else None)}")


def _find_coco_jsons_in_dir(dataset_dir: str) -> List[str]:
    # Typical Roboflow COCO export: <dataset>/{train,valid,test}/_annotations.coco.json
    out: List[str] = []
    for split in ("train", "valid", "test"):
        p = os.path.join(dataset_dir, split, "_annotations.coco.json")
        if os.path.exists(p):
            out.append(p)
    return out


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Usage: python coco_audit.py <dataset_dir | coco_json_path>", file=sys.stderr)
        return 2

    target = argv[1]
    paths: List[str] = []
    if os.path.isdir(target):
        paths = _find_coco_jsons_in_dir(target)
        if not paths:
            print(f"No _annotations.coco.json found under {target}", file=sys.stderr)
            return 1
    else:
        paths = [target]

    for p in paths:
        stats = audit_coco_json(p)
        print_stats(stats)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

