#!/usr/bin/env python3
"""
Phase 3 — Baselines (kept intentionally simple).

What this script does:
1) Cracks (segmentation available):
   - Train a small UNet for a few epochs
   - Report mIoU + Dice on the validation split
   - Save a few "orig | GT | pred" images for the report

2) Drywall (box-only):
   - Create a naive "box-fill mask" baseline (filled rectangles)
   - Save a few "orig | box-mask" examples to show why boxes are not masks

Why this step is here:
- It gives a direct baseline for cracks.
- It also shows why box-fill is not enough for drywall segmentation.

Usage:
  python phase3_baselines.py /path/to/project_root

Outputs:
  outputs/phase3/
    cracks_unet.pt
    cracks_metrics.json
    cracks_examples/
    drywall_boxfill_examples/
"""

from __future__ import annotations

import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


from tqdm import tqdm


# -------------------------
# COCO helpers
# -------------------------


def coco_index(coco: dict, images_dir: str):
    img_map = {im["id"]: os.path.join(images_dir, im["file_name"]) for im in coco.get("images", [])}
    size_map = {im["id"]: (int(im.get("width", 0)), int(im.get("height", 0))) for im in coco.get("images", [])}
    ann_map: Dict[int, List[dict]] = {}
    for a in coco.get("annotations", []):
        ann_map.setdefault(a["image_id"], []).append(a)
    return img_map, size_map, ann_map


def polygons_to_mask(polygons: List[List[float]], w: int, h: int):
    from PIL import Image, ImageDraw  # noqa: PLC0415

    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in polygons:
        if not (isinstance(poly, list) and len(poly) >= 6):
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        d.polygon(pts, outline=1, fill=1)
    return m


def boxes_to_mask(boxes: List[List[float]], w: int, h: int):
    from PIL import Image, ImageDraw  # noqa: PLC0415

    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for bbox in boxes:
        if not (isinstance(bbox, list) and len(bbox) == 4):
            continue
        x, y, bw, bh = bbox
        if bw <= 1 or bh <= 1:
            continue
        d.rectangle([x, y, x + bw, y + bh], outline=1, fill=1)
    return m


# -------------------------
# Model (small UNet)
# -------------------------


def make_unet(in_ch: int = 3, base: int = 32):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415

    def CBR(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.enc1 = CBR(in_ch, base)
            self.enc2 = CBR(base, base * 2)
            self.enc3 = CBR(base * 2, base * 4)
            self.pool = nn.MaxPool2d(2)

            self.bottleneck = CBR(base * 4, base * 8)

            self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
            self.dec3 = CBR(base * 8, base * 4)
            self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
            self.dec2 = CBR(base * 4, base * 2)
            self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
            self.dec1 = CBR(base * 2, base)

            self.out = nn.Conv2d(base, 1, 1)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            b = self.bottleneck(self.pool(e3))

            d3 = self.up3(b)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out(d1)

    return UNet()


# -------------------------
# Metrics (binary)
# -------------------------


def dice_iou_from_logits(logits, targets, eps: float = 1e-6):
    # logits: (N,1,H,W), targets: (N,1,H,W) in {0,1}
    import torch  # noqa: PLC0415

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    pred_sum = preds.sum(dim=(1, 2, 3))
    tgt_sum = targets.sum(dim=(1, 2, 3))

    dice = (2 * inter + eps) / (pred_sum + tgt_sum + eps)
    iou = (inter + eps) / (union + eps)
    return dice.mean().item(), iou.mean().item()


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    import torch  # noqa: PLC0415

    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


# -------------------------
# Datasets
# -------------------------


@dataclass
class CrackSample:
    image_path: str
    w: int
    h: int
    polygons: List[List[float]]  # list of polygon coordinate lists


def load_cracks_samples(coco_path: str, images_dir: str) -> List[CrackSample]:
    coco = load_json(coco_path)
    img_map, size_map, ann_map = coco_index(coco, images_dir)

    samples: List[CrackSample] = []
    for image_id, image_path in img_map.items():
        w, h = size_map.get(image_id, (640, 640))
        polygons: List[List[float]] = []
        for a in ann_map.get(image_id, []):
            seg = a.get("segmentation")
            if isinstance(seg, list) and len(seg) > 0:
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:
                        polygons.append(poly)
        samples.append(CrackSample(image_path=image_path, w=w, h=h, polygons=polygons))
    return samples


class CracksDataset:
    def __init__(self, samples: List[CrackSample], train: bool):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Keep this readable: PIL -> numpy -> torch tensors.
        import numpy as np  # noqa: PLC0415
        import torch  # noqa: PLC0415
        from PIL import Image  # noqa: PLC0415

        s = self.samples[idx]
        im = Image.open(s.image_path).convert("RGB")
        m = polygons_to_mask(s.polygons, s.w, s.h)

        # Simple augment: horizontal flip (helps a bit, low risk).
        if self.train and random.random() < 0.5:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
            m = m.transpose(Image.FLIP_LEFT_RIGHT)

        x = np.array(im, dtype=np.float32) / 255.0  # HWC in [0,1]
        y = np.array(m, dtype=np.float32)  # HW in {0,1}

        x = torch.from_numpy(x).permute(2, 0, 1)  # CHW
        y = torch.from_numpy(y).unsqueeze(0)  # 1HW

        return x, y


# -------------------------
# Visual helpers
# -------------------------


def save_triplet(image_tensor, gt_mask, pred_mask, out_path: str):
    # Save a simple side-by-side image: orig | GT | pred
    from PIL import Image  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415

    x = image_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    x = (x * 255.0).clip(0, 255).astype("uint8")
    gt = (gt_mask.detach().cpu().numpy()[0] * 255.0).clip(0, 255).astype("uint8")
    pr = (pred_mask.detach().cpu().numpy()[0] * 255.0).clip(0, 255).astype("uint8")

    im = Image.fromarray(x)
    gt_im = Image.fromarray(gt).convert("RGB")
    pr_im = Image.fromarray(pr).convert("RGB")

    out = Image.new("RGB", (im.width * 3, im.height))
    out.paste(im, (0, 0))
    out.paste(gt_im, (im.width, 0))
    out.paste(pr_im, (im.width * 2, 0))
    out.save(out_path)


def save_overlay(image_path: str, mask_img, out_path: str, color=(0, 255, 0), alpha=120):
    # image + translucent mask overlay (good for the drywall box-fill baseline)
    # Implemented with numpy to keep it fast and simple.
    import numpy as np  # noqa: PLC0415
    from PIL import Image  # noqa: PLC0415

    im = Image.open(image_path).convert("RGB")
    m = mask_img.convert("L")

    img = np.array(im, dtype=np.uint8)
    mask = np.array(m, dtype=np.uint8) > 0

    overlay = img.copy()
    overlay[mask] = (color[0], color[1], color[2])

    a = alpha / 255.0
    out = (img * (1 - a) + overlay * a).astype(np.uint8)
    Image.fromarray(out).save(out_path)


# -------------------------
# Phase 3 entrypoint
# -------------------------


def run_cracks_unet(project_root: str, epochs: int = 3, batch_size: int = 8, lr: float = 1e-3, seed: int = 7):
    import torch  # noqa: PLC0415
    import torch.nn as nn  # noqa: PLC0415
    from torch.utils.data import DataLoader  # noqa: PLC0415

    seed_everything(seed)
    tqdm_local = tqdm

    train_ann = os.path.join(project_root, "data", "cracks.v1i.coco", "train", "_annotations.coco.json")
    valid_ann = os.path.join(project_root, "data", "cracks.v1i.coco", "valid", "_annotations.coco.json")
    train_dir = os.path.join(project_root, "data", "cracks.v1i.coco", "train")
    valid_dir = os.path.join(project_root, "data", "cracks.v1i.coco", "valid")

    train_samples = load_cracks_samples(train_ann, train_dir)
    valid_samples = load_cracks_samples(valid_ann, valid_dir)

    train_ds = CracksDataset(train_samples, train=True)
    valid_ds = CracksDataset(valid_samples, train=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"[cracks] Using GPU: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True
    else:
        print("[cracks] WARNING: CUDA not available. Training will be slow on CPU.")

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device == "cuda"),
    )

    model = make_unet().to(device)

    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    # Train only a few epochs: this is a baseline, not the final tuning.
    for ep in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = 0.0
        pbar = tqdm_local(train_loader, desc=f"[cracks] train ep {ep}/{epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            logits = model(x)
            loss = bce(logits, y) + dice_loss_from_logits(logits, y)
            loss.backward()
            opt.step()
            total += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        avg_loss = total / max(1, len(train_loader))

        model.eval()
        dices = []
        ious = []
        with torch.no_grad():
            pbar = tqdm_local(valid_loader, desc=f"[cracks] valid ep {ep}/{epochs}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                logits = model(x)
                d, i = dice_iou_from_logits(logits, y)
                dices.append(d)
                ious.append(i)

        dt = time.time() - t0
        print(
            f"[cracks] epoch {ep}/{epochs}  train_loss={avg_loss:.4f}  "
            f"val_dice={sum(dices)/len(dices):.4f}  val_iou={sum(ious)/len(ious):.4f}  "
            f"time={dt:.1f}s"
        )

    # Save model + metrics + a few visuals 
    out_root = os.path.join(project_root, "outputs", "phase3")
    ensure_dir(out_root)

    model_path = os.path.join(out_root, "cracks_unet.pt")
    torch.save(model.state_dict(), model_path)

    metrics = {"val_dice": float(sum(dices) / len(dices)), "val_mIoU": float(sum(ious) / len(ious)), "epochs": epochs}
    with open(os.path.join(out_root, "cracks_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    ex_dir = os.path.join(out_root, "cracks_examples")
    ensure_dir(ex_dir)

    # Save 12 examples from the validation set
    model.eval()
    saved = 0
    with torch.no_grad():
        for x, y in valid_loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits)
            pred = (probs > 0.5).float()

            for i in range(x.size(0)):
                out_path = os.path.join(ex_dir, f"val_{saved:03d}.png")
                save_triplet(x[i].cpu(), y[i].cpu(), pred[i].cpu(), out_path)
                saved += 1
                if saved >= 12:
                    break
            if saved >= 12:
                break

    print("Saved:", model_path)
    print("Saved:", os.path.join(out_root, "cracks_metrics.json"))
    print("Saved examples:", ex_dir)


def run_drywall_boxfill(project_root: str, num_examples: int = 12, seed: int = 7):
    seed_everything(seed)

    ann_path = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", "valid", "_annotations.coco.json")
    images_dir = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", "valid")

    coco = load_json(ann_path)
    img_map, size_map, ann_map = coco_index(coco, images_dir)

    ids = list(img_map.keys())
    if not ids:
        print("No drywall images found.")
        return

    sample = random.sample(ids, min(num_examples, len(ids)))

    out_root = os.path.join(project_root, "outputs", "phase3")
    ex_dir = os.path.join(out_root, "drywall_boxfill_examples")
    ensure_dir(ex_dir)

    print(f"[drywall] Creating {len(sample)} box-fill overlay examples.")
    for k, iid in enumerate(tqdm(sample, desc="[drywall] box-fill examples", leave=False)):
        image_path = img_map[iid]
        w, h = size_map.get(iid, (640, 640))
        boxes = [a.get("bbox") for a in ann_map.get(iid, [])]
        m = boxes_to_mask(boxes, w, h)
        out_path = os.path.join(ex_dir, f"val_{k:03d}.png")
        save_overlay(image_path, m, out_path, color=(0, 255, 0), alpha=110)

    print("Saved examples:", ex_dir)
    print("These examples show how box-fill tends to over-cover the target region.")


def main():
    if len(sys.argv) != 2:
        print("Usage: python phase3_baselines.py /path/to/project_root", file=sys.stderr)
        return 2

    project_root = sys.argv[1]

    print("[phase3] Starting Phase 3 baselines...")
    run_cracks_unet(project_root, epochs=3, batch_size=4, lr=1e-3, seed=7)
    run_drywall_boxfill(project_root, num_examples=12, seed=7)
    print("[phase3] Done. Check outputs/phase3/ for metrics and example images.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

