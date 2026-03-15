#!/usr/bin/env python3
"""
Phase 5 — Train the UNet models.

We train two binary segmentation models using the same small UNet:
1) Cracks (true polygon masks) — upgraded training:
   - epochs=20, early stopping (patience=3)
   - lr=1e-3 + ReduceLROnPlateau (monitor val Dice)
   - augmentations: flip + brightness/contrast + gamma + mild blur
   - loss: BCE + Dice

2) Drywall taping area (pseudo masks from SAM v2):
   - epochs=15, early stopping (patience=3)
   - lr=1e-3 + ReduceLROnPlateau (monitor val Dice vs pseudo-GT)
   - augmentations: flip + mild brightness/contrast
   - loss: BCE + Dice

Usage:
  python phase5_train.py /path/to/project_root

Outputs:
  outputs/phase5/
    cracks_best.pt
    cracks_metrics.json
    cracks_examples/*.png         (orig | GT | pred)
    drywall_best.pt
    drywall_metrics.json
    drywall_examples/*.png        (orig | pseudoGT | pred)
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


from tqdm import tqdm


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -------------------------
# Masks from labels
# -------------------------


def polygons_to_mask(polygons: List[List[float]], w: int, h: int):
    # Rasterize polygons using PIL (good enough, simple).
    from PIL import Image, ImageDraw

    m = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(m)
    for poly in polygons:
        if not (isinstance(poly, list) and len(poly) >= 6):
            continue
        pts = [(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)]
        d.polygon(pts, outline=1, fill=1)
    return m


# -------------------------
# Augmentations
# -------------------------


def aug_flip(img, mask):
    from PIL import Image

    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img, mask


def aug_rotate(img, mask, max_deg=15):
    from PIL import Image

    if random.random() < 0.5:
        angle = random.uniform(-max_deg, max_deg)
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)
        mask = mask.rotate(angle, resample=Image.Resampling.NEAREST, fillcolor=0)
    return img, mask


def aug_brightness_contrast(img, brightness=0.12, contrast=0.15):
    # brightness/contrast jitter using PIL (no numpy dependency).
    from PIL import ImageEnhance

    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast, contrast)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img


def aug_gamma(img, gamma_range=(0.85, 1.25)):
    # Gamma correction using a lookup table (PIL point), no numpy dependency.
    g = random.uniform(gamma_range[0], gamma_range[1])
    lut = [int(((i / 255.0) ** g) * 255.0) for i in range(256)]
    if img.mode == "RGB":
        return img.point(lut * 3)
    return img.point(lut)


def _pil_rgb_to_tensor(img):
    """
    Convert PIL RGB -> torch float tensor CHW in [0,1] without numpy.
    This avoids torch.from_numpy(), which fails if torch can't initialize NumPy.
    """
    import torch

    w, h = img.size
    b = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    x = b.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    return x


def _pil_l_to_mask_tensor(mask):
    """
    Convert PIL L -> torch float tensor 1HW in {0,1} without numpy.
    """
    import torch

    w, h = mask.size
    b = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    y = b.view(h, w).float()
    y = (y > 0).float().unsqueeze(0)
    return y


def aug_blur(img, p=0.2, radius_range=(0.4, 1.0)):
    from PIL import ImageFilter

    if random.random() < p:
        r = random.uniform(radius_range[0], radius_range[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
    return img


# -------------------------
# Model (same UNet as Phase 3)
# -------------------------


def make_unet(in_ch: int = 3, base: int = 48):
    import torch
    import torch.nn as nn

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
            # Encoder: 4 levels (48 -> 96 -> 192 -> 384) + bottleneck (768)
            self.enc1 = CBR(in_ch, base)          # 640
            self.enc2 = CBR(base, base * 2)        # 320
            self.enc3 = CBR(base * 2, base * 4)    # 160
            self.enc4 = CBR(base * 4, base * 8)    # 80
            self.pool = nn.MaxPool2d(2)

            self.bottleneck = CBR(base * 8, base * 16)  # 40
            self.drop = nn.Dropout2d(0.1)

            # Decoder: mirror of encoder
            self.up4 = nn.ConvTranspose2d(base * 16, base * 8, 2, stride=2)
            self.dec4 = CBR(base * 16, base * 8)
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
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            b = self.drop(b)

            d4 = self.up4(b)
            d4 = self.dec4(torch.cat([d4, e4], dim=1))
            d3 = self.up3(d4)
            d3 = self.dec3(torch.cat([d3, e3], dim=1))
            d2 = self.up2(d3)
            d2 = self.dec2(torch.cat([d2, e2], dim=1))
            d1 = self.up1(d2)
            d1 = self.dec1(torch.cat([d1, e1], dim=1))
            return self.out(d1)

    return UNet()


# -------------------------
# Loss + metrics (binary)
# -------------------------


def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def dice_iou_from_logits(logits, targets):
    """Return raw sums for global accumulation (not per-batch averages)."""
    import torch

    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    inter = (preds * targets).sum().item()
    pred_sum = preds.sum().item()
    tgt_sum = targets.sum().item()
    union = (preds + targets - preds * targets).sum().item()

    return inter, pred_sum, tgt_sum, union


# -------------------------
# Datasets
# -------------------------


@dataclass
class CrackSample:
    image_path: str
    w: int
    h: int
    polygons: List[List[float]]


def coco_index(coco: dict, images_dir: str):
    img_map = {im["id"]: os.path.join(images_dir, im["file_name"]) for im in coco.get("images", [])}
    size_map = {im["id"]: (int(im.get("width", 640)), int(im.get("height", 640))) for im in coco.get("images", [])}
    ann_map: Dict[int, List[dict]] = {}
    for a in coco.get("annotations", []):
        ann_map.setdefault(a["image_id"], []).append(a)
    return img_map, size_map, ann_map


def load_cracks_samples(coco_path: str, images_dir: str) -> List[CrackSample]:
    coco = load_json(coco_path)
    img_map, size_map, ann_map = coco_index(coco, images_dir)

    out: List[CrackSample] = []
    for image_id, image_path in img_map.items():
        w, h = size_map.get(image_id, (640, 640))
        polygons: List[List[float]] = []
        for a in ann_map.get(image_id, []):
            seg = a.get("segmentation")
            if isinstance(seg, list) and len(seg) > 0:
                for poly in seg:
                    if isinstance(poly, list) and len(poly) >= 6:
                        polygons.append(poly)
        out.append(CrackSample(image_path=image_path, w=w, h=h, polygons=polygons))
    return out


class CracksDataset:
    def __init__(self, samples: List[CrackSample], train: bool):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        from PIL import Image

        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        mask = polygons_to_mask(s.polygons, s.w, s.h)

        if self.train:
            img, mask = aug_flip(img, mask)
            img, mask = aug_rotate(img, mask)
            img = aug_brightness_contrast(img)
            img = aug_gamma(img)
            img = aug_blur(img)

        x = _pil_rgb_to_tensor(img)
        y = _pil_l_to_mask_tensor(mask)
        return x, y


@dataclass
class DrywallSample:
    image_path: str
    mask_path: str


def load_drywall_samples(project_root: str, split: str) -> List[DrywallSample]:
    # We use v2 pseudo-masks as supervision targets.
    img_dir = os.path.join(project_root, "data", "Drywall-Join-Detect.v2i.coco", split)
    mask_dir = os.path.join(project_root, "outputs", "phase4_v2", "pseudo_masks", split)

    if not os.path.exists(mask_dir):
        raise FileNotFoundError(f"Missing v2 pseudo-masks: {mask_dir}")

    out: List[DrywallSample] = []
    for f in os.listdir(mask_dir):
        if not f.lower().endswith(".png"):
            continue
        base = os.path.splitext(f)[0]
        mask_path = os.path.join(mask_dir, f)

        # Find corresponding image file by stem
        image_path = None
        for ext in [".jpg", ".jpeg", ".png"]:
            p = os.path.join(img_dir, base + ext)
            if os.path.exists(p):
                image_path = p
                break
        if image_path is None:
            continue

        out.append(DrywallSample(image_path=image_path, mask_path=mask_path))

    out.sort(key=lambda s: s.image_path)
    return out


class DrywallDataset:
    def __init__(self, samples: List[DrywallSample], train: bool):
        self.samples = samples
        self.train = train

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import torch
        from PIL import Image

        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        mask = Image.open(s.mask_path).convert("L")

        if self.train:
            img, mask = aug_flip(img, mask)
            img = aug_brightness_contrast(img, brightness=0.10, contrast=0.12)

        x = _pil_rgb_to_tensor(img)
        y = _pil_l_to_mask_tensor(mask)
        return x, y


def _mask_tensor_to_pil(mask_1hw):
    """
    Convert a 1xHxW torch tensor in {0,1} to a PIL 'L' image (0/255),
    without using numpy.
    """
    import torch
    from PIL import Image

    m = (mask_1hw.detach().cpu()[0] * 255.0).clamp(0, 255).to(torch.uint8)  # HW
    h, w = int(m.shape[0]), int(m.shape[1])
    data = bytes(m.contiguous().view(-1).tolist())
    return Image.frombytes("L", (w, h), data)


def save_triplet_pil(orig_rgb, gt_mask_l, pred_mask_1hw, out_path: str) -> None:
    """
    Save a simple triplet: orig | GT (or pseudo-GT) | pred.
    We load orig/GT from disk (PIL) and only convert the predicted mask tensor.
    This avoids large tensor->python list conversions for the RGB image.
    """
    from PIL import Image

    pred_l = _mask_tensor_to_pil(pred_mask_1hw)

    # Scale GT to 0/255 so it's actually visible (polygons_to_mask uses fill=1).
    gt_vis = gt_mask_l.convert("L")
    gt_vis = gt_vis.point(lambda px: 255 if px > 0 else 0)

    out = Image.new("RGB", (orig_rgb.width * 3, orig_rgb.height))
    out.paste(orig_rgb.convert("RGB"), (0, 0))
    out.paste(gt_vis.convert("RGB"), (orig_rgb.width, 0))
    out.paste(pred_l.convert("RGB"), (orig_rgb.width * 2, 0))
    out.save(out_path)


# -------------------------
# Training loop (shared)
# -------------------------


def train_model(
    *,
    name: str,
    model,
    train_loader,
    valid_loader,
    device: str,
    out_dir: str,
    max_epochs: int,
    patience: int,
    lr: float,
    pos_weight: Optional[float],
    examples_dir: str,
):
    import torch
    import torch.nn as nn

    tqdm_local = tqdm

    if pos_weight is not None:
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
        print(f"[{name}] Using pos_weight={pos_weight} for BCE")
    else:
        bce = nn.BCEWithLogitsLoss()

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

    # Mixed precision for T4 speedup (Tensor Cores do float16 faster)
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    if use_amp:
        print(f"[{name}] Using mixed precision (AMP)")

    best_dice = -1.0
    best_iou = -1.0
    best_epoch = -1
    bad_epochs = 0
    best_state = None

    for ep in range(1, max_epochs + 1):
        t0 = time.time()

        model.train()
        train_loss = 0.0
        pbar = tqdm_local(train_loader, desc=f"[{name}] train {ep}/{max_epochs}", leave=False)
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = bce(logits, y) + dice_loss_from_logits(logits, y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            train_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss /= max(1, len(train_loader))

        model.eval()
        total_inter = 0.0
        total_pred = 0.0
        total_tgt = 0.0
        total_union = 0.0
        with torch.no_grad():
            pbar = tqdm_local(valid_loader, desc=f"[{name}] valid {ep}/{max_epochs}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                inter, ps, ts, un = dice_iou_from_logits(logits, y)
                total_inter += inter
                total_pred += ps
                total_tgt += ts
                total_union += un

        eps = 1e-6
        val_dice = float((2 * total_inter + eps) / (total_pred + total_tgt + eps))
        val_iou = float((total_inter + eps) / (total_union + eps))

        sched.step(val_dice)
        lr_now = float(opt.param_groups[0]["lr"])
        dt = time.time() - t0

        print(
            f"[{name}] epoch {ep}/{max_epochs}  lr={lr_now:.2e}  train_loss={train_loss:.4f}  "
            f"val_dice={val_dice:.4f}  val_iou={val_iou:.4f}  time={dt:.1f}s"
        )

        improved = val_dice > best_dice + 1e-4
        if improved:
            best_dice = val_dice
            best_iou = val_iou
            best_epoch = ep
            bad_epochs = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[{name}] new best val_dice={best_dice:.4f} (epoch {best_epoch})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{name}] early stop (no improvement in {patience} epochs).")
                break

    # Restore best weights
    if best_state is not None:
        model.load_state_dict(best_state)

    # Save model + metrics
    ensure_dir(out_dir)
    model_path = os.path.join(out_dir, f"{name}_best.pt")
    torch.save(model.state_dict(), model_path)

    metrics = {"best_val_dice": best_dice, "best_val_mIoU": best_iou, "best_epoch": best_epoch}
    with open(os.path.join(out_dir, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{name}] Saved model: {model_path}")
    print(f"[{name}] Saved metrics: {os.path.join(out_dir, f'{name}_metrics.json')}")
    return {"best_val_dice": best_dice, "best_val_mIoU": best_iou, "best_epoch": best_epoch, "model_path": model_path}


def save_examples_cracks(model, samples: List[CrackSample], device: str, out_dir: str, n: int = 12) -> None:
    import torch
    from PIL import Image

    ensure_dir(out_dir)
    model.eval()
    picked = samples[:n]
    with torch.no_grad():
        for i, s in enumerate(picked):
            img = Image.open(s.image_path).convert("RGB")
            gt = polygons_to_mask(s.polygons, s.w, s.h)  # L in {0,1}

            x = _pil_rgb_to_tensor(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()

            out_path = os.path.join(out_dir, f"val_{i:03d}.png")
            save_triplet_pil(img, gt, pred[0], out_path)


def save_examples_drywall(model, samples: List[DrywallSample], device: str, out_dir: str, n: int = 12) -> None:
    import torch
    from PIL import Image

    ensure_dir(out_dir)
    model.eval()
    picked = samples[:n]
    with torch.no_grad():
        for i, s in enumerate(picked):
            img = Image.open(s.image_path).convert("RGB")
            gt = Image.open(s.mask_path).convert("L")  # 0/255 pseudo-GT

            x = _pil_rgb_to_tensor(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()

            out_path = os.path.join(out_dir, f"val_{i:03d}.png")
            save_triplet_pil(img, gt, pred[0], out_path)


def main():
    seed_everything(7)

    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    if not project_root:
        print("Usage: python phase5_train.py /path/to/project_root", file=sys.stderr)
        return 2

    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("[phase5] Using GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("[phase5] WARNING: CUDA not available. Training will be slow on CPU.")

    out_dir = os.path.join(project_root, "outputs", "phase5")
    ensure_dir(out_dir)

    # ----------------- Cracks (upgraded) -----------------
    print("\n[phase5] Training cracks (upgraded UNet)...")
    cracks_train = load_cracks_samples(
        os.path.join(project_root, "data", "cracks.v1i.coco", "train", "_annotations.coco.json"),
        os.path.join(project_root, "data", "cracks.v1i.coco", "train"),
    )
    cracks_valid = load_cracks_samples(
        os.path.join(project_root, "data", "cracks.v1i.coco", "valid", "_annotations.coco.json"),
        os.path.join(project_root, "data", "cracks.v1i.coco", "valid"),
    )

    cracks_train_ds = CracksDataset(cracks_train, train=True)
    cracks_valid_ds = CracksDataset(cracks_valid, train=False)

    cracks_train_loader = DataLoader(
        cracks_train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=(device == "cuda")
    )
    cracks_valid_loader = DataLoader(
        cracks_valid_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=(device == "cuda")
    )

    cracks_model = make_unet().to(device)
    train_model(
        name="cracks",
        model=cracks_model,
        train_loader=cracks_train_loader,
        valid_loader=cracks_valid_loader,
        device=device,
        out_dir=out_dir,
        max_epochs=40,
        patience=5,
        lr=5e-4,
        pos_weight=5.0,
        examples_dir=os.path.join(out_dir, "cracks_examples"),
    )
    print("[phase5] Saving crack example triplets...")
    save_examples_cracks(cracks_model, cracks_valid, device=device, out_dir=os.path.join(out_dir, "cracks_examples"), n=12)

    # ----------------- Drywall (train on v2 pseudo masks) -----------------
    print("\n[phase5] Training drywall (UNet on v2 pseudo-masks)...")
    drywall_train = load_drywall_samples(project_root, split="train")
    drywall_valid = load_drywall_samples(project_root, split="valid")

    print(f"[drywall] train samples: {len(drywall_train)}  valid samples: {len(drywall_valid)}")

    drywall_train_ds = DrywallDataset(drywall_train, train=True)
    drywall_valid_ds = DrywallDataset(drywall_valid, train=False)

    drywall_train_loader = DataLoader(
        drywall_train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=(device == "cuda")
    )
    drywall_valid_loader = DataLoader(
        drywall_valid_ds, batch_size=4, shuffle=False, num_workers=2, pin_memory=(device == "cuda")
    )

    drywall_model = make_unet().to(device)
    train_model(
        name="drywall",
        model=drywall_model,
        train_loader=drywall_train_loader,
        valid_loader=drywall_valid_loader,
        device=device,
        out_dir=out_dir,
        max_epochs=30,
        patience=5,
        lr=5e-4,
        pos_weight=None,
        examples_dir=os.path.join(out_dir, "drywall_examples"),
    )
    print("[phase5] Saving drywall example triplets...")
    save_examples_drywall(drywall_model, drywall_valid, device=device, out_dir=os.path.join(out_dir, "drywall_examples"), n=12)

    print("\n[phase5] Done. Check outputs/phase5/ for models, metrics, and example triplets.")
    return 0


if __name__ == "__main__":
    main()

