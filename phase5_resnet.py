#!/usr/bin/env python3
"""
Phase 5 (ResNet34 variant) — Train segmentation using pretrained ResNet34 encoder.

Uses segmentation_models_pytorch (smp) for a UNet with ResNet34 backbone
pretrained on ImageNet. Everything else (loss, augmentations, datasets,
metrics, saving) is the same as phase5_train.py.

Transfer learning rationale:
  ResNet34 already knows edges, textures, gradients from ImageNet.
  Fine-tuning on our small crack/drywall datasets should give a
  significant boost over a randomly initialized UNet.

Usage:
  pip install segmentation-models-pytorch
  python phase5_resnet.py /path/to/project_root

Outputs:
  outputs/phase5_resnet/
    cracks_best.pt, cracks_metrics.json, cracks_examples/
    drywall_best.pt, drywall_metrics.json, drywall_examples/
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
    from PIL import ImageEnhance

    b = 1.0 + random.uniform(-brightness, brightness)
    c = 1.0 + random.uniform(-contrast, contrast)
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return img


def aug_gamma(img, gamma_range=(0.85, 1.25)):
    g = random.uniform(gamma_range[0], gamma_range[1])
    lut = [int(((i / 255.0) ** g) * 255.0) for i in range(256)]
    if img.mode == "RGB":
        return img.point(lut * 3)
    return img.point(lut)


def aug_blur(img, p=0.2, radius_range=(0.4, 1.0)):
    from PIL import ImageFilter

    if random.random() < p:
        r = random.uniform(radius_range[0], radius_range[1])
        img = img.filter(ImageFilter.GaussianBlur(radius=r))
    return img


def _pil_rgb_to_tensor(img):
    import torch

    w, h = img.size
    b = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    x = b.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    return x


def _pil_l_to_mask_tensor(mask):
    import torch

    w, h = mask.size
    b = torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
    y = b.view(h, w).float()
    y = (y > 0).float().unsqueeze(0)
    return y


# -------------------------
# Model — ResNet34 UNet via smp
# -------------------------

def make_resnet_unet():
    import segmentation_models_pytorch as smp

    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    )
    return model


# -------------------------
# Loss + metrics
# -------------------------

def dice_loss_from_logits(logits, targets, eps: float = 1e-6):
    import torch

    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * inter + eps) / (denom + eps)
    return 1.0 - dice.mean()


def dice_iou_from_logits(logits, targets):
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


# -------------------------
# Visualization
# -------------------------

def _mask_tensor_to_pil(mask_1hw):
    import torch
    from PIL import Image

    m = (mask_1hw.detach().cpu()[0] * 255.0).clamp(0, 255).to(torch.uint8)
    h, w = int(m.shape[0]), int(m.shape[1])
    data = bytes(m.contiguous().view(-1).tolist())
    return Image.frombytes("L", (w, h), data)


def save_triplet_pil(orig_rgb, gt_mask_l, pred_mask_1hw, out_path: str) -> None:
    from PIL import Image

    pred_l = _mask_tensor_to_pil(pred_mask_1hw)

    gt_vis = gt_mask_l.convert("L")
    gt_vis = gt_vis.point(lambda px: 255 if px > 0 else 0)

    out = Image.new("RGB", (orig_rgb.width * 3, orig_rgb.height))
    out.paste(orig_rgb.convert("RGB"), (0, 0))
    out.paste(gt_vis.convert("RGB"), (orig_rgb.width, 0))
    out.paste(pred_l.convert("RGB"), (orig_rgb.width * 2, 0))
    out.save(out_path)


# -------------------------
# Training loop
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

    # Use different LR for pretrained encoder vs randomly-init decoder
    encoder_params = []
    decoder_params = []
    for pname, param in model.named_parameters():
        if pname.startswith("encoder"):
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    opt = torch.optim.Adam([
        {"params": encoder_params, "lr": lr * 0.1},
        {"params": decoder_params, "lr": lr},
    ])
    print(f"[{name}] Encoder LR: {lr * 0.1:.1e}, Decoder LR: {lr:.1e}")

    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=3)

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
        lr_now = float(opt.param_groups[1]["lr"])
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

    if best_state is not None:
        model.load_state_dict(best_state)

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
            gt = polygons_to_mask(s.polygons, s.w, s.h)

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
            gt = Image.open(s.mask_path).convert("L")

            x = _pil_rgb_to_tensor(img).unsqueeze(0).to(device)
            logits = model(x)
            pred = (torch.sigmoid(logits) > 0.5).float().cpu()

            out_path = os.path.join(out_dir, f"val_{i:03d}.png")
            save_triplet_pil(img, gt, pred[0], out_path)


# -------------------------
# Main
# -------------------------

def main():
    seed_everything(7)

    project_root = sys.argv[1] if len(sys.argv) > 1 else None
    if not project_root:
        print("Usage: python phase5_resnet.py /path/to/project_root", file=sys.stderr)
        return 2

    import torch
    from torch.utils.data import DataLoader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("[resnet] Using GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    else:
        print("[resnet] WARNING: no CUDA. Training will be slow.")

    out_dir = os.path.join(project_root, "outputs", "phase5_resnet")
    ensure_dir(out_dir)

    # ----------------- Cracks -----------------
    print("\n[resnet] Training cracks (ResNet34 UNet, pretrained)...")
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

    cracks_model = make_resnet_unet().to(device)
    param_count = sum(p.numel() for p in cracks_model.parameters())
    print(f"[resnet] Model parameters: {param_count:,}")

    train_model(
        name="cracks",
        model=cracks_model,
        train_loader=cracks_train_loader,
        valid_loader=cracks_valid_loader,
        device=device,
        out_dir=out_dir,
        max_epochs=30,
        patience=5,
        lr=5e-4,
        pos_weight=5.0,
        examples_dir=os.path.join(out_dir, "cracks_examples"),
    )
    print("[resnet] Saving crack example triplets...")
    save_examples_cracks(cracks_model, cracks_valid, device=device, out_dir=os.path.join(out_dir, "cracks_examples"), n=12)

    # ----------------- Drywall -----------------
    print("\n[resnet] Training drywall (ResNet34 UNet, pretrained)...")
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

    drywall_model = make_resnet_unet().to(device)
    train_model(
        name="drywall",
        model=drywall_model,
        train_loader=drywall_train_loader,
        valid_loader=drywall_valid_loader,
        device=device,
        out_dir=out_dir,
        max_epochs=25,
        patience=5,
        lr=5e-4,
        pos_weight=None,
        examples_dir=os.path.join(out_dir, "drywall_examples"),
    )
    print("[resnet] Saving drywall example triplets...")
    save_examples_drywall(drywall_model, drywall_valid, device=device, out_dir=os.path.join(out_dir, "drywall_examples"), n=12)

    print("\n[resnet] Done. Check outputs/phase5_resnet/ for models, metrics, and examples.")
    return 0


if __name__ == "__main__":
    main()
