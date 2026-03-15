#!/usr/bin/env python3
"""
Phase 6 inference.

Given a folder of images and a text prompt, this script routes the prompt
to the right model, runs segmentation, and saves binary PNG masks.

Output format (from the assignment PDF):
  - PNG, single channel, values {0, 255}, same size as source image
  - Filename: {image_id}__segment_{prompt}.png

Usage:
  python inference.py /path/to/project_root \
      --image_dir /path/to/images \
      --prompt "segment crack" \
      --output_dir /path/to/output_masks
"""

import argparse
import os
import sys
import time


def build_model():
    import segmentation_models_pytorch as smp

    return smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
    )


def pil_to_tensor(img):
    """PIL RGB image -> float tensor [1, 3, H, W] in [0,1]. No numpy needed."""
    import torch

    w, h = img.size
    raw = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
    t = raw.view(h, w, 3).permute(2, 0, 1).float() / 255.0
    return t.unsqueeze(0)


def route_prompt(prompt: str) -> str:
    """
    Decide which model to use based on the text prompt.
    Simple keyword matching -- good enough for the two-class case.
    """
    p = prompt.lower().strip()

    crack_words = ["crack"]
    drywall_words = ["taping", "tape", "joint", "seam", "drywall"]

    for w in crack_words:
        if w in p:
            return "cracks"

    for w in drywall_words:
        if w in p:
            return "drywall"

    raise ValueError(f"Unsupported prompt: {prompt}")


def normalize_prompt(prompt: str) -> str:
    """Turn a prompt string into a filename-safe slug."""
    return prompt.strip().lower().replace(" ", "_")


def run_inference(
    project_root: str,
    image_dir: str,
    prompt: str,
    output_dir: str,
):
    import torch
    from PIL import Image

    task = route_prompt(prompt)
    print(f"Prompt: \"{prompt}\"")
    print(f"Routed to: {task}")

    # pick the right checkpoint
    ckpt_path = os.path.join(project_root, "outputs", "phase5_resnet", f"{task}_best.pt")
    if not os.path.isfile(ckpt_path):
        print(f"[error] Checkpoint not found: {ckpt_path}")
        return 1

    # load model
    model = build_model()
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    print(f"Model loaded from {ckpt_path}  (device={device})")

    # gather images
    valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    files = sorted([
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_ext
    ])
    if not files:
        print(f"[error] No images found in {image_dir}")
        return 1

    print(f"Found {len(files)} images in {image_dir}")

    os.makedirs(output_dir, exist_ok=True)
    slug = normalize_prompt(prompt)
    times = []

    with torch.no_grad():
        for fname in files:
            t0 = time.time()

            img = Image.open(os.path.join(image_dir, fname)).convert("RGB")
            orig_w, orig_h = img.size

            x = pil_to_tensor(img).to(device)

            with torch.cuda.amp.autocast(enabled=(device == "cuda")):
                logits = model(x)

            mask = (torch.sigmoid(logits) > 0.5).float()

            m = mask[0, 0].cpu()
            m_bytes = (m * 255).to(torch.uint8)
            h_out, w_out = m_bytes.shape
            pil_mask = Image.frombytes("L", (w_out, h_out), bytes(m_bytes.contiguous().view(-1).tolist()))

            if (w_out, h_out) != (orig_w, orig_h):
                pil_mask = pil_mask.resize((orig_w, orig_h), Image.Resampling.NEAREST)

            image_id = os.path.splitext(fname)[0]
            out_name = f"{image_id}__{slug}.png"
            pil_mask.save(os.path.join(output_dir, out_name))

            dt = time.time() - t0
            times.append(dt)

    avg_ms = 1000.0 * sum(times) / len(times)
    total_s = sum(times)
    print(f"\nDone. {len(files)} masks saved to {output_dir}")
    print(f"Avg inference: {avg_ms:.1f} ms/image   Total: {total_s:.1f}s")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Prompted segmentation inference")
    parser.add_argument("project_root", help="Path to project root (contains outputs/)")
    parser.add_argument("--image_dir", required=True, help="Folder with input images")
    parser.add_argument("--prompt", required=True, help='Text prompt, e.g. "segment crack"')
    parser.add_argument("--output_dir", required=True, help="Where to save output masks")
    args = parser.parse_args()

    return run_inference(
        project_root=args.project_root,
        image_dir=args.image_dir,
        prompt=args.prompt,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    rc = main()
    if rc:
        sys.exit(rc)
