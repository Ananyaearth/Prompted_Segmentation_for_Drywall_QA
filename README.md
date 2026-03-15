# Prompted Segmentation for Drywall QA

Text-conditioned segmentation pipeline that takes an image and a natural-language prompt and returns a binary mask for construction defect detection.

## Supported Prompts
- `segment crack` → crack segmentation model
- `segment taping area` → drywall taping/joint segmentation model

## Methodology
1. **Data audit** — Cracks dataset has real polygon masks; drywall dataset is box-only
2. **Pseudo-mask generation** — SAM with bounding box prompts used to create training masks for drywall
3. **Model training** — ResNet34 UNet (pretrained encoder) trained separately for each task
4. **Inference** — Text prompt routes to the correct model, outputs binary PNG mask

## Data Preparation

### Cracks dataset
- Contains real **polygon segmentation masks** for all splits (train/valid/test)
- Used directly for supervised training — no additional preparation needed

### Drywall dataset
- Labels are **bounding boxes only** — no pixel masks in the annotations
- **SAM (Segment Anything Model)** was used to generate pseudo-masks:
  - Each bounding box was passed as a prompt to SAM
  - The predicted mask was hard-clipped to the bbox boundary to prevent leakage
  - All instance masks per image were merged into one binary mask
- v2 pseudo-masks (no bbox padding + hard clipping) were used as the final training targets
- Stricter post-processing versions were tested but discarded — they over-filtered and left most masks empty

## Results

| Prompt | Dice | mIoU | GT type |
|--------|------|------|---------|
| `segment crack` | 0.7045 | 0.5438 | True polygon GT |
| `segment taping area` | 0.7070 | 0.5468 | SAM pseudo-GT |

## Run Inference

```bash
pip install segmentation-models-pytorch torch torchvision

python phase6_inference.py /path/to/project \
    --image_dir /path/to/images \
    --prompt "segment crack" \
    --output_dir /path/to/outputs