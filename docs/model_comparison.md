# Model Comparison Plan

## Positioning

YOLO remains the baseline, not the project centerpiece.

The main claim should be:

```text
YOLO is a fast baseline.
D-FINE / RT-DETR are stronger detector candidates for traffic-camera training.
SAM/SAM2 helps produce masks and rule evidence, but is not the detector baseline by itself.
```

This is stronger than saying "we used YOLO + VLM", because the project becomes
a reusable training and validation pipeline:

1. Convert public traffic-camera datasets into a common format.
2. Train multiple detector families.
3. Use SAM/SAM2 for segmentation-assisted labels and road-rule evidence.
4. Compare detector mAP, event precision/recall, edge FPS, hot-spot coverage,
   and A1/A2 risk coverage.

## Current GPU Sanity Run

The first local GPU run is intentionally small. It proves that the pipeline can
train on the local RTX 3060, not that the detector is good yet.

Command:

```powershell
.venv\Scripts\yolo.exe detect train `
  model=yolov8n.pt `
  data=data/processed/training/fisheye8k_yolo_quick/data.yaml `
  epochs=1 `
  imgsz=640 `
  batch=4 `
  device=0 `
  workers=0 `
  project=outputs/training `
  name=fisheye8k_yolov8n_quick `
  exist_ok=True
```

Result:

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3060 12GB |
| Dataset | FishEye8K quick subset |
| Train images | 80 |
| Validation images | 80 |
| Epochs | 1 |
| mAP50 | 0.00265 |
| mAP50-95 | 0.00194 |
| Inference speed | about 2.3 ms/image |

This result is expected to be poor because the run uses a tiny subset and only
one epoch. It is a smoke test for the training path.

## Why The Earlier Estimate Was Higher

The estimate in `docs/training_plan.md` assumes:

- full FishEye8K scale, about 8,000 images
- about 157K boxes
- transfer learning for a normal training schedule
- enough epochs for convergence

The sanity run used:

- 80 train images
- 80 validation images
- one epoch

So the sanity result should not be compared against the full-data estimate.

## Comparison Matrix

| Model family | Role | Why it matters |
| --- | --- | --- |
| YOLOv8n / YOLO small | Baseline | Fast, familiar, easy edge deployment. |
| RT-DETR / RT-DETR v2 | Main detector candidate | Transformer detector with strong real-time detection behavior. |
| D-FINE | Main detector candidate | Modern real-time detector family; better project emphasis than YOLO-only. |
| SAM/SAM2 | Label and evidence assistant | Helps generate masks for vehicles, red lines, curb zones, and occlusion reasoning. |
| VLM | Review stage | Explains candidate events and helps human-review prioritization. |

## Open-Source Goal

The repo should let another country do the same workflow:

1. Replace the dataset manifest with local traffic-camera images.
2. Convert labels into YOLO or COCO.
3. Train YOLO, RT-DETR, or D-FINE.
4. Use SAM/SAM2 for masks if local road markings differ.
5. Validate against local violation records or crash hot spots.

That is the real reusable contribution: not a Taiwan-only red-line detector,
but a framework for building local traffic-violation models from public data.
