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

## Current Error Analysis

The first source-backed YOLOv8n run is useful because it exposes why a
YOLO-only project is weak:

| Class | mAP50 | Issue |
| --- | ---: | --- |
| Car | 0.517 | Usable baseline for vehicle detection. |
| Bike | 0.494 | Near usable, but still low for deployment. |
| Bus | 0.479 | Detectable, but needs stronger localization. |
| Truck | 0.122 | Weak rare-class performance. |
| Pedestrian | 0.030 | Fails on small/vulnerable-road-user class. |

The generated analysis report identifies:

- best mAP50 epoch: 58
- best mAP50: 0.32968
- best mAP50-95: 0.17529
- weak classes: Pedestrian, Truck
- instance imbalance ratio: about 57.9x

This supports the next experiment:

```text
YOLOv8n baseline is fast but not enough
  -> train RT-DETR / D-FINE for stronger real-time detection
  -> use SAM/SAM2 to add mask-level rule evidence
  -> compare event precision/recall, not only detector mAP
```

## RT-DETR Baseline Result

The first non-YOLO detector baseline used the same FishEye8K 1K/1K subset as
the YOLOv8n run. This makes the comparison useful even though it is not yet a
full-dataset result.

RT-DETR is a strong candidate because Ultralytics describes it as a real-time,
end-to-end transformer detector with hybrid encoder design, anchor-free
detection, and NMS-free inference. The local run used the official
`rtdetr-l.pt` checkpoint, `imgsz=640`, and a 30-epoch transfer-learning schedule.

Command:

```powershell
.venv\Scripts\yolo.exe detect train `
  model=rtdetr-l.pt `
  data=data/processed/training/fisheye8k_yolo_1k/data.yaml `
  epochs=30 `
  imgsz=640 `
  batch=2 `
  device=0 `
  workers=0 `
  patience=15 `
  project=outputs/training `
  name=fisheye8k_rtdetr_l_1k_e30 `
  exist_ok=True
```

Best-checkpoint validation:

| Model | Epochs | Precision | Recall | mAP50 | mAP50-95 | Inference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n | 100 | 0.430 | 0.350 | 0.329 | 0.175 | 3.9 ms/image |
| RT-DETR-L | 30 | 0.641 | 0.498 | 0.533 | 0.298 | 29.0 ms/image |

Per-class RT-DETR-L result:

| Class | mAP50 | mAP50-95 |
| --- | ---: | ---: |
| Bus | 0.662 | 0.439 |
| Bike | 0.560 | 0.240 |
| Car | 0.721 | 0.417 |
| Pedestrian | 0.128 | 0.053 |
| Truck | 0.593 | 0.340 |

Interpretation:

- RT-DETR-L gives a much stronger detector baseline than YOLOv8n on this
  subset: `+0.204` mAP50 and `+0.123` mAP50-95.
- The gain is not free. RT-DETR-L is about 7.4x slower per image on this local
  validation run.
- The weak class remains pedestrian. This points to data balance, small-object
  handling, and SAM/SAM2-assisted mask or crop evidence as the next meaningful
  improvements.

This is now the clearest project claim:

```text
YOLOv8n proves the edge baseline.
RT-DETR-L proves the stronger detector path.
SAM/SAM2 should improve evidence quality and labels around road markings.
The final system must choose a detector based on event accuracy and edge cost,
not popularity.
```

## SAM/SAM2 Evidence Path

SAM is not used as a detector replacement in this project. Its role is to turn
a detector bbox into better rule evidence:

```text
YOLO / RT-DETR bbox
  -> SAM/SAM2 prompt mask
  -> vehicle-mask overlap with red-line or no-parking region
  -> event-level evidence fields
```

The current implementation includes:

- bbox-mask fallback for environments without SAM weights
- optional `SamPromptMaskProvider` for Meta Segment Anything checkpoints
- event JSON fields for mask source, vehicle mask area, restricted overlap
  pixels, and restricted overlap ratio

The next SAM step is pseudo-labeling: use SAM/SAM2 to help create vehicle and
road-marking masks for a small annotated validation set, then compare
event-level precision/recall against the bbox-only baseline.

## Open-Source Goal

The repo should let another country do the same workflow:

1. Replace the dataset manifest with local traffic-camera images.
2. Convert labels into YOLO or COCO.
3. Train YOLO, RT-DETR, or D-FINE.
4. Use SAM/SAM2 for masks if local road markings differ.
5. Validate against local violation records or crash hot spots.

That is the real reusable contribution: not a Taiwan-only red-line detector,
but a framework for building local traffic-violation models from public data.
