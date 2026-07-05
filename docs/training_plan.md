# Training Plan

## Current Slice

The project now has a training planner for FishEye8K and FE-DETRAC style YOLO
datasets.

Run:

```powershell
python scripts\plan_training.py `
  --dataset-id fisheye8k `
  --model-size yolov8n `
  --image-size 640
```

It creates:

- `data/processed/training/fisheye8k.yaml`
- `data/processed/training/fisheye8k_training_plan.json`

Both outputs are generated artifacts and are intentionally ignored by git.

## Expected Result Before Training

Before downloading the full image dataset, the planner uses public dataset
metadata and project baselines to produce an estimated result:

| Item | Current estimate |
| --- | --- |
| Detector | YOLOv8n fine-tuned on FishEye8K-style YOLO data |
| Dataset scale | 8,000 images, about 157K boxes |
| Expected mAP50 | 0.65 to 0.79 |
| Expected event precision | 0.55 to 0.73 |
| Expected edge FPS | 12 to 28 FPS |
| Expected training time | 2 to 5 hours |

These values are not final experiment results. They are planning estimates.
They are allowed before training because:

- the dataset scale and class set are public
- prior traffic-camera YOLO benchmarks give a reasonable performance range
- the project already has live-FPS, event, violation hot-spot, and A1/A2 risk baselines
- the report marks the result as `estimated_before_training`

After training, this table must be replaced with measured validation mAP,
event precision/recall, trigger delay, edge FPS, and hot-spot coverage.

## Training Command

The generated report includes the command:

```powershell
yolo detect train model=yolov8n.pt data=data/processed/training/fisheye8k.yaml epochs=100 imgsz=640
```

For stronger accuracy, test `yolov8s.pt` or a newer Ultralytics small model.
For edge deployment, keep `yolov8n.pt` as the baseline because it is easier to
profile on CPU, NPU, or Snapdragon-class hardware.

## Source-Backed YOLO Baseline

The first full YOLO baseline is not an arbitrary setting. It is grounded in:

- FishEye8K paper: the dataset is designed for fisheye road-object detection
  and reports benchmark experiments with YOLOv5, YOLOR, YOLOv7, and YOLOv8.
- FishEye8K paper: YOLOv8 is reported as a strong model at `640x640` input.
- Ultralytics training documentation: `model`, `data`, `epochs`, `batch`,
  `imgsz`, `device`, `workers`, `project`, and `name` are standard training
  arguments.
- Ultralytics configuration documentation: default training examples use
  `epochs=100` and `imgsz=640`.
- Ultralytics training tips: pretrained weights, GPU-aware batch sizing, AMP,
  and early stopping are recommended training practices.

Reference URLs:

- FishEye8K: https://arxiv.org/abs/2305.17449
- FishEye8K GitHub: https://github.com/MoyoG/FishEye8K
- Ultralytics train mode: https://docs.ultralytics.com/modes/train/
- Ultralytics train settings: https://docs.ultralytics.com/usage/cfg/#train-settings
- Ultralytics training tips: https://docs.ultralytics.com/guides/model-training-tips/

Source-backed first baseline:

```powershell
.venv\Scripts\yolo.exe detect train `
  model=yolov8n.pt `
  data=data/processed/training/fisheye8k_yolo_full/data.yaml `
  epochs=100 `
  imgsz=640 `
  batch=-1 `
  device=0 `
  workers=0 `
  patience=50 `
  project=outputs/training `
  name=fisheye8k_yolov8n_full_e100 `
  exist_ok=True
```

Why not start with 300 epochs? Ultralytics training tips mention 300 epochs as
a typical starting point, but the configuration docs also define `epochs=100`
as the standard train-setting default. On a local RTX 3060, the first full
baseline should finish in a reasonable time and produce a measured result. If
the validation curve is still improving, resume or extend the run.

## First Source-Backed Baseline Result

The first source-backed run used a 1K/1K FishEye8K subset because the full image
download exceeded the first one-hour data preparation window. It is still a
meaningful baseline because it uses a larger validation set and a full
100-epoch training schedule.

Command:

```powershell
.venv\Scripts\python.exe scripts\prepare_fisheye8k_yolo.py `
  --download-missing `
  --max-per-split 1000 `
  --output data/processed/training/fisheye8k_yolo_1k

.venv\Scripts\yolo.exe detect train `
  model=yolov8n.pt `
  data=data/processed/training/fisheye8k_yolo_1k/data.yaml `
  epochs=100 `
  imgsz=640 `
  batch=-1 `
  device=0 `
  workers=0 `
  patience=50 `
  project=outputs/training `
  name=fisheye8k_yolov8n_1k_e100 `
  exist_ok=True
```

Dataset:

| Split | Images |
| --- | ---: |
| Train | 1000 |
| Val/Test | 1000 |
| Boxes | 43991 |

Best validation result:

| Metric | Value |
| --- | ---: |
| Precision | 0.430 |
| Recall | 0.350 |
| mAP50 | 0.329 |
| mAP50-95 | 0.175 |
| Inference | 3.9 ms/image |

Per-class mAP50:

| Class | mAP50 |
| --- | ---: |
| Bus | 0.479 |
| Bike | 0.494 |
| Car | 0.517 |
| Pedestrian | 0.030 |
| Truck | 0.122 |

Interpretation: YOLOv8n learns vehicles on the FishEye8K subset, but it is weak
on pedestrian and truck classes. This is a useful baseline for showing why the
project should compare D-FINE / RT-DETR and add SAM/SAM2-assisted evidence
rather than stopping at YOLO.

## Local GPU Setup

The current local training environment uses a project-local virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python.exe -m pip install ultralytics huggingface_hub pyyaml
```

Verified hardware:

```text
NVIDIA GeForce RTX 3060, 12GB VRAM
torch 2.11.0+cu128
CUDA available: True
```

## FishEye8K Conversion

The Hugging Face FishEye8K mirror is stored as FiftyOne samples. Convert a
quick subset with:

```powershell
.venv\Scripts\python.exe scripts\prepare_fisheye8k_yolo.py `
  --download-missing `
  --max-per-split 80 `
  --output data/processed/training/fisheye8k_yolo_quick
```

The converter writes YOLO labels and a `data.yaml`. If the dataset has no
validation split, it uses `test` as `val` for Ultralytics training.
