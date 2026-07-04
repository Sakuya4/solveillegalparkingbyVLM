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
