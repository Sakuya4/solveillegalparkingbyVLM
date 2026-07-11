# Project Direction

## Selected Direction

Build an edge-first traffic incident detection and risk-prioritization system for
fixed roadside cameras.

Red-line parking is now a completed event case. The next mainline is traffic
accident detection: identify when and where an incident happens, classify the
event, create evidence, and prioritize the alert using local road-safety data.

## Public Training Dataset

The primary dataset is [ACCIDENT](https://github.com/accidentbench/ACCIDENT), a
fixed-view CCTV benchmark with 2,027 real accident clips and 2,211 CARLA clips.
Its labels cover accident time, image location, collision type, geographic
split, weather, scene, quality, and day/night conditions.

- Kaggle dataset: https://www.kaggle.com/datasets/picekl/accident
- License: CC BY-NC-SA 4.0
- Verified size: 16,758,864,720 bytes
- Local access: Kaggle authentication and metadata download verified

[TUMTraf-A](https://tum-traffic-dataset.github.io/tumtraf-a/) is a secondary
roadside-sensor validation source for trajectories, track IDs, masks, camera and
LiDAR data. It is not required for the first training run.

## System Contribution

The contribution is not another vehicle detector. It is a layered event system:

```text
roadside camera
  -> detector + tracker
  -> motion/trajectory features
  -> temporal incident model
  -> SAM2 evidence localization
  -> VLM classification and explanation
  -> human review / response alert
  -> Taiwan hot-spot priority and operational report
```

The model comparison therefore has three levels:

1. Detector front end: YOLO, RT-DETR, and D-FINE.
2. Temporal event model: feature-based TCN/LSTM baseline versus VideoMAE.
3. Event reviewer: deterministic rules versus local VLM.

## Government Data Role

Taiwan government data supplements training instead of pretending to be image
labels:

- A1/A2 records define accident hot spots and deployment coverage.
- Violation records add a behavioral-risk prior.
- VD traffic volume normalizes risk by road exposure.
- Public CCTV validates runtime, image quality, and domain shift.

The system may claim faster detection, better hot-spot coverage, and lower
manual monitoring load. It must not claim that A1/A2 casualties were reduced
until a real before/after intervention study exists.

## Two Phases

### Phase 1: Public-Dataset Incident Model

- Download and validate ACCIDENT.
- Reproduce its metadata and heuristic smoke baselines.
- Extract detector/tracker motion features around `accident_time`.
- Train a lightweight temporal baseline and a VideoMAE comparison.
- Evaluate IID and geographic OOD splits with event recall, false alarms per
  hour, temporal localization error, collision-type accuracy, and latency.
- Reuse the existing evidence, SAM, VLM, and privacy pipeline for incident
  packages.

### Phase 2: Edge And Hardware-Aware Validation

- Export and quantize the chosen detector and temporal model.
- Add Verilog motion, overlap, and time-to-collision trigger blocks.
- Add a SystemC camera-to-NPU queue and throughput simulation.
- Join predictions with A1/A2, violations, VD flow, and CCTV locations.
- Report camera coverage, alert delay, dropped-frame rate, NPU utilization,
  false alarms per camera-hour, and estimated review savings.

Snapdragon NPU behavior will initially be represented by measured/profiled
latency traces. Cycle-accurate simulation of Qualcomm's proprietary NPU is not
claimed without the required SDK, model compiler, and target hardware.
