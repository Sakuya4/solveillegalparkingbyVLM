# ACCIDENT Phase 1 Results

## Dataset Verification

- Public benchmark: https://github.com/accidentbench/ACCIDENT
- Downloaded archive: 16,758,864,720 bytes
- SHA-256: `420D79FF1B5C5E82E93F4D8F8BE7F43509943D5844218CCBBBD977C34C26AAE6`
- Real CCTV clips: 2,027
- Synthetic CARLA clips: 2,211
- Real video duration: 44,286.16 seconds (about 12.30 hours)
- IID split: 507 train, 1,520 test
- Geographic split: 454 train, 1,573 test
- Collision classes: 117 head-on, 328 rear-end, 245 sideswipe, 680
  single-vehicle, and 657 t-bone clips

Two official rows use `accident_frame == no_frames`. The adapter preserves the
source value for audit and uses `no_frames - 1` as the effective frame index.

With 32-frame windows and an 8-frame non-overlap guard, the current metadata
produces 3,387 training windows: 1,366 pre-incident normal windows and 2,021 incident windows.
Six clips shorter than one full window are excluded.

The archive inventory contains all 2,027 real and 2,211 synthetic MP4 files.
The current experiment extracted and file-validated all real clips; synthetic
clips remain in the verified archive until the domain-gap experiment.

## Implemented Training Path

```text
ACCIDENT metadata + video
  -> validated temporal windows
  -> global frame difference / optical flow sequence
  -> causal TCN deployment baseline
  -> optional YOLO + ByteTrack trajectory / image-plane TTC features
  -> accident-annotation ROI only as an oracle upper bound
  -> IID or geographic OOD metrics and per-quality breakdown
```

All deployment-facing results use only global image features available at
runtime. The ACCIDENT accident bounding box is never available before an event,
so ROI-based results are reported only as an oracle upper bound.

## Full Real-Video Benchmark

All 2,027 real clips produced 3,387 windows with zero decode failures. Motion
aggregation at 320 px took 1,782.45 seconds. The 160 px sequence tensors took
1,034.07 seconds and have shape `3387 x 15 x 8`; the deployment TCN selects the
four global channels and excludes all four annotation-ROI channels.

| Model | Split | Accuracy | Precision | Recall | F1 | Window FPR |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Global motion logistic | IID | 0.521 | 0.683 | 0.357 | 0.469 | 0.241 |
| Global motion logistic | Geographic | 0.534 | 0.667 | 0.434 | 0.526 | 0.319 |
| Global causal TCN | IID | 0.577 | 0.688 | 0.522 | **0.594** | 0.345 |
| Global causal TCN | Geographic | 0.585 | 0.672 | 0.592 | **0.629** | 0.426 |
| Oracle accident-ROI logistic | IID | 0.850 | 0.904 | 0.836 | 0.869 | 0.129 |
| Oracle accident-ROI logistic | Geographic | 0.847 | 0.885 | 0.854 | 0.870 | 0.163 |

The causal TCN improves deployable global-feature F1 by 0.125 IID and 0.103
geographic. The much larger oracle-ROI gap is not a deployable achievement; it
quantifies the value of the next research target: producing a reliable online
candidate ROI from tracking, motion proposals, or SAM2.

## 500-Clip Ablation

A deterministic stratified sample of 500 real clips (seed 42) produced 822
windows with zero decode failures. Sampling balances collision type and split
groups for pipeline coverage, so these numbers are not the official
distribution-weighted ACCIDENT benchmark.

| Feature path | IID F1 | Geographic F1 | Runtime interpretation |
| --- | ---: | ---: | --- |
| Global motion | 0.405 | 0.515 | Deployable |
| YOLOv8n + ByteTrack trajectory/TTC | 0.532 | 0.575 | Deployable proxy |
| Global motion + trajectory/TTC | **0.562** | 0.569 | Deployable fusion |
| Oracle accident ROI motion | 0.838 | 0.854 | Upper bound only |
| Oracle ROI + trajectory/TTC | 0.854 | **0.872** | Upper bound only |

Trajectory extraction processed 500 clips in 339.29 seconds on the RTX 3060,
with zero failures. However, 103 of 822 windows had no stable vehicle track.
Image-plane TTC is therefore useful as complementary evidence, not a physical
TTC measurement or a replacement for motion. On IID data it raises the
deployment-safe global baseline from F1 0.405 to 0.562.

The reported false-positive rate is window-level. Normal samples are
non-overlapping pre-incident windows from accident clips, not independent
normal CCTV footage, so false alarms per camera-hour cannot yet be claimed.

The TCN layers are causal, but each positive window is centered on the annotated
accident frame and contains post-event frames. These results measure event
detection and confirmation, not accident anticipation or early warning.

## Edge Queue Result

The queue model used four cameras at 15 FPS, an 8-frame NPU queue, 1% candidate
rate, and a 3 ms temporal stage.

| Detector path | Cameras | Frame drop | NPU utilization | p95 latency |
| --- | ---: | ---: | ---: | ---: |
| YOLOv8n measured 3.9 ms | 4 | 0.0% | 41.4% | 27.6 ms |
| RT-DETR-L measured 29 ms | 2 | 0.0% | 96.0% | 64.0 ms |
| RT-DETR-L measured 29 ms | 3 | 30.3% | 100.0% | 285.3 ms |
| RT-DETR-L measured 29 ms | 4 | 47.8% | 100.0% | 285.3 ms |

This is a queue simulation using measured desktop inference latency, not a
Snapdragon power or cycle-accurate result. It establishes the testable reason
for motion-triggered inference, model quantization, and multi-rate scheduling.

## Remaining Before Model Claims

- Replace oracle accident ROIs with online tracker/motion/SAM2 candidate ROIs.
- Add normal-only CCTV footage before reporting false alarms per camera-hour.
- Train VideoMAE and detector-front-end comparisons on the same split contract.
- Measure event localization error and trigger delay, not only window labels.

## Method Sources

- ACCIDENT benchmark and official heuristic baselines:
  https://github.com/accidentbench/ACCIDENT
- Ultralytics tracking and ByteTrack configuration:
  https://docs.ultralytics.com/modes/track/
- Causal dilated temporal convolution starting point:
  https://arxiv.org/abs/1803.01271
