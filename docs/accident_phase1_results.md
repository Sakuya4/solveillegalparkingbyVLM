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
  -> global and accident-ROI frame difference / optical flow
  -> mean, maximum, standard deviation, and peak-position features
  -> standardized logistic edge baseline
  -> IID or geographic OOD metrics and per-quality breakdown
```

This baseline is intentionally small and serializable. It verifies labels,
splits, feature extraction, and edge interfaces before TCN or VideoMAE training.

## Real-Video Smoke Benchmark

A deterministic stratified sample of 500 real clips (seed 42) produced 822
windows with zero decode failures. Sampling balances collision type and split
groups for pipeline coverage, so these numbers are not the official
distribution-weighted ACCIDENT benchmark.

| Split contract | Test windows | Accuracy | Precision | Recall | F1 | Window FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| IID | 450 | 0.820 | 0.913 | 0.775 | 0.838 | 0.112 |
| Geographic OOD | 410 | 0.827 | 0.877 | 0.831 | 0.854 | 0.180 |

Geographic OOD F1 did not fall on this sample. The cautious interpretation is
that abrupt motion is less location-dependent than appearance, not that domain
shift is solved. Rear-end incidents are the weakest class (IID F1 0.667;
geographic F1 0.786), while t-bone incidents are the strongest (IID F1 0.911;
geographic F1 0.971). This gives the next temporal model a concrete target.

The reported false-positive rate is window-level. Normal samples are
non-overlapping pre-incident windows from accident clips, not independent
normal CCTV footage, so false alarms per camera-hour cannot yet be claimed.

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

- Run the same experiment on the full official split without stratified
  subsampling.
- Add normal-only CCTV footage before reporting false alarms per camera-hour.
- Train TCN and VideoMAE comparison models on the same split contract.
- Add detector trajectory and time-to-collision features, especially for
  rear-end incidents.
