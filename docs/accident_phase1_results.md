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
  -> YOLO + ByteTrack and motion candidate ROI
  -> optional SAM2 box-prompt refinement
  -> aggregate logistic or causal TCN deployment baseline
  -> optional trajectory / image-plane TTC features
  -> accident-annotation ROI only as an oracle upper bound
  -> IID or geographic OOD metrics and per-quality breakdown
```

Deployment-facing results use only global or online candidate features available
at runtime. The ACCIDENT accident bounding box is never available before an
event, so annotation-ROI features remain an explicit oracle upper bound.

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
| Online candidate ROI logistic | 0.581 | 0.615 | Deployable |
| Global + candidate ROI logistic | 0.569 | 0.603 | Deployable fusion |
| Global causal TCN | **0.642** | 0.569 | Deployable |
| Online candidate ROI TCN | 0.629 | **0.668** | Deployable |
| Global + candidate ROI TCN | 0.628 | 0.597 | Deployable fusion |
| Oracle accident ROI motion | 0.838 | 0.854 | Upper bound only |
| Oracle ROI + trajectory/TTC | 0.854 | **0.872** | Upper bound only |

Trajectory extraction processed 500 clips in 339.29 seconds on the RTX 3060,
with zero failures. However, 103 of 822 windows had no stable vehicle track.
Image-plane TTC is therefore useful as complementary evidence, not a physical
TTC measurement or a replacement for motion. On IID data it raises the
deployment-safe global baseline from F1 0.405 to 0.562.

The annotation-free candidate ROI extractor processed the same 500 clips into
822 windows and `822 x 15 x 14` tensors in 407.00 seconds on the RTX 3060, with
zero failures. Candidate availability was 96.82%, with 4.41 proposals per step.
Ground-truth boxes were used only after extraction for proposal diagnostics:
recall@0.1 was 0.672 and recall@0.3 was 0.417. These diagnostics are absent from
the CSV, NPZ, and edge trace model inputs.

Candidate-only temporal features improve geographic TCN F1 from 0.569 to 0.668,
but simple global/candidate concatenation reaches only 0.597. The online ROI is
therefore useful evidence, while fusion and threshold calibration remain open
problems. All table values use a fixed 0.5 threshold; the relatively high window
FPR must be reported alongside F1.

## Frozen VideoMAE Comparison

`MCG-NJU/videomae-small-finetuned-kinetics` was used as a frozen 16-frame
encoder. A linear classifier was trained on its 384-dimensional embeddings
under the same IID/geographic split contract. Accident boxes, collision types,
regions, and test labels were not used as model inputs.

| Feature path | Split | F1 | Window FPR | Precision | Recall |
| --- | --- | ---: | ---: | ---: | ---: |
| Frozen VideoMAE | IID | **0.646** | 0.453 | 0.675 | 0.620 |
| Frozen VideoMAE | Geographic | 0.610 | 0.590 | 0.614 | 0.606 |
| Candidate ROI + frozen VideoMAE | IID | 0.634 | 0.436 | 0.681 | 0.598 |
| Candidate ROI + frozen VideoMAE | Geographic | **0.615** | 0.553 | 0.628 | 0.602 |

All 822 windows were extracted with zero failures on the RTX 3060. Extraction
took 562.42 seconds at 1.46 windows/s with 645.97 MB peak allocated CUDA memory
and batch size 16. The current Transformers release uses split query/key/value
bias tensors, so the legacy checkpoint q/v biases are deterministically mapped
and then loaded with strict state-dict validation.

This is a frozen-feature baseline, not end-to-end VideoMAE fine-tuning. It
slightly improves fixed-threshold IID F1 over candidate TCN (0.646 versus
0.629), but is worse on the geographic split (0.610 versus 0.668). The result
supports a multi-model comparison while rejecting the claim that a larger
video backbone is automatically more robust to camera-location shift.

## Train-Holdout Threshold Calibration

The operating threshold is selected only from a group-disjoint 20% holdout of
the training clips. The test split remains untouched. At a requested 0.20
calibration FPR, candidate-only results are:

| Model | Split | Threshold | Calibration FPR | Test FPR | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic | IID | 0.599 | 0.179 | 0.207 | 0.670 | 0.277 | 0.392 |
| Logistic | Geographic | 0.669 | 0.180 | 0.280 | 0.628 | 0.305 | 0.411 |
| Candidate TCN | IID | 0.855 | 0.179 | 0.235 | 0.698 | 0.358 | 0.473 |
| Candidate TCN | Geographic | 0.740 | 0.180 | 0.304 | 0.669 | 0.398 | 0.499 |
| Frozen VideoMAE | IID | 0.865 | 0.179 | 0.196 | 0.715 | 0.325 | 0.447 |
| Frozen VideoMAE | Geographic | 0.879 | 0.180 | 0.217 | 0.639 | 0.249 | 0.358 |
| Candidate + VideoMAE | IID | 0.871 | 0.179 | **0.162** | 0.770 | 0.358 | **0.489** |
| Candidate + VideoMAE | Geographic | 0.976 | 0.180 | 0.118 | 0.661 | 0.149 | 0.243 |

Calibration substantially reduces test FPR compared with the fixed 0.5
operating point, but also lowers recall and F1. Geographic FPR remains above the
training budget, which is direct evidence of domain shift rather than a reason
to retune on the test set.

Candidate/VideoMAE fusion gives the best calibrated IID result in this ablation,
raising F1 from 0.473 to 0.489 while lowering FPR from 0.235 to 0.162. Its
geographic threshold becomes over-conservative and recall falls to 0.149, so it
is not selected as a deployment operating point.

The reported false-positive rate is window-level. Normal samples are
non-overlapping pre-incident windows from accident clips, not independent
normal CCTV footage, so false alarms per camera-hour cannot yet be claimed.

The TCN layers are causal, but each positive window is centered on the annotated
accident frame and contains post-event frames. These results measure event
detection and confirmation, not accident anticipation or early warning.

## Reproduce Online Candidate ROI

The local archive used here is extracted under
`data/raw/accident/full/extracted`. Ultralytics accepts GPU index `0`, while the
PyTorch TCN CLI uses `auto` or `cuda`.

```powershell
& .\.venv\Scripts\python.exe scripts/extract_accident_candidate_roi_features.py `
  --metadata data/raw/accident/full/extracted/metadata-real.csv `
  --dataset-root data/raw/accident/full/extracted `
  --max-clips 500 --device 0 `
  --output data/processed/accident/candidate_roi_features_500.csv `
  --sequence-output data/processed/accident/candidate_roi_sequences_500.npz

& .\.venv\Scripts\python.exe scripts/train_incident_motion_baseline.py `
  --features data/processed/accident/candidate_roi_features_500.csv `
  --include-prefix candidate_

& .\.venv\Scripts\python.exe scripts/train_incident_tcn.py `
  --sequences data/processed/accident/candidate_roi_sequences_500.npz `
  --include-prefix candidate_ --device auto
```

Add `--sam2-model sam2.1_t.pt --sam2-device 0` to extraction only when SAM2
latency and proposal quality are being measured. The default benchmark keeps
SAM2 disabled so tracker/motion remains the edge baseline.

## Public CCTV Pilot

On 2026-07-17, the official Taichung C000129 page resolved to a live 1280x720
MJPEG stream. The server advertised 25 FPS but delivered about 2.7 unique frames
per wall-clock second during capture. The first recorded artifact contains 83
frames normalized to 7.5 FPS, or 11.07 seconds of model time.

The calibrated IID candidate TCN marked 2 of 5 non-overlapping windows positive.
Both were isolated, so a two-consecutive-window event gate reduced the result
from 2 raw positive windows to 0 review events. The resulting 0 events over
0.0031 camera-hours is only a connectivity/domain-shift pilot, not a stable
false-alert estimate. More normal-only hours and manual incident annotations
are still required.

For zero observed events, the one-sided 95% Poisson upper bound is 974.52 false
alerts per camera-hour because the exposure is only 11.07 seconds. The
multi-session aggregator preserves the model, threshold, and persistence gate
contract and combines future sessions without presenting `0/hour` as proof of
low false alarms.

```powershell
python scripts/aggregate_normal_cctv_reports.py `
  --report-glob "outputs/cctv/*_model_output.json"
```

## Incident Evidence And VLM Review

The annotation-free model output can now be converted into an ordered
before/trigger/after evidence package. The real demo selected frames 53, 68,
and 83 around the first two-window-confirmed event. The package records model
probability 0.9990, threshold 0.8547, the privacy method, and the fact that the
inference path did not read accident annotations.

```powershell
python scripts/prepare_incident_vlm_review.py `
  --inference-report outputs/accident/candidate_incident_demo.json `
  --annotated-video outputs/accident/candidate_incident_demo.mp4 `
  --output-dir outputs/accident/incident_vlm_evidence

python scripts/run_vlm_review.py `
  --request-json outputs/accident/incident_vlm_evidence/vlm_review_request.json `
  --output outputs/accident/incident_vlm_evidence/offline_review_result.json
```

The deterministic offline reviewer confirms only the score/persistence gate;
it is not a visual-language model result and always requests human review. A
real VLM comparison requires an annotated incident-review set. Plate redaction
is currently heuristic inside tracked vehicle boxes and must be upgraded before
government deployment.

## Event-Level Timing Evaluation

`evaluate_incident_reports.py` evaluates completed annotation-free inference
reports against ACCIDENT timing metadata. The trigger timestamp is the end of
the sequence window, when all model inputs are actually available; the center
frame used for visual evidence is not reported as decision latency. ACCIDENT
metadata is loaded only by this post-inference evaluator.

```powershell
python scripts/evaluate_incident_reports.py `
  --metadata data/raw/accident/full/extracted/metadata-real.csv `
  --report outputs/accident/candidate_incident_demo.json `
  --normal-report outputs/cctv/taichung_c000129_model_output.json `
  --early-tolerance-sec 1.0 `
  --late-tolerance-sec 3.0 `
  --output outputs/accident/incident_event_evaluation_demo.json
```

For the existing t-bone demo, the annotated accident time is 9.208 seconds and
the first valid two-window trigger becomes available at 9.9998 seconds, giving
a measured delay of +0.792 seconds. This is a one-clip integration result
(`n=1`), not a dataset-level recall claim. Normal CCTV false alerts per hour and
their exposure bound remain a separate section of the same report.

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

The annotation-free 500-clip candidate trace was also replayed directly instead
of using a synthetic 1% event rate. At candidate score >= 0.95, 1,702 of 12,330
source steps escalated. Replicated across four 7.5 FPS cameras, the Python golden
model processed all 49,320 frames with 0 drops and 20.7% NPU utilization. The
300 ms review stage received 6,808 candidates and dropped 4,652 (68.3%), showing
that review capacity, not detector throughput, is the current bottleneck.

`npu_queue_sim.cpp` accepts the same generated flags through
`--candidate-flags`. The local machine has CMake and MSVC, but not the SystemC
SDK (`SystemCLanguageConfig.cmake`), so this round verifies the executable
workload with the Python golden model while recording the SystemC build blocker.

## Remaining Before Model Claims

- Improve online candidate localization and reduce the calibrated recall/FPR
  tradeoff.
- Extend the 11.07-second public-CCTV pilot to independently reviewed normal
  footage measured in camera-hours.
- Fine-tune the final VideoMAE encoder block and compare detector front ends on
  the same split contract.
- Run the event-level evaluator over a representative IID/geographic report set
  to measure recall and trigger-delay distributions.

## Method Sources

- ACCIDENT benchmark and official heuristic baselines:
  https://github.com/accidentbench/ACCIDENT
- Ultralytics tracking and ByteTrack configuration:
  https://docs.ultralytics.com/modes/track/
- Ultralytics SAM2 box-prompt interface:
  https://docs.ultralytics.com/models/sam-2
- VideoMAE implementation and checkpoint:
  https://huggingface.co/docs/transformers/model_doc/videomae and
  https://huggingface.co/MCG-NJU/videomae-small-finetuned-kinetics
- Causal dilated temporal convolution starting point:
  https://arxiv.org/abs/1803.01271
