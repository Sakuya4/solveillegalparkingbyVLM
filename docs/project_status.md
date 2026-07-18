# Project Status

## Current Position

The repository is an integrated, demo-ready research prototype for edge-first
traffic incident detection and risk prioritization. Red-line parking is the
completed rule-based event case. ACCIDENT-based temporal incident detection is
the public-dataset training and evaluation mainline.

Completion is reported by evidence strength instead of a single percentage:

| State | Count | Meaning |
| --- | ---: | --- |
| Verified | 9 | Reproducible implementation with checked-in result or executable test |
| Pilot | 3 | Real execution completed, but sample or exposure is still limited |
| External | 3 | Requires hardware, credentials, long-duration video, or field data |

The same matrix is generated from checked-in artifacts for the integrated
[project Dashboard](../dashboard/index.html).

## Verified

- [x] Red-line parking rule, two-sided curb interpretation, privacy treatment,
  SAM evidence, and event artifacts.
- [x] Annotation-free accident candidate proposal and causal TCN event trigger.
- [x] TCN, frozen VideoMAE, and partial VideoMAE fine-tuning comparison on the
  same 500 clips, splits, windows, and holdout-calibrated operating point.
- [x] Taiwan A1/A2, violation, and public-camera spatial integration.
- [x] Three self-checking Verilog trigger testbenches.
- [x] SystemC 3.0.2 multi-camera queue replay with Python parity.
- [x] TCN ONNX export and 128-sample parity verification.
- [x] Official NAFNet restoration on a privacy-safe difficult-frame case.
- [x] Official SM3Det release, compute, license, and CCTV transfer audit.

## Pilot Complete

- [x] SAM2.1-t same-sample proposal ablation on 10 clips.
- [x] Qwen2.5-VL-3B and LLaVA/Qwen clean-blind evidence review cases.
- [x] Two normal government CCTV sessions totaling 0.0867 camera-hours.

## External Validation Required

- [ ] Accumulate independently reviewed normal CCTV camera-hours and report a
  useful false-alert exposure bound.
- [ ] Submit/profile the ONNX model on QNN after QAI Hub credentials or a
  Snapdragon device are available.
- [ ] Measure before/after intervention effects with repeated government data;
  projected A1/A2 reduction and cost savings remain planning scenarios.

## Measured Results

- Responsive event recall: `0.750` over 100 IID accident clips.
- Trigger delay: median `0.394s`, p95 `2.251s`.
- Holdout-calibrated geographic F1: TCN `0.499`, frozen VideoMAE `0.358`,
  partial fine-tuned VideoMAE `0.382`.
- SystemC workload: 49,320 frames, zero frame drops, 6,808 candidates, and
  2,156 completed VLM reviews.
- Existing New Taipei camera sites cover `18.95%` of joined top-risk score
  within 1 km.

See [the formal evaluation report](final_evaluation_report.md) for methodology,
limitations, and the complete tables. Remaining research tasks are tracked in
[todo.md](todo.md).
