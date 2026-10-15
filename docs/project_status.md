# Project Status

## Current Position

The repository provides a native desktop research workbench for edge-first
traffic incident detection and government-informed camera allocation. Red-line parking is the
completed rule-based event case. ACCIDENT-based temporal incident detection is
the public-dataset training and evaluation mainline.

Completion is reported by evidence strength instead of a single percentage:

| State | Count | Meaning |
| --- | ---: | --- |
| Verified | 11 | Reproducible implementation with checked-in result or executable test |
| Pilot | 3 | Real execution completed, but sample or exposure is still limited |
| External | 3 | Requires hardware, credentials, long-duration video, or field data |

The same matrix is generated from checked-in artifacts for the integrated
[project Dashboard](../dashboard/index.html).

## Verified

- [x] Native PySide6 application: playback, asynchronous analysis, cancellation,
  event navigation, export, and portable ONNX execution.
- [x] Temporal government-data allocation: 180 matched-budget comparisons with
  January--September selection and October--December evaluation.
- [x] Red-line parking rule, two-sided curb interpretation, privacy treatment,
  SAM evidence, and event artifacts.
- [x] Annotation-free accident candidate proposal and causal TCN event trigger.
- [x] TCN, frozen VideoMAE, and partial VideoMAE fine-tuning comparison on the
  same 500 clips, splits, windows, and holdout-calibrated operating point.
- [x] Taiwan A1/A2, violation, and public-camera spatial integration.
- [x] Three self-checking Verilog trigger testbenches.
- [x] SystemC 3.0.2 multi-camera queue replay with Python parity.
  The measured-service extension verifies all five routing policies locally.
- [x] TCN ONNX export and 128-sample parity verification.
- [x] Official NAFNet restoration on a privacy-safe difficult-frame case.
- [x] Official SM3Det release, compute, license, and CCTV transfer audit.

## Pilot Complete

- [x] SAM2.1-t same-sample proposal ablation on 10 clips.
- [x] Qwen2.5-VL-3B and LLaVA/Qwen review on 20 paired evidence sequences,
  with matched greedy decoding and three/eight-frame density analysis.
- [x] Two normal government CCTV sessions totaling 0.0867 camera-hours.

## External Validation Required

- [ ] Accumulate independently reviewed normal CCTV camera-hours and report a
  useful false-alert exposure bound.
- [ ] Submit/profile the ONNX model on QNN after QAI Hub credentials or a
  Snapdragon device are available.
- [ ] Measure before/after intervention effects with repeated government data;
  projected A1/A2 reduction and cost savings remain planning scenarios.

## Measured Results

- Ten-camera held-out risk coverage: greedy `15.49%`, independent ranking
  `13.69%`, and random mean `11.38%` at A1:A2 5:1 and radius 1 km.
- Responsive event recall: `0.750` over 100 IID accident clips.
- Trigger delay: median `0.394s`, p95 `2.251s`.
- Holdout-calibrated geographic F1: TCN `0.499`, frozen VideoMAE `0.358`,
  partial fine-tuned VideoMAE `0.382`.
- SystemC configured 300 ms workload: 49,320 frames, zero frame drops,
  6,808 candidates, and 2,156 completed reviews.
- Measured Qwen service-time routing: 198 completed, three pending, zero queue
  drops, and 1,367 budget denials after coalescing; Python/SystemC agree.
- Full cohort: 2,021 eligible clips; candidate/global TCN IID F1 0.346/0.521
  and geographic F1 0.395/0.357. Full-cohort VideoMAE extraction is active.
- Existing New Taipei camera sites cover `18.95%` of joined top-risk score
  within 1 km.

See [the formal evaluation report](final_evaluation_report.md) for methodology,
limitations, and the complete tables. Remaining research tasks are tracked in
[todo.md](todo.md).
