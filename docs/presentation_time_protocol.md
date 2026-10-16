# Presentation-Time Matched Temporal Study

## Pre-Fitting Specification

This study compares a source-frame context policy with a bounded-jitter
presentation-time policy. The former retains 32-frame windows and stride two;
the latter uses 16 distinct images on a global 7.5-Hz target grid. Both produce
15 optical-flow steps with unchanged candidate/global feature definitions.
The nominal target span is 2 s. Each selected frame is the first source frame
at or after its target time, with lateness at most one target period and a
1-microsecond numerical comparison tolerance. Duplicate frames are rejected;
no interpolation or artificial observations are introduced. The actual span
therefore remains bounded by 2 s plus or minus one target period.

Ordered packet presentation timestamps are normalized to the first source
timestamp. Packet count must equal the container frame count. Before a clip
enters the paired manifest, the union of all selected frame indices and the
supplied annotation frame is decoded and checked against frame identity and
presentation time within 0.1 ms. This verifies the actual model-input support,
not only an average-FPS metadata approximation.

## Labels and Eligibility

The original supplied annotation frame defines event time on the verified
presentation clock. Legacy boundary clamping is recorded separately and is
not applied to the new ground-truth anchor. Container counts equal to the
metadata count or one greater are supported; other count discrepancies are
excluded. Both policies must observe the event in their positive window.
Eligible negative windows must be available under both policies.
Both arms are re-extracted under verified container support; the original
reference tensors are not reused or relabeled.

The presentation-grid event window begins eight target slots before its
event target slot, clamped only to available grid support. Its negative window
ends four target slots before that start. The source-frame policy retains its
eight-frame negative gap. The contrast thus jointly changes sampling, physical
context and pre-event gap. It is not an isolated architecture or cadence effect.
Normal labels denote pre-event windows within accident clips, not independent
normal-camera observations.

## Frozen Estimation Contracts

Official IID and geographic train/test memberships remain unchanged. A
seed-42, 20% clip-grouped holdout within each eligible training split determines
calibration membership. Both arms use identical rows and partitions. Their
candidate/global TCNs retain 32 channels, dropout 0.1, batch size 64, 40 epochs,
AdamW learning rate 0.001, weight decay 0.0001 and fitting-only standardization.
Spatial width and detector image size remain 320, with confidence 0.15 and
ByteTrack. Thresholds target a calibration-window FPR of 0.2.

Primary reporting uses paired source-clip window-F1 contrasts and 2,000
bootstrap resamples with seed 42. AUROC, average precision, actual test FPR
and fitting/calibration counts accompany every model.
The four contrast intervals are pointwise and conditional on the fitted models
and thresholds; no family-wise coverage is asserted. Subsequent continuous
PTS inference uses 15-step hops and two-window persistence. Its episode budget
is 0.5 per accident clip, selected only on the corresponding training holdout.
The test event cohort is the eligible intersection with each previously frozen
100-clip evaluation cohort, with no replacement sampling.

The machine-readable manifest is generated before feature extraction or
fitting. It hashes every source video, frame-window manifest, detector,
implementation, and reference model. Decoder validation is resumable under a
signature including metadata, implementation and FFprobe hashes. Failures are
retained as eligibility records, not dropped according to predictions.

The original 500-clip model, desktop defaults, legacy clock studies and their
published endpoints remain unchanged. New models and PTS results use separate
artifact names. Test scores do not determine replacement defaults.

```powershell
python scripts/prepare_presentation_protocol.py
python -m pytest tests/test_incident_presentation.py
```

## Frozen Cohort

All 2,027 sources were inspected before fitting. The decoded-PTS verification
retains **1,861 clips and 2,985 paired windows**. Exclusions comprise 126
fixed-event cadence/lateness violations, 27 decoder frame-identity or clock
disagreements, seven insufficient-context cases, four unexpected container
counts and two events beyond the source-frame policy's final sampled frame.
Decoder disagreements describe this verified reader contract, not independent
judgments that the underlying videos or annotations are invalid.

| Split | Partition | Clips | Windows | Incident windows |
| --- | --- | ---: | ---: | ---: |
| IID | Fit | 376 | 572 | 376 |
| IID | Calibration | 94 | 141 | 94 |
| IID | Test | 1,391 | 2,272 | 1,391 |
| Geographic | Fit | 332 | 496 | 332 |
| Geographic | Calibration | 83 | 122 | 83 |
| Geographic | Test | 1,446 | 2,367 | 1,446 |

The separately frozen continuous-event intersections contain 88 IID and
90 geographic test clips, with 94 and 83 training-holdout calibration clips.
No excluded test clips are replaced. Comparisons with earlier cohorts must
retain these changes in sample composition and event clocks.

[Protocol, source hashes and frozen contracts](results/presentation_protocol.json) ·
[Paired frame indices, presentation times and membership](results/presentation_protocol.jsonl).

## Sequential Execution

```powershell
python scripts/run_overnight_research.py --phase cadence --deadline 2026-10-17T04:00:00+00:00
```

The new queue extracts both complete feature arms before fitting eight
candidate/global by IID/geographic models. It uses one GPU job and two CPU
threads per worker, retains resumable per-window caches, and publishes verified
predictions, model/input hashes, runtime versions, class prevalence and four
paired contrasts after all fits finish. Source decode uses the same per-clip
support verified at cohort preparation; only the chosen arm's 16 images enter
the detector and motion model. Smoke subsets are explicitly barred from fitting.
Continuous PTS event replay remains a subsequent, separate evaluation.
