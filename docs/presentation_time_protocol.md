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
and fitting/calibration counts accompany every model. Subsequent continuous
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
