# ADR 002: Online Candidate ROI For Incident Detection

Status: Accepted

## Context

ACCIDENT supplies an accident bounding box, but a deployed roadside camera does
not know that box before detecting an event. Models trained on annotation ROI
features produced strong scores that cannot be reproduced at runtime.

## Decision

Generate runtime candidate regions from YOLO vehicle detections, ByteTrack state,
and frame motion. Fuse overlapping track and motion proposals, retain an explicit
global fallback, and expose proposal availability/source as model features.

SAM2 is an optional box-prompt refiner. The default edge path must run without
SAM2, a VLM, cloud access, or accident annotations.

Ground-truth accident boxes may calculate proposal IoU and recall after feature
extraction. Oracle diagnostics are written only to the experiment report and are
excluded from CSV, sequence tensors, and edge traces.

## Consequences

- The same runtime contract can feed logistic models, causal TCNs, and hardware
  scheduling traces.
- Candidate-only features improve geographic OOD results on the 500-clip
  ablation, but naive global/candidate concatenation is not consistently better.
- Proposal quality, score calibration, and false-positive control remain model
  selection criteria.
- Annotation ROI remains available only behind explicit oracle opt-in for upper
  bound experiments.
