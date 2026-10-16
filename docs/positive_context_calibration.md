# Positive-Context-Constrained Event Calibration

## Research Question

Can a training-only positive-context constraint prevent the near-always-positive
operating policies admitted by an alert-onset budget, while retaining useful
event recall on a separately declared continuous-replay cohort?

This follow-up retains all eight fitted TCNs and their score functions. It
changes the decision objective, not their architecture, weights, features or
physical-time inference contract. The distinction between score estimation
and operating decisions follows the [ROC analysis foundation](https://doi.org/10.1016/j.patrec.2005.10.010)
and [independent threshold-selection guidance](https://scikit-learn.org/stable/modules/classification_threshold.html).
The numerical budgets below are declared application constraints, not claimed
universal optima from those references.

## Predeclared Objective

Each valid context whose last-frame PTS is strictly earlier than the supplied
event PTS minus one second contributes to the annotation-derived pre-event
population. A clip's positive fraction is the number of these contexts with
score at least the threshold, divided by its eligible context count. The
primary exposure statistic averages these fractions equally across supported
clips, avoiding greater weight for longer clips. Pooled fractions and the
number of clips without eligible contexts accompany it. Unsupported clips
do not receive an invented zero exposure.

Thresholds maximize training-holdout event recall subject to both:

- At most **0.5 off-target alert episodes per accident clip**.
- At most **0.2 macro pre-event positive-context fraction**, with at least
  **20 supported calibration clips**.

The second constraint retains the earlier study's 20% operating-budget level
but defines a different endpoint from its window-FPR calibration. Selection
uses the inherited 0.01-spaced probability grid plus the existing
window-calibrated threshold. Ties favor fewer off-target episodes and then
a higher threshold. Infeasible or insufficient-support policies remain
unselected. There is no test-set search for a replacement default.

These prefixes remain accident-clip annotations. Their context fractions
are not independent normal-camera false-alarm rates or physical alert
durations. Event-level denominators include all declared clips, including
those with no valid inference context.

## Cohort and Execution Freeze

The original 94 IID and 83 geographic training-holdout clips supply all
calibration. Of these, 59 and 40 have eligible pre-event contexts.
Existing calibration features/probabilities are reused under their verified
batch hash; no detector rerun or model fitting is required for selection.

The replay inventory excludes **463 previously processed clip files** before
choosing new event cohorts. Each scheme retains exactly **100 test clips**;
their combined union contains **191 distinct files**. A fixed SHA256 ordering
of seed 42, split name and clip path selects membership without reading model
scores. Available candidate counts before selection are 1,212 IID and 1,251
geographic clips. All prior calibration/test replay identities and supplied
annotation-frame PTS anchors remain traceable.

The follow-up was motivated by the completed onset-budget study. Although
these files have not entered the inventoried continuous event replays, their
paired-window test results were previously observed. The study therefore
evaluates a declared new event-replay endpoint, not a completely untouched
dataset or a retrospective replacement for the original experiment.

The protocol SHA256 is
`3f0fe6e023f92252af49e3bac220715fd73a28be6aea6e4565774f06194a375d`.
Model/source hashes, prior replay inventory, source membership, constraints,
feature implementation and full calibration curves are frozen before replay.
A separate execution appendix binds the new worker/publisher code to the
same protocol and selected thresholds before GPU inference begins.

## Training-Holdout Selections

| Model / split | Threshold | Event recall | Pre-event macro fraction | Off-target / clip |
| --- | ---: | ---: | ---: | ---: |
| Source-frame candidate / IID | 0.84 | 0.117 | 0.186 | 0.202 |
| Source-frame global / IID | 0.78 | 0.117 | 0.144 | 0.117 |
| Source-frame candidate / geographic | 0.99 | 0.036 | 0.191 | 0.084 |
| Source-frame global / geographic | 0.81 | 0.096 | 0.183 | 0.024 |
| Fixed-time candidate / IID | 0.98 | 0.064 | 0.184 | 0.117 |
| Fixed-time global / IID | 0.71 | 0.074 | 0.168 | 0.170 |
| Fixed-time candidate / geographic | 0.99 | 0.084 | 0.153 | 0.036 |
| Fixed-time global / geographic | 0.58 | 0.145 | 0.197 | 0.120 |

All eight training-only settings satisfy both declared constraints. Their
lower calibration recall quantifies the cost of excluding trivial positive
policies; it does not predict a test-set improvement. The new test replay
must report recall and both constraint outcomes together.

## Planned Comparisons

For each model, evaluate the constrained policy and its original onset-only
training-selected threshold on the same new clips. Retain same-grid
clock-only positive and always-negative controls. Report all sixteen policy
outcomes, pre-event support, test constraint exceedance, signed trigger delay
and per-clip outcomes. Eight paired recall contrasts use 2,000 source-clip
bootstrap resamples with seed 42, conditional on the fixed models/selections.
These intervals are pointwise rather than family-wise.

The publisher reproduces the complete calibration curves before reading
test endpoints and recomputes every new checkpoint probability from saved
feature tensors. Invalid cadence contexts receive no prediction, missing hops
reset persistence, and source/decoder mismatches stop publication rather
than changing cohort membership. The 500-clip desktop models and all prior
published measurements remain unchanged.

```powershell
python scripts/prepare_positive_exposure_study.py
python scripts/replay_positive_exposure.py --freeze-execution
python scripts/run_overnight_research.py --phase positive_exposure --deadline 2026-10-17T04:00:00+00:00
python -m pytest tests/test_positive_exposure.py
```

[Frozen protocol and replay inventory](results/positive_exposure_protocol.json) ·
[Complete calibration curves and selections](results/positive_exposure_selection.json) ·
[Execution appendix](results/positive_exposure_execution.json).
