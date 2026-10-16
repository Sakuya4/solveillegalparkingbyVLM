# Training-Holdout Event-Local Diagnostics

## Question and Scope

The constrained replay reveals a sensitivity/workload tradeoff, but does not
identify its mechanism. This diagnostic separates annotation-relative score
ranking, physical-time support, positive-pair persistence and episode-onset
suppression. It retains all eight models and uses only each model's existing
official-TRAIN calibration clips: 94 IID and 83 geographic sources.

The protocol is specified after earlier window/event outcomes were observed.
These calibration clips have already informed threshold selection. The new
endpoints are therefore exploratory development diagnostics, not an untouched
test or a replacement for published evaluations. No model is fitted, reversed,
selected or promoted to a new native default.

## Input Freeze

Before computing any new endpoint, export calibration-only inventory and score
CSVs from the hash-bound public context traces. Preserve every assigned clip,
including clips without valid contexts. Freeze source membership, model hashes,
annotation endpoints, score-file hashes, runtime versions, endpoint definitions,
two inherited operating policies and all paired contrasts. A separate execution
appendix will bind the diagnostic implementation before real-data analysis.
Test rows do not enter these derived inputs or the analysis reader.

## Ranking Endpoints

For a supplied event PTS `a`, contexts ending before `a-1` are prefix negatives.
Contexts ending in the inclusive interval `[a-1, a+3]` are annotation-relative
targets. Later contexts are counted and summarized, but excluded from this
ranking task. These endpoint labels are not independent visual frame labels.

Report pooled AUROC and noninterpolated average precision (AP), target
prevalence and AP minus prevalence. Separately compute each metric within
clips containing both labels and average equally over supported clips. Disclose
all unsupported populations; do not give them invented zero metrics. Within
each supported clip, also summarize maximum target minus maximum prefix score.
Larger model scores always mean target; unfavorable rankings are not inverted.

Precision-recall analysis accompanies AUROC because its interpretation depends
on the positive fraction. The methodological source is [Saito and Rehmsmeier,
PLOS ONE 10(3), e0118432 (2015)](https://doi.org/10.1371/journal.pone.0118432),
whose authors, date and article metadata were verified against the publisher.
AP follows the [scikit-learn definition](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html),
not trapezoidal PR-curve integration. A constant-score control uses the same
contexts and exposes the prevalence baseline without claiming visual skill.

## Timing and Controller Diagnostics

The one-context structural diagnostic asks whether any valid endpoint falls
inside the target band. The two-context diagnostic additionally requires an
immediately preceding valid hop-15 context. This is an annotation-aided timing
upper bound, ignoring both score quality and operating caps. It is not an
implementable recognition policy.

For both previously selected thresholds, classify every calibration clip into
one mutually exclusive category, in this order:

1. Detected: a two-positive episode onset ends within the target band.
2. Structurally unreachable: no admissible adjacent pair can end in that band.
3. No target positive pair: the grid supports detection, but the thresholded
   scores never form an adjacent positive pair ending in the band.
4. Onset suppression: such a pair exists, but sustained positivity emitted
   its episode before the target band and never re-armed inside it.

All clips remain in the denominator. Missing hops reset persistence. This
decomposition attributes deterministic score/controller behavior, not unseen
visual causes or real-world normal-camera false alarms.

## Paired Reporting

Compare fixed-time minus source-frame training for candidate/global features
and IID/geographic models, on identical supported source identities. Report
eight pointwise contrasts: four within-clip AUROC and four within-clip AP
differences, with 2,000 source-clip percentile bootstrap resamples, seed 42.
Keep both signs, class/support counts and conditional uncertainty. No new test
scores or test-driven hyperparameters enter this diagnostic.
