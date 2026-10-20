# Continuous-Context Training Alignment

## Rationale

The TRAIN-only event-local diagnostic identifies weak within-source score
ranking and insufficient adjacent target-positive decisions under constrained
thresholds. A new training comparison will therefore examine the alignment of
training contexts with continuous inference, retaining the existing TCN
architecture and candidate/global feature definitions as controls.

Before extracting features or fitting models, a fit-only feasibility audit
fixes its source membership and physical-time label definitions. It inherits
the grouped fit partitions from the paired presentation study; calibration and
test identities cannot enter a split's fit audit. A source that belongs to fit
in only one split contributes only to that split's summaries.

## Predeclared Support Audit

The inference grid remains 7.5 Hz, with 16 distinct images, 15 motion steps
and hop 15. Lateness bounds and missing-context rejection use the immutable
presentation helpers. Relative to the supplied annotation's verified PTS:

- Prefix: decision endpoint strictly before event time minus one second.
- Target: decision endpoint in the inclusive interval [event minus one,
  event plus three seconds].
- Post: later endpoints, counted but excluded from proposed binary fitting.

These are annotation-relative supervision labels, not independently reviewed
visual-state labels or normal-camera observations. Every fit source remains in
the support denominator, including sources with no valid contexts or without
both classes. No predictions, score ranking or test endpoints are read.

Predeclared outputs include source/context counts, both-class support,
one-window and two-window structural reachability, adjacent target-target
support, exact frame-identity overlap with inherited fixed-time anchored
contexts, and inherited-label versus endpoint-label disagreements. Packet
timestamps and media hashes must match the parent audit. Expanded-context
decoder verification remains mandatory before feature extraction.

```powershell
python scripts/audit_continuous_training_support.py freeze
# Commit the protocol before computing support endpoints.
python scripts/audit_continuous_training_support.py run
python -m pytest tests/test_incident_training_support.py
```

## Training Gate

The support audit is not a fitted-model comparison. A subsequent immutable
protocol must specify exact fit/calibration/test membership, settings,
class/clip weighting, stopping rules, calibration objective and evaluation
endpoints before extraction or fitting. Compare fixed-time annotation-anchored
training with continuous-grid training under the same spatial and inference
contract. This changes context phase and supervision jointly, not architecture
alone. Any video-representation arm must share the same source/context/label
contract. Previously used TRAIN calibration rows are development data; earlier
window-test observations remain disclosed. Existing reference binaries and
operating defaults remain unchanged.

## Completed Fit-Support Audit

The protocol was committed as `98d364a` before computing the new endpoints.
All 557 distinct fit-union sources match the parent video and packet-PTS
hashes. The public inventory retains every corresponding fit source.

| Fit partition | IID | Geographic |
| --- | ---: | ---: |
| Source clips | 376 | 332 |
| Valid continuous contexts | 2,894 | 1,535 |
| Rejected contexts | 89 | 91 |
| Prefix / target / post contexts | 511 / 705 / 1,678 | 392 / 586 / 557 |
| Sources with both fitting classes | 237 | 186 |
| Gate-1 structural support | 375 | 330 |
| Gate-2 structural support | 337 | 282 |
| Adjacent target-target support | 322 | 255 |
| Anchored contexts | 572 | 496 |
| Exact anchored/continuous frame matches | 43 | 44 |

There are no empty-context sources in these fit partitions. No inherited
anchored labels disagree with the declared endpoint labels; the audit does
not identify mislabeled examples. Instead, it documents different context
phase and sampling support. Excluding post-event contexts leaves 1,216 IID
and 978 geographic binary-fitting contexts, without adding source clips.
This establishes feasibility, not improved recognition or deployment recall.
The complete training comparison and expanded-context decoder validation
remain subsequent gates.

[Frozen support protocol](results/continuous_training_support_protocol.json) ·
[Split summaries](results/continuous_training_support.json) ·
[557-source inventory](results/continuous_training_support.csv) ·
[3,830 fit-union contexts](results/continuous_training_support_contexts.csv).

## Frozen Model Comparison

The [training protocol](results/continuous_training_protocol.json) now fixes
eight anchored/continuous by candidate/global by IID/geographic TCN fits.
The corresponding 376/94 and 332/83 fit/calibration partitions are unchanged.
Both arms use 32 channels, dropout 0.1, 40 epochs, batch size 64, AdamW at
0.001 with weight decay 0.0001, seed 42, fit-only standardization and the
same class-weighted binary loss. The final epoch is retained without
calibration-driven checkpoint selection. Context counts and phase change
jointly; equal epochs do not imply equal optimization steps or training cost.

Each new event test cohort contains 100 clips, with a 190-source union, and
excludes 654 identities from prior event replays and holdouts. Selection is a
deterministic source-name hash, not score-dependent sampling. Previously
observed window-test results are explicitly acknowledged. All eight models
must be fitted before training-holdout event calibration; every operating
policy must then be frozen before test replay. Calibration fixes the 0.5
off-target episode budget and 0.2 macro prefix-positive cap, with the declared
tie order and unsupported-source handling. All four paired recall contrasts,
budget exceedances and constant controls will be reported.

The [2,633-row fitting manifest](results/continuous_training_protocol.jsonl)
contains only each source's corresponding fit memberships. Execution workers
and their hash-bound appendix must be tested and frozen before real extraction
or fitting. No new training or event-test result is claimed at this stage.

## Execution Appendix

The new execution workers leave all earlier study implementations unchanged.
Fixture tests exercise exact threshold tie-breaking, unsupported populations,
partition leakage rejection, finite feature tensors, smoke-data exclusion,
final-epoch reproducibility and the complete eight-model publisher. A
synthetic end-to-end publication retains an empty test source and verifies
every checkpoint probability against its stored tensor.

The execution freeze records two common-fit smoke sources, runtime and tool
versions, code hashes and permitted cache reuse. Parent fixed-time anchored
features and the existing continuous TRAIN-calibration tensors may be reused
only after hash, runtime and context-identity checks. Old model scores are not
used. Continuous fitting and new event-test tensors are generated separately.
Reused and newly extracted feature costs remain distinct.

```powershell
python scripts/freeze_continuous_context_execution.py
# Commit the execution appendix before real jobs.
python scripts/run_continuous_context_study.py smoke --device cpu
python scripts/run_continuous_context_study.py verify-fit
python scripts/run_continuous_context_study.py extract-fit
python scripts/run_continuous_context_study.py fit
python scripts/run_continuous_context_study.py calibrate
# Preserve all TRAIN-only selected policies before test replay.
python scripts/run_continuous_context_study.py test
python scripts/publish_continuous_context_study.py
```

All expanded fitting-frame identities must pass the decoder audit before
complete extraction or fitting. Failed sources remain in the audit and halt
the study rather than changing its cohort. Every new checkpoint's embedded
0.5 threshold is explicitly an unselected placeholder; only the separately
frozen training-holdout selection defines operating policies. Reference
desktop models are not replaced.

### Decoder Gate and Cost Contracts

The complete fitting decoder gate passes **27,986 selected frames** from
557 sources and all 2,633 manifest contexts, with zero failed sources. The
full audit is retained under `outputs/accident/continuous_context/`; its SHA256
is `6c0d7a2605ae5f8c8688eb945a4bbad3ebf5d44733c7641ec1b77dbd0549d212`.

Separate contracts, fixed before new model results, specify temporal-only
CPU latency and integer-controller verification. All models receive the same
first context from ten supported TRAIN-calibration sources. Twenty warm-up
calls precede 1,000 timed batch-one calls per model, with two CPU threads.
The benchmark includes standardization but excludes detection, decoding,
networking, queueing and VLM review. It runs after the extraction/replay jobs.

The new hardware check applies the original UNORM16 encoder, Python golden
controller and Icarus testbench to all selected calibration/test streams. It
counts float/integer positive, review, episode, event and trigger-time changes,
including empty sources and gap resets. It does not assume quantization parity
or claim NPU arithmetic, synthesized timing or target power.

[CPU timing contract](results/continuous_context_latency_protocol.json) ·
[New-score controller contract](results/continuous_context_rtl_protocol.json).
