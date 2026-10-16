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
