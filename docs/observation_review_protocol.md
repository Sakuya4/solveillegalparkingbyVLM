# Source-Clock Observation Review and Normal Exposure

## Review Unit and Separation

A continuous observation supplies two synchronized sampled-frame videos:
privacy-treated model playback and privacy-treated clean playback. The latter
excludes added model boxes, probability text, event state, and artificial
timestamp overlays. The clean video and original frame timestamps are hashed
in the source report. Earlier reports without this complete clean recording
are not retroactively promoted to blinded observation records.

The native annotation dialog opens only clean playback. The default label is
`uncertain`; complete viewing and non-consultation of model outputs are separate,
unchecked reviewer attestations. Reviewer identity is mandatory. Saving writes
a new annotation file and does not modify the model report or infer a label
from VLM output. The attestations record the reviewer's declarations; the
application does not independently certify reviewer behavior or blindness.

## Exposure Contract

The fixed observation-gap limit is one second, specified before the additional
camera acquisition. Eligible elapsed time is the sum of adjacent source-clock
intervals no greater than that limit. Larger gaps and the unobserved interval
before the first or after the last sampled frame contribute no exposure.
Playback duration at 7.5 FPS is not substituted for source-clock exposure.
The limit defines an acquisition-coverage rule, not continuous visibility
between every pair of frames.

The initial 120-second acquisition request terminated after 30.303 s with a
source-read failure; its original record is retained. The subsequent transport
protocol allows three consecutive reconnect attempts, with interruptible
0.25/0.5/1-second backoff. A successful frame resets the consecutive-failure
budget. Source indices and decoder observation time remain monotonic across
connections. The main worker polls the mailbox every 0.5 s so duration and
cancellation checks continue while reconnecting.

When adjacent sampled frames differ by more than one second, motion features,
tracker state and persistence are reset. Pending event evidence is finalized
with the last available pre-gap image and marked truncated. A new causal
window requires fresh post-gap features; no artificial frames or continuous
exposure are inserted across the interruption. Gap records and reconnect
counters accompany subsequent acquisition reports.

Primary normal exposure requires label `normal`, complete available-frame review,
and a declaration that model outputs were not consulted. Uncertain, incomplete,
and model-informed records are excluded. Contradictory eligible labels exclude
the observation pending resolution. Duplicate clean-video hashes contribute
once, regardless of reviewer count. Confirmed incident observations and local
file replay do not contribute normal camera-hours. Different model, detector,
feature-width, hop, persistence, or threshold contracts are evaluated separately.

The numerator counts temporal alert episodes whose source trigger lies within
an eligible normal interval. The endpoint does not substitute VLM classifications
or raw positive windows for alert episodes. Report review coverage, camera-hours,
episode count, and the observed rate together.

## Rate Uncertainty

For count n and eligible exposure T in hours, the one-sided 95% Poisson upper
limit is `chi2.ppf(0.95, 2*(n+1)) / (2*T)`. At n = 0 this reduces to
`-log(0.05)/T`. Zero eligible exposure yields undefined rate and upper limit,
not a zero-error result. The construction follows the count-interval treatment
of [Garwood, Biometrika 28(3--4), 437--442 (1936)](https://doi.org/10.1093/biomet/28.3-4.437).
Publisher metadata verifies the title, author, volume, pages and DOI.

The interval is conditional on stationary Poisson event arrivals. Traffic,
camera conditions and temporal dependence can violate that assumption; longer
multi-session evaluation should retain session/district strata and investigate
overdispersion. Label reliability and acquisition gaps are separate uncertainty
sources, not corrected by a count interval.

## Execution

Open a new continuous camera report in the native workbench and select the
observation annotation action. The clean-video dialog stores records under
`outputs/observation_reviews/`. Only an actual reviewer assigns labels.

```powershell
python scripts/evaluate_observation_reviews.py --review-glob "outputs/observation_reviews/*.json" --output outputs/validation/reviewed_normal_exposure.json
python scripts/publish_observation_inventory.py --report outputs/desktop/RUN_ID/report.json
python -m pytest tests/test_observation_review.py tests/test_streaming_integration.py tests/test_desktop_app.py
```

The acquisition inventory reports unassigned review status and zero added
normal exposure until reviewer records satisfy the protocol. Public numeric
artifacts do not redistribute public-camera imagery.
