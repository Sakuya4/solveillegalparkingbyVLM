# Capacity-Aware Multimodal Evidence Routing

## Research Question

Can causal proposal coalescing and service-aware admission control reduce
uncontrolled review-queue loss under a measured VLM service budget?

The design follows the separation between early video filtering and expensive
downstream analytics discussed in [Reducto, SIGCOMM 2020](https://www.cs.princeton.edu/~ravian/publications/reducto_sigcomm20.pdf).
Admission uses the strict token-bucket model described in
[RFC 3290, Appendix A](https://www.rfc-editor.org/rfc/rfc3290.html#appendix-A).
The implementation adapts these established ideas to candidate evidence;
it does not reproduce Reducto's learned frame-filter configuration.

## Protocol

The workload contains 12,330 model-produced candidate records, replicated
across four camera streams at a nominal 7.5 FPS. These records concatenate
sampled ACCIDENT windows. Their arrival density defines a capacity experiment,
while incident recall and continuous-camera exposure have separate evaluations.

Proposal eligibility requires availability and score at least 0.95. Causal
coalescing retains the first proposal and merges later overlapping candidates
within 15 replay steps at IoU at least 0.1. It resets at source-file or frame
discontinuities. Routing does not read event labels or accident annotations.

The shared meter refills at 80% or 100% of one review per 8.27361 seconds,
the mean of the controlled Qwen experiment. Burst capacity is four requests.
Rotating camera priority prevents a synchronized camera from monopolizing
available credits. The downstream queue has eight waiting positions. Detector
and temporal service times remain configured at 3.9 and 3.0 ms, respectively.

Every raw proposal belongs to exactly one accounting category: coalesced,
budget-denied, completed, pending, or queue-dropped. Budget-denied candidates
are not classified as completed evidence or successful incident detections.

## Results

Each policy processes 49,320 frames without front-end loss.

| Policy | Coalesced | Budget denied | Completed | Pending | Queue drops |
| --- | ---: | ---: | ---: | ---: | ---: |
| All proposals | 0 | 0 | 198 | 9 | 6,601 |
| Overlap merge | 5,240 | 0 | 198 | 8 | 1,362 |
| Metered proposals | 0 | 6,607 | 198 | 3 | 0 |
| Merge + 80% capacity | 5,240 | 1,406 | 161 | 1 | 0 |
| Merge + 100% capacity | 5,240 | 1,367 | 198 | 3 | 0 |

Overlap coalescing consolidates 76.97% of raw proposal arrivals but does not
resolve the remaining service deficit. At full measured capacity, metering
retains 198 completed reviews, reduces pending work from nine to three, and
replaces overflow with explicit admission decisions. Per-camera admissions are
51/50/50/50. The 80% profile leaves service headroom and completes 161 reviews.

All eight integer counters agree between Python and SystemC 3.0.2 for all five
policies. Finite frame-count generation avoids an additional terminal arrival
batch caused by simulation-time rounding. The native application displays the
same numerical artifact.

![Native routing diagnostics](assets/desktop_edge_routing.png)

## Reproduction

On Windows with Visual Studio 2022 C++ Build Tools, CMake, and Git:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build_systemc_windows.ps1
python scripts/evaluate_review_routing.py --systemc-executable outputs/toolchain/traffic-systemc-msvc-build/Release/npu_queue_sim.exe
python -m pytest tests/test_review_routing.py tests/test_systemc_review_replay.py
```

The build script verifies the official SystemC 3.0.2 release commit, limits
parallel compilation to two jobs, and removes its temporary drive alias on
exit. The result records trace, review-report, executable, and simulator-source
hashes. Existing CMake targets also support other SystemC installations.

[Numerical artifact](results/review_routing.json)

## TCN-Gated Event Retention

The second workload uses continuous-clip TCN outputs rather than raw candidate
proposals. For each of 100 IID and 100 geographic clips, persistence is applied
causally and only the first window of each positive episode generates a review
arrival. Accident annotations are attached afterward to measure retention
within the existing -1/+3-second event-matching interval. The workload preserves
fixed clip order, zero inter-clip gaps, and synchronized one/four-camera replay;
it measures benchmark demand rather than field incident frequency.

Original IID reports use five-step decision hops; geographic reports use
15-step hops. Routing comparisons hold each source batch fixed. Cross-split
differences therefore are not isolated effects of geography.

All 24 policy/profile/camera/split combinations reproduce the eight integer
counters in the locally built SystemC model. Candidate identities retained by
the Python simulation additionally link review completion to annotated events.

Selected conservative-profile comparisons are shown below. Completed event
counts refer to service completion for a matching detected episode. They do
not incorporate VLM classification correctness.

| Split | Cameras | Policy | Detected events | Completed matched events | Queue drops | Review latency p95 (s) |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| IID | 1 | All episodes | 44 | 44 | 0 | 33.00 |
| IID | 1 | 100% meter | 44 | 30 | 0 | 8.28 |
| IID | 4 | All episodes | 176 | 84 | 329 | 73.77 |
| IID | 4 | 100% meter | 176 | 69 | 0 | 33.10 |
| Geographic | 1 | All episodes | 30 | 29 | 0 | 10.55 |
| Geographic | 1 | 100% meter | 30 | 26 | 0 | 8.28 |
| Geographic | 4 | All episodes | 120 | 85 | 98 | 72.79 |
| Geographic | 4 | 100% meter | 120 | 69 | 0 | 33.10 |

Removing overflow does not guarantee improved event retention. For example,
single-camera IID conservative replay retains all 44 detected events without
metering, whereas the full-capacity meter retains 30. Conversely, in responsive
four-camera IID replay, the meter increases completed matched events from 51
to 62 out of 300 detected event-camera instances. Burst structure, persistence,
and admission jointly determine the retention/latency tradeoff.

These results motivate separating immediate edge notification from optional
VLM evidence review. A review budget controls evidence-service expenditure;
it should not silently suppress the detector's original event record. Camera
replicas do not represent independent statistical observations. The review
service is a constant measured mean; front-end times remain explicit scenario
parameters.

[IID event-retention artifact](results/incident_review_routing_iid.json) ·
[Geographic event-retention artifact](results/incident_review_routing_geographic.json)

![Native event-retention diagnostics](assets/desktop_event_retention.png)

```powershell
python scripts/evaluate_incident_review_routing.py --systemc-executable outputs/toolchain/traffic-systemc-msvc-build/Release/npu_queue_sim.exe
python scripts/evaluate_incident_review_routing.py --batch-dir outputs/accident/event_batch_geographic_100_20261016 --output docs/results/incident_review_routing_geographic.json --systemc-executable outputs/toolchain/traffic-systemc-msvc-build/Release/npu_queue_sim.exe
```

## Subsequent Evaluation

Extend event-retention experiments to normal-stream exposure, variable service
times, and heterogeneous camera arrivals. Government risk can inform resource allocation, with minimum camera
service guarantees evaluated alongside risk-weighted prioritization.
