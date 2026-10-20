# Public-Camera Runtime Experiment

## Protocol and Execution

A 120-second continuous observation of Taichung public camera C000129 began
at **03:23:26 Asia/Taipei on 17 October 2026**. The source request, reference
weights, inference contract and endpoints were committed as `f66429f` before
acquisition. The native streaming worker used two CPU threads and ONNX Runtime
CPU execution; no VLM workload ran during acquisition.

The unchanged 500-clip reference model used 15-step windows, hop 15 and
two-window persistence at threshold 0.8547053337097168. Input sampling targeted
7.5 Hz with persistent tracking on 640-pixel-wide frames. This native contract
is distinct from the per-context tracker resets in the newer PTS studies.

## Measured Results

| Endpoint | Measurement |
| --- | ---: |
| Requested observation | 120 s |
| Worker observation elapsed time | 120.250 s |
| Sampled source-clock span | 119.835 s |
| Decoded frames | 1,189 |
| Processed sampled frames | 884 |
| Effective sampled cadence | 7.368 Hz |
| Temporal decisions | 58 |
| Sampled-frame processing, median / p95 | 100.5 / 125.0 ms |
| Decoder arrival to decision, median / p95 | 107.0 / 140.7 ms |
| Initialization before observation | 6.898 s |
| Model warm-up within initialization | 2.683 s |
| Connections / cumulative retry attempts | 4 / 4 |
| Decoder read failures | 4 |
| Sampled-frame gaps above one second | 0 |
| Eligible acquisition at the one-second gap limit | 119.835 s |
| Candidate alert episodes | 0 |
| End condition | Requested duration reached |

The bounded reconnect mechanism sustained the requested observation despite
transport interruptions. Initialization precedes the observation interval.
Reported decision latency begins at the local decoder arrival clock; it is
not camera sensor-to-notification latency or incident-onset delay. Source
sensor timestamps and image freshness are not independently certified by this
acquisition clock.

The observation remains **UNASSIGNED** for incident/normal ground truth.
Zero candidate episodes describe this model execution, not detection accuracy
or a false-alarm rate. Independently reviewed normal exposure remains unchanged.

## Reproduction and Artifacts

```powershell
$session = Join-Path outputs/streaming ([guid]::NewGuid().ToString('N'))
python scripts/run_streaming_incident.py --input "https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000129" --source-mode page --duration-sec 120 --min-positive-windows 2 --provider CPUExecutionProvider --output-dir $session
python scripts/publish_live_runtime.py --report (Join-Path $session report.json) --protocol docs/results/live_runtime_20261017_protocol.json --output (Join-Path $session runtime_summary.json)
python scripts/run_desktop.py --report (Join-Path $session report.json)
```

A subsequent acquisition is a new session, not a reproduction of the same
traffic scene; retain its own report and output paths. The numerical release
includes every sampled-frame timestamp and temporal decision, processing
summaries, source/model/code/media hashes and explicit label status. Tests
reconstruct eligible acquisition and median decision latency from these
public traces and reject mismatched requests, weights, clocks or decisions.

Local playback and the native screenshot remain under
`outputs/streaming/live_20261017_0325/`. Public-camera imagery is not included
in the Git release; numerical records support inspection without redistributing
footage whose reuse terms have not been separately established.

[Predeclared protocol](results/live_runtime_20261017_protocol.json) ·
[Complete numerical trace and provenance](results/live_runtime_20261017.json) ·
[Public camera page](https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000129).

## Daytime Observation on 20 October

A separately predeclared request began at 12:09:42 Asia/Taipei on
20 October 2026. It retained the same reference model and native camera
contract, with two CPU threads. A fit-only decoder audit ran concurrently;
these measurements are not a resource-isolated benchmark or a paired
day/night model comparison.

| Endpoint | Daytime measurement |
| --- | ---: |
| Requested / elapsed observation | 120 / 120.198 s |
| Decoded / sampled frames | 1,169 / 841 |
| Temporal decisions | 56 |
| Eligible acquisition at the one-second gap limit | 119.700 s |
| Decoder-arrival-to-decision median / p95 | 129.0 / 195.4 ms |
| Sampled-frame processing median / p95 | 110.1 / 154.9 ms |
| Connections / read failures / retry attempts | 4 / 4 / 4 |
| Sampled-frame gaps above one second | 0 |
| Candidate alert episodes | 4 |

The four source-clock triggers occur at 30.593, 58.024, 98.929 and 117.801 s.
Their full privacy-treated clean observation and event evidence remain local
under `outputs/streaming/live_20261020_daytime/`. All four alerts and the full
observation remain **UNASSIGNED**. They support independent review and
subsequent error analysis; they do not add reviewed-normal exposure or a
measured false-alarm rate. A model-informed visual review is not a substitute
for independent observation labels.

[Daytime protocol](results/live_runtime_20261020_protocol.json) ·
[Daytime frame/decision trace and hashes](results/live_runtime_20261020.json).

### Saved-Event Multimodal Review

All four retained candidates subsequently pass through the pinned Qwen2.5-VL-3B
review service using three privacy-treated clean images per event and generation
contract version 2. A separate protocol fixes every request/image/prompt hash,
greedy generation, revision, token limit and the existing three-event batch
limit before execution. Two batches retain the complete report chain without
rerunning detection or modifying the original observation report.

All four outputs are schema-valid. Each returns a positive triage opinion at
reported confidence 0.85 while also identifying one missing-evidence item; the
policy therefore retains all four as `incomplete_visual_evidence`. The model's
confidence is not an empirical correctness estimate. Per-event generation and
parsing take 9.370--12.490 s, excluding model load; both batch totals including
load are recorded separately. These later services do not redefine the earlier
edge notification timestamps or establish independent event truth.

The final local report is
`outputs/streaming/live_20261020_daytime_review_all/report.json` and can be
opened directly in the native application. Text and camera media remain local;
the public release includes numeric opinions, dispositions and provenance.
Every independent label remains unassigned and added reviewed-normal exposure
remains zero.

[Review protocol](results/live_review_20261020_protocol.json) ·
[Four-event review results](results/live_review_20261020.json).
