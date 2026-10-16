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
