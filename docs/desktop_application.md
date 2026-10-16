# Native Desktop Application

## Purpose

The PySide6 workbench integrates traffic-video analysis, calibrated temporal
inference, evidence inspection, model evaluation, and government-informed camera
allocation in a native laptop application. The operator opens a clip, analyzes
it, inspects event timestamps, and exports the resulting JSON record.

## Installation and Launch

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r Application/DebugVersion/requirements.txt -r requirements-desktop.txt
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
.\.venv\Scripts\python.exe scripts/run_desktop.py
```

The repository includes a compact ONNX temporal-model bundle and a complete
30-second reference video. Vehicle detector weights are obtained from the
upstream Ultralytics distribution. The default desktop runtime uses two CPU
threads for proposal tracking and ONNX temporal inference.

`RunTrafficWorkbench.cmd` provides a Windows double-click entry point after
installation. The native application does not require a browser or local web
server.

The **Open results** action restores a completed `report.json`, including its
annotated video, event navigation, and event-specific visual evidence. Analysis
need not be repeated to inspect a saved result.

## Workspaces

1. Incident workspace: video playback, seek controls, asynchronous analysis,
   cancellation, event timestamps, result export, and camera observations.
2. Model evaluation: event-level recall/delay and matched-cohort temporal-model
   comparisons at holdout-calibrated operating points.
3. Visual evidence: before/trigger/after context, NAFNet restoration, and
   SM3Det architecture/transfer analysis. New events export their own clean
   contact sheet and structured VLM request; optional Qwen review attaches
   schema validity, visual reasons, confidence, and review duration.
4. Government risk: camera-budget performance on temporal holdout records and
   crash-hotspot proximity to public-camera locations.
5. Edge deployment: available ONNX providers and Verilog/SystemC validation.

## Runtime Contract

The camera action accepts a compatible public page, a direct MJPEG URL, or a
snapshot URL. Observations are bounded to 30--120 seconds and support preview
and cooperative cancellation. Continuous inference produces causal window-end
events during acquisition, with a one-frame mailbox and bounded context ring.
Compatible MJPEG feeds use one persistent connection. The alternative capture
workflow spools frames on disk, then encodes playback at
their measured mean cadence. Source timestamps, hashes, and transport metadata
remain in `capture.json`; analysis verifies the video hash before using that
clock. The optional subsequent analysis executes in its own worker.

```powershell
python scripts/run_desktop.py --capture-page "https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000129" --capture-duration 30 --analyze
python scripts/run_desktop.py --capture-page "https://motoretag.taichung.gov.tw/ATIS_TCC/Device/Showcctv?id=C000129" --capture-duration 30 --continuous
```

The [streaming protocol](streaming_inference.md) separates immediate event
notification, deferred post-context evidence, and subsequent VLM service.

Each analysis runs in a separate `QProcess`, preserving UI responsiveness and
isolating worker termination. Input clips are limited to 120 seconds for
interactive analysis. Every job has a unique output directory under
`outputs/desktop/`; the report records its model hash, input, processing time,
threshold, temporal window scores, confirmed event timestamps, and execution
provider. Export preserves this record.

Temporal scores become visible only after the corresponding input window is
complete. The display holds the latest available decision until the next
window completes. The same timing convention applies to event evaluation.

## Qualcomm Deployment

The application discovers execution providers through ONNX Runtime. A QNN
entry is available when `QNNExecutionProvider` is installed. QNN HTP execution
requires a target-compatible quantized graph and the Qualcomm backend runtime.
The runtime sets `session.disable_cpu_ep_fallback=1` and disables session
fallback so a requested HTP run has an explicit execution contract.

CPU and QNN are temporal-model execution choices; the current desktop vehicle
proposal stage uses the CPU. End-to-end latency is recorded separately from
the temporal model's execution-provider configuration. Target profiling should
record device, runtime version, graph precision, warmup, repetitions, latency
distribution, and energy measurement method.

The bundled QNN preparation uses fixed input shape `(1, 15, 10)`, uint16
activations, uint8 weights, and training-only calibration. Over 450 windows in
the IID test partition, QDQ probability MAE is 0.00201 and thresholded decisions agree
with float32 on all windows. The operating threshold remains fixed. The
[quantization report](results/qnn_quantization.json) records the split, hashes,
precision, and validation provider.

```powershell
python scripts/quantize_incident_tcn_qnn.py
```

```powershell
python scripts/analyze_desktop_video.py --input docs/assets/incident_input.mp4 --output-dir outputs/desktop/qnn_trial --provider QNNExecutionProvider --model path/to/incident_tcn.qdq.onnx --metadata Application/desktop/models/incident_tcn.json --qnn-backend path/to/QnnHtp.dll
```

Reference: [ONNX Runtime QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html).

## Saved-Event Evidence Review

The toolbar's review action submits the selected saved event, including events
from a completed camera observation. No local source-video file or detector
rerun is required. A separate worker loads the pinned Qwen model and writes a
new report, raw response, request/image hashes, model revision, and source-report
lineage. The original event probability, source timing, acquisition metrics,
and processing duration remain unchanged; review duration is recorded separately.
The worker accepts at most three explicitly selected events per CLI invocation.

An operating-system file lock prevents overlapping workbench and research-runner
GPU jobs. Before entry, the guard also verifies recorded research child PIDs
against actual command lines. A terminated worker releases the OS lock; stale
PID records do not authorize overlap with a matching live child. Command-line
experiment scripts run through the guarded sequential runner.

```powershell
python scripts/review_desktop_events.py --report outputs/desktop/RUN_ID/report.json --event-index 0 --output-dir outputs/desktop/NEW_REVIEW_ID
```

The desktop triage prompt covers collisions, abrupt conflicts and stopped
hazards. Its outcomes remain separate from the controlled collision-specific
paired benchmark. The [evidence disposition policy](vlm_evidence_routing.md)
keeps missing evidence and requested operator review visible.

## Reproducible UI Verification

```powershell
python scripts/run_desktop.py --analyze --screenshot docs/assets/desktop_workbench.png
python scripts/run_desktop.py --report outputs/desktop/RUN_ID/report.json --page 2 --screenshot outputs/desktop/evidence.png
python -m pytest tests/test_desktop_app.py tests/test_onnx_runtime.py tests/test_incident_causal_rendering.py
```

The screenshot command renders the native Qt window, executes reference-video
analysis, selects the first event when present, saves the result, and exits.

The [native walkthrough](assets/desktop_walkthrough.mp4) records completed
analysis in the actual application. To produce a new recording:

```powershell
python scripts/record_desktop_demo.py --report outputs/desktop/RUN_ID/report.json
```

The script records window pixels and writes capture metadata alongside its
video. Recording reuses completed inference rather than launching another GPU
workload.
