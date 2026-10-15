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

## Workspaces

1. Incident workspace: video playback, seek controls, asynchronous analysis,
   cancellation, event timestamps, and result export.
2. Model evaluation: event-level recall/delay and matched-cohort temporal-model
   comparisons at holdout-calibrated operating points.
3. Visual evidence: before/trigger/after context, NAFNet restoration, and
   SM3Det architecture/transfer analysis.
4. Government risk: camera-budget performance on temporal holdout records and
   crash-hotspot proximity to public-camera locations.
5. Edge deployment: available ONNX providers and Verilog/SystemC validation.

## Runtime Contract

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

```powershell
python scripts/analyze_desktop_video.py --input docs/assets/incident_input.mp4 --output-dir outputs/desktop/qnn_trial --provider QNNExecutionProvider --model path/to/incident_tcn.qdq.onnx --metadata Application/desktop/models/incident_tcn.json --qnn-backend path/to/QnnHtp.dll
```

Reference: [ONNX Runtime QNN documentation](https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html).

## Reproducible UI Verification

```powershell
python scripts/run_desktop.py --analyze --screenshot docs/assets/desktop_workbench.png
python -m pytest tests/test_desktop_app.py tests/test_onnx_runtime.py tests/test_incident_causal_rendering.py
```

The screenshot command renders the native Qt window, executes reference-video
analysis, selects the first event when present, saves the result, and exits.
