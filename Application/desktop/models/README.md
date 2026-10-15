# Desktop Inference Bundle

`incident_tcn.onnx` contains the calibrated candidate-ROI causal TCN and its
feature standardization. `incident_tcn.json` records the feature contract,
threshold, model hash, and training provenance. The model uses the ACCIDENT
500-clip cohort and grouped training-holdout calibration.

`incident_tcn.qdq.onnx` and its JSON metadata provide the fixed-batch A16W8
variant prepared with the ONNX Runtime QNN quantization helpers. Calibration
uses training windows only; both variants preserve the same operating threshold.

The bundle is provided under CC BY-NC-SA 4.0 for research and non-commercial
demonstration, consistent with the source benchmark. General project source
code remains under the repository's MIT license. Upstream detector weights
retain their own licenses and are downloaded separately.

The portable graph is executed by ONNX Runtime. CPU execution is available
with the desktop requirements. QNN HTP deployment uses a target-compatible
quantized model and the Qualcomm runtime described in
[the deployment protocol](../../../docs/desktop_application.md).
