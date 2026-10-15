import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from illegal_parking.onnx_runtime import OnnxIncidentPredictor, runtime_capabilities


def test_unavailable_provider_is_rejected_before_session_creation(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        OnnxIncidentPredictor(tmp_path / "none.onnx", tmp_path / "none.json", provider="UnknownExecutionProvider")


def test_capabilities_report_installed_providers():
    capabilities = runtime_capabilities()
    assert "CPUExecutionProvider" in capabilities["available_providers"]
    assert capabilities["qnn_available"] == ("QNNExecutionProvider" in capabilities["available_providers"])


def test_qnn_request_disables_cpu_and_io_quantization_fallback(monkeypatch, tmp_path):
    import onnxruntime as ort

    metadata = tmp_path / "metadata.json"
    metadata.write_text(json.dumps({"feature_names": ["x"], "threshold": 0.5}), encoding="utf-8")
    observed = {}

    class Session:
        def disable_fallback(self):
            observed["fallback_disabled"] = True

        def get_inputs(self):
            return [SimpleNamespace(name="raw_features", shape=[1, 15, 1])]

        def get_providers(self):
            return ["QNNExecutionProvider"]

    def create_session(path, **kwargs):
        observed.update(kwargs)
        return Session()

    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider", "QNNExecutionProvider"])
    monkeypatch.setattr(ort, "InferenceSession", create_session)
    OnnxIncidentPredictor(tmp_path / "model.onnx", metadata, provider="QNNExecutionProvider")
    assert observed["sess_options"].get_session_config_entry("session.disable_cpu_ep_fallback") == "1"
    assert observed["provider_options"][0]["offload_graph_io_quantization"] == "0"
    assert observed["fallback_disabled"] is True


def test_onnx_predictions_match_torch_checkpoint_when_artifacts_are_present():
    from illegal_parking.incident_tcn_inference import IncidentTcnPredictor

    root = Path(__file__).resolve().parents[1]
    model = root / "outputs/hardware/incident_tcn.onnx"
    metadata = root / "outputs/hardware/incident_tcn_onnx_report.json"
    checkpoint = root / "models/candidate500_tcn_only_iid_fpr20.pt"
    if not all(path.is_file() for path in (model, metadata, checkpoint)):
        pytest.skip("Local trained model bundle is required for runtime parity")
    onnx = OnnxIncidentPredictor(model, metadata)
    torch = IncidentTcnPredictor.from_checkpoint(checkpoint)
    values = np.random.default_rng(42).normal(size=(8, 15, len(onnx.feature_names))).astype(np.float32)
    np.testing.assert_allclose(onnx.predict_probabilities(values), torch.predict_probabilities(values), atol=1e-6)
    with pytest.raises(ValueError, match="finite"):
        onnx.predict_probabilities(np.full_like(values, np.nan))
