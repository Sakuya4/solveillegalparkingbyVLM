from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


class OnnxIncidentPredictor:
    def __init__(self, model_path: Path, metadata_path: Path, *, provider: str = "CPUExecutionProvider", backend_path: str | None = None, threads: int = 2):
        import onnxruntime as ort

        if provider not in ort.get_available_providers():
            raise ValueError(f"{provider} is unavailable. Installed providers: {ort.get_available_providers()}")
        if threads <= 0:
            raise ValueError("threads must be positive")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.feature_names = tuple(metadata["feature_names"])
        self.threshold = float(metadata["threshold"])
        if not self.feature_names or not math.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("Model metadata must contain features and a valid threshold")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        provider_options: dict[str, str] = {}
        if provider == "QNNExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
            provider_options = {"backend_path": backend_path} if backend_path else {"backend_type": "htp"}
            provider_options["offload_graph_io_quantization"] = "0"
        self.session = ort.InferenceSession(str(model_path), sess_options=options, providers=[provider], provider_options=[provider_options])
        self.session.disable_fallback()
        self.input = self.session.get_inputs()[0]
        if provider not in self.session.get_providers():
            raise RuntimeError(f"Requested execution provider was not activated: {provider}")
        self.runtime = {
            "engine": "ONNX Runtime", "version": ort.__version__,
            "requested_provider": provider, "session_providers": self.session.get_providers(),
            "cpu_fallback_disabled": provider == "QNNExecutionProvider",
            "provider_options": provider_options, "threads": threads,
        }

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float32)
        if values.ndim != 3 or values.shape[2] != len(self.feature_names) or not np.all(np.isfinite(values)):
            raise ValueError("features must be finite (batch, time, features) values matching model metadata")
        if isinstance(self.input.shape[1], int) and values.shape[1] != self.input.shape[1]:
            raise ValueError(f"Expected {self.input.shape[1]} time steps")
        if self.input.shape[0] == 1:
            outputs = [self.session.run(None, {self.input.name: window[None]})[0] for window in values]
            return np.asarray(outputs, dtype=np.float64).reshape(-1)
        return np.asarray(self.session.run(None, {self.input.name: values})[0], dtype=np.float64).reshape(-1)


def runtime_capabilities() -> dict[str, Any]:
    import onnxruntime as ort

    providers = ort.get_available_providers()
    return {"onnxruntime_version": ort.__version__, "available_providers": providers, "qnn_available": "QNNExecutionProvider" in providers}
