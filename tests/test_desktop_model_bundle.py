import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("onnxruntime")
from illegal_parking.onnx_runtime import OnnxIncidentPredictor


@pytest.mark.parametrize("basename", ["incident_tcn", "incident_tcn.qdq"])
def test_versioned_desktop_model_hash_and_portable_inference(basename):
    bundle = Path(__file__).resolve().parents[1] / "Application/desktop/models"
    model_path = bundle / f"{basename}.onnx"
    metadata_path = bundle / f"{basename}.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == metadata["sha256"]
    predictor = OnnxIncidentPredictor(model_path, metadata_path)
    values = np.zeros((2, 15, 10), dtype=np.float32)
    probabilities = predictor.predict_probabilities(values)
    assert probabilities.shape == (2,)
    assert np.all((0 <= probabilities) & (probabilities <= 1))
