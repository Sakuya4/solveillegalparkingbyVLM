from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))


def main():
    parser = argparse.ArgumentParser(description="Prepare a fixed-shape QNN A16W8 temporal model using training-only calibration.")
    parser.add_argument("--model", type=Path, default=ROOT / "Application/desktop/models/incident_tcn.onnx")
    parser.add_argument("--metadata", type=Path, default=ROOT / "Application/desktop/models/incident_tcn.json")
    parser.add_argument("--sequences", type=Path, default=ROOT / "data/processed/accident/candidate_roi_sequences_500.npz")
    parser.add_argument("--calibration-samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "Application/desktop/models/incident_tcn.qdq.onnx")
    parser.add_argument("--report", type=Path, default=ROOT / "docs/results/qnn_quantization.json")
    args = parser.parse_args()
    if args.calibration_samples <= 0:
        parser.error("--calibration-samples must be positive")
    import onnx
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationDataReader, QuantType, quantize
    from onnxruntime.quantization.execution_providers.qnn import get_qnn_qdq_config, qnn_preprocess_model
    from illegal_parking.onnx_runtime import OnnxIncidentPredictor

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    payload = np.load(args.sequences, allow_pickle=False)
    split_key = "iid_split" if metadata["split_scheme"] == "iid" else "geographic_split"
    features = tuple(payload["feature_names"].astype(str))
    columns = [features.index(name) for name in metadata["feature_names"]]
    values = payload["features"][:, :, columns].astype(np.float32)
    train_indices = np.flatnonzero(payload[split_key] == "train")
    test_indices = np.flatnonzero(payload[split_key] == "test")
    if not len(train_indices) or not len(test_indices):
        raise ValueError("Training calibration and independent test partitions are required")
    selected = np.random.default_rng(args.seed).choice(train_indices, min(args.calibration_samples, len(train_indices)), replace=False)
    if set(payload["path"][selected]) & set(payload["path"][test_indices]):
        raise ValueError("Calibration and test clip identifiers overlap")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = ROOT / "outputs/hardware/qnn_quantization"
    temporary_dir.mkdir(parents=True, exist_ok=True)
    fixed_path = temporary_dir / "incident_tcn.fixed.onnx"
    graph = onnx.load(args.model)
    for value in [*graph.graph.input, *graph.graph.output]:
        dimension = value.type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_param")
        dimension.dim_value = 1
    graph = onnx.shape_inference.infer_shapes(graph)
    onnx.checker.check_model(graph)
    onnx.save(graph, fixed_path)
    preprocessed = temporary_dir / "incident_tcn.preprocessed.onnx"
    changed = qnn_preprocess_model(fixed_path, preprocessed)
    quant_input = preprocessed if changed else fixed_path

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.rewind()

        def get_next(self):
            return next(self.iterator, None)

        def rewind(self):
            self.iterator = iter({"raw_features": values[index:index + 1]} for index in selected)

    config = get_qnn_qdq_config(quant_input, Reader(), activation_type=QuantType.QUInt16, weight_type=QuantType.QUInt8, calibration_providers=["CPUExecutionProvider"])
    # ModelProto avoids Windows Unicode-path failures in ONNX file-based shape inference.
    quantize(onnx.load(quant_input), args.output, config)
    quantized_graph = onnx.load(args.output)
    onnx.checker.check_model(quantized_graph)
    opset = next(item.version for item in quantized_graph.opset_import if item.domain == "")
    quant_metadata = {**metadata, "opset": opset, "input_shape": [1, 15, len(columns)], "precision": "A16W8 QDQ", "calibration_split": "train", "calibration_samples": len(selected), "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(quant_metadata, indent=2) + "\n", encoding="utf-8")
    reference = OnnxIncidentPredictor(args.model, args.metadata)
    quantized = OnnxIncidentPredictor(args.output, metadata_path)
    expected = reference.predict_probabilities(values[test_indices])
    actual = quantized.predict_probabilities(values[test_indices])
    target = payload["target"][test_indices]

    def metrics(probabilities):
        predicted = probabilities >= reference.threshold
        tp = int(np.sum(predicted & (target == 1)))
        fp = int(np.sum(predicted & (target == 0)))
        fn = int(np.sum(~predicted & (target == 1)))
        tn = int(np.sum(~predicted & (target == 0)))
        return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "f1": 2 * tp / max(1, 2 * tp + fp + fn), "fpr": fp / max(1, fp + tn)}

    report = {
        "precision": "uint16 activations / uint8 weights", "input_shape": [1, 15, len(columns)], "opset": opset,
        "split_scheme": metadata["split_scheme"], "calibration_partition": "train", "test_partition": "test",
        "calibration_samples": len(selected), "calibration_seed": args.seed,
        "calibration_clip_count": len(set(payload["path"][selected])), "test_samples": len(test_indices),
        "calibration_test_clip_overlap": 0, "frozen_threshold": reference.threshold,
        "probability_mae": float(np.mean(np.abs(expected - actual))),
        "probability_max_error": float(np.max(np.abs(expected - actual))),
        "decision_agreement": float(np.mean((expected >= reference.threshold) == (actual >= reference.threshold))),
        "float32_test": metrics(expected), "qdq_test": metrics(actual),
        "source_model_sha256": metadata["sha256"], "quantized_model_sha256": quant_metadata["sha256"],
        "sequence_archive_sha256": hashlib.sha256(args.sequences.read_bytes()).hexdigest(),
        "onnxruntime_version": ort.__version__, "validation_provider": "CPUExecutionProvider",
        "target_profile_status": "QNN HTP target execution is the next deployment stage",
        "source": "https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html",
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
