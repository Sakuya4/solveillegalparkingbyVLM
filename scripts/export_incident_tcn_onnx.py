from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.incident_deployment import StandardizedIncidentTcn
from illegal_parking.incident_tcn_inference import IncidentTcnPredictor


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Export the calibrated incident TCN preprocessing and model to ONNX."
    )
    parser.add_argument("--model", default="models/candidate500_tcn_only_iid_fpr20.pt")
    parser.add_argument(
        "--sequences", default="data/processed/accident/candidate_roi_sequences_500.npz"
    )
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="outputs/hardware/incident_tcn.onnx")
    parser.add_argument("--report", default="outputs/hardware/incident_tcn_onnx_report.json")
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise SystemExit("--sample-count must be positive")

    import onnx
    import onnxruntime as ort

    model_path = _resolve(args.model)
    sequence_path = _resolve(args.sequences)
    output_path = _resolve(args.output)
    report_path = _resolve(args.report)
    predictor = IncidentTcnPredictor.from_checkpoint(model_path, "cpu")
    deployment = StandardizedIncidentTcn(predictor.model, predictor.standardizer).eval()
    sequence_payload = np.load(sequence_path, allow_pickle=False)
    all_feature_names = tuple(sequence_payload["feature_names"].astype(str).tolist())
    feature_indices = [all_feature_names.index(name) for name in predictor.feature_names]
    features = np.asarray(sequence_payload["features"], dtype=np.float32)[:, :, feature_indices]
    generator = np.random.default_rng(args.seed)
    sample_indices = generator.choice(
        len(features), size=min(args.sample_count, len(features)), replace=False
    )
    samples = features[sample_indices]
    example = torch.from_numpy(samples[:1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        deployment,
        (example,),
        output_path,
        input_names=["raw_features"],
        output_names=["incident_probability"],
        dynamic_axes={"raw_features": {0: "batch"}, "incident_probability": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        torch_output = deployment(torch.from_numpy(samples)).numpy()
    onnx_output = session.run(None, {"raw_features": samples})[0]
    absolute_error = np.abs(torch_output - onnx_output)
    qai_config = Path.home() / ".qai_hub" / "client.ini"
    report = {
        "source_model": str(model_path),
        "onnx_model": str(output_path),
        "opset": 18,
        "input_name": "raw_features",
        "input_shape": ["batch", features.shape[1], features.shape[2]],
        "output_name": "incident_probability",
        "feature_names": list(predictor.feature_names),
        "threshold": predictor.threshold,
        "parity_samples": len(samples),
        "max_absolute_error": float(absolute_error.max()),
        "mean_absolute_error": float(absolute_error.mean()),
        "onnx_check_passed": True,
        "qai_hub": {
            "client_configured": qai_config.is_file(),
            "target_device": "Samsung Galaxy S23",
            "target_runtime": "qnn_dlc",
            "compute_unit": "npu",
            "submission_command": (
                "qai-hub submit-compile-job --model outputs/hardware/incident_tcn.onnx "
                "--device \"Samsung Galaxy S23\" "
                "--compile_options \"--target_runtime qnn_dlc --compute_unit npu\""
            ),
            "status": (
                "ready_for_authenticated_submission"
                if not qai_config.is_file()
                else "client_config_present_submission_not_requested"
            ),
        },
        "sources": [
            "https://workbench.aihub.qualcomm.com/docs/hub/generated/qai_hub.submit_compile_job.html",
            "https://workbench.aihub.qualcomm.com/docs/hub/faq.html",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
