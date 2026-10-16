from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.extract_presentation_features import load_manifest
from scripts.prepare_presentation_protocol import sha256
from illegal_parking.incident_baseline import grouped_calibration_indices
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES


def validate_frozen_sequences(protocol_path, directory, split):
    protocol, rows = load_manifest(protocol_path)
    report = json.loads((directory / "extraction.json").read_text(encoding="utf-8"))
    sequence = directory / "sequences.npz"
    if report["protocol_sha256"] != sha256(protocol_path) or report["sequences_sha256"] != sha256(sequence) or report["smoke_clips"] is not None:
        raise ValueError("Training requires a complete hash-matched extraction")
    with np.load(sequence, allow_pickle=False) as data:
        if str(data["protocol_sha256"]) != sha256(protocol_path) or str(data["arm"]) != report["arm"]:
            raise ValueError("Sequence provenance differs from extraction")
        if data["features"].shape != (len(rows), 15, len(CANDIDATE_MOTION_FEATURE_NAMES)) or not np.all(np.isfinite(data["features"])):
            raise ValueError("Sequence feature shape or values differ from the declared model input")
        if not np.array_equal(data["feature_names"], np.asarray(CANDIDATE_MOTION_FEATURE_NAMES)):
            raise ValueError("Sequence feature channels differ from the annotation-free candidate/global definition")
        for key in ("path", "label", "target", "iid_split", "geographic_split", "iid_partition", "geographic_partition"):
            if not np.array_equal(data[key], np.asarray([row[key] for row in rows])):
                raise ValueError("Sequence rows differ from frozen membership")
        if not np.array_equal(data["frame_indices"], np.asarray([row[report["arm"] + "_indices"] for row in rows])):
            raise ValueError("Source-frame indices differ from the frozen manifest")
        if not np.allclose(data["frame_times_sec"], np.asarray([row[report["arm"] + "_times_sec"] for row in rows]), atol=1e-9, rtol=0):
            raise ValueError("Presentation timestamps differ from the frozen manifest")
        train = np.flatnonzero(data[split + "_split"] == "train")
        fit, calibration = grouped_calibration_indices(data["path"][train], protocol["training"]["calibration_fraction"], protocol["training"]["seed"])
        if not np.all(data[split + "_partition"][train[fit]] == "fit") or not np.all(data[split + "_partition"][train[calibration]] == "calibration"):
            raise ValueError("Reconstructed training holdout differs from the frozen partition")
    for relative, digest in protocol["implementation_sha256"].items():
        if sha256(ROOT / relative) != digest:
            raise ValueError("Frozen training implementation changed")
    for name, digest in protocol["reference_model_sha256"].items():
        if sha256(ROOT / "Application/desktop/models" / name) != digest:
            raise ValueError("Reference model changed during the paired study")
    return protocol, report, sequence


def main():
    parser = argparse.ArgumentParser(description="Fit the existing TCN under the frozen presentation-study membership and hyperparameters.")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/results/presentation_protocol.json")
    parser.add_argument("--features-dir", type=Path, required=True)
    parser.add_argument("--feature", choices=("candidate_", "global_"), required=True)
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    protocol, extraction, sequence = validate_frozen_sequences(args.protocol, args.features_dir, args.split_scheme)
    stem = f"presentation_{extraction['arm']}_{args.feature.rstrip('_')}_{args.split_scheme}"
    model = ROOT / "models" / (stem + ".pt")
    report_path = ROOT / "outputs/accident/presentation" / (stem + "_report.json")
    predictions = report_path.with_name(stem + "_predictions.csv")
    if any(path.exists() for path in (model, report_path, predictions)):
        raise FileExistsError("Presentation fit artifacts are immutable; inspect existing outputs before resuming")
    settings = protocol["training"]
    arguments = ["train_incident_tcn.py", "--sequences", str(sequence), "--split-scheme", args.split_scheme,
                 "--include-prefix", args.feature, "--device", args.device, "--threads", "2",
                 "--model-output", str(model), "--report-output", str(report_path), "--predictions-output", str(predictions)]
    for key in ("epochs", "channels", "dropout", "batch_size", "learning_rate", "weight_decay", "seed", "target_train_fpr", "calibration_fraction"):
        arguments.extend(["--" + key.replace("_", "-"), str(settings[key])])
    from scripts import train_incident_tcn
    original = sys.argv
    try:
        sys.argv = arguments
        code = train_incident_tcn.main()
    finally:
        sys.argv = original
    if code:
        raise RuntimeError(f"Frozen TCN fitting failed: {code}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update({"protocol_sha256": sha256(args.protocol), "sequences_sha256": extraction["sequences_sha256"],
                   "extraction_sha256": sha256(args.features_dir / "extraction.json"), "arm": extraction["arm"],
                   "model_sha256": sha256(model), "predictions_sha256": sha256(predictions),
                   "fit_wrapper_sha256": sha256(Path(__file__)), "time_contract": protocol["clock"]})
    report["training_torch_version"] = train_incident_tcn.torch.__version__
    report["cuda_runtime"] = train_incident_tcn.torch.version.cuda
    report["gpu_name"] = (train_incident_tcn.torch.cuda.get_device_name(train_incident_tcn.torch.device(report["device"]))
                          if str(report.get("device", "cpu")).startswith("cuda") else None)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(report_path), "model_sha256": report["model_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
