from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.incident_baseline import binary_classification_metrics, grouped_calibration_indices


def verify_predictions(payload, predictions: list[dict], split: str, report: dict) -> None:
    test = payload[split + "_split"] == "test"
    keys = list(zip(payload["path"][test].astype(str), payload["label"][test].astype(str)))
    indexed = {(row["path"], row["label"]): row for row in predictions}
    if len(indexed) != len(predictions) or set(indexed) != set(keys):
        raise ValueError("Prediction rows must match the complete test cohort exactly")
    targets = payload["target"][test]
    rows = [indexed[key] for key in keys]
    if not np.array_equal(np.asarray([int(row["target"]) for row in rows]), targets):
        raise ValueError("Prediction targets differ from the sequence archive")
    probabilities = np.asarray([float(row["probability"]) for row in rows])
    if not np.all(np.isfinite(probabilities)) or np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Probabilities must be finite and in [0, 1]")
    actual = binary_classification_metrics(targets, probabilities, threshold=report["decision_threshold"]["effective"])
    if actual["confusion_matrix"] != report["test"]["confusion_matrix"] or any(not np.isclose(value, report["test"][key]) for key, value in actual.items() if key != "confusion_matrix"):
        raise ValueError("Published metrics differ from stored predictions")


def main():
    parser = argparse.ArgumentParser(description="Verify and publish the full-cohort temporal feature comparison.")
    parser.add_argument("--sequences", type=Path, default=ROOT / "data/processed/accident/candidate_roi_sequences_full.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/full_cohort_temporal.json")
    args = parser.parse_args()
    payload = np.load(args.sequences, allow_pickle=False)
    metadata = ROOT / "data/raw/accident/metadata-real.csv"
    clips = load_accident_manifest(metadata)
    extraction = json.loads((ROOT / "outputs/accident/candidate_roi_features_full_report.json").read_text(encoding="utf-8"))
    if extraction["failure_count"]:
        raise ValueError("Publish only a complete failure-free extraction")
    represented = set(payload["path"].astype(str))
    omitted = [clip for clip in clips if clip.relative_path.as_posix() not in represented]
    if any(clip.frame_count >= 32 for clip in omitted):
        raise ValueError("Unexpected eligible clips are missing from the sequence archive")
    models = []
    for feature in ("candidate", "global"):
        for split in ("iid", "geographic"):
            stem = f"{feature}_full_tcn_{split}_fpr20"
            path = ROOT / "outputs/accident" / f"{stem}_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            with path.with_name(f"{stem}_predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            verify_predictions(payload, predictions, split, report)
            train = np.flatnonzero(payload[split + "_split"] == "train")
            fit, calibration = grouped_calibration_indices(payload["path"][train], fraction=0.2, seed=42)
            models.append({
                "feature_set": feature, "split": split,
                "model_sha256": hashlib.sha256((ROOT / "models" / f"{stem}.pt").read_bytes()).hexdigest(),
                "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "fit_clips": len(set(payload["path"][train[fit]])),
                "calibration_clips": len(set(payload["path"][train[calibration]])),
                "test_clips": len(set(payload["path"][payload[split + "_split"] == "test"])),
                "decision_threshold": report["decision_threshold"],
                "feature_names": report["feature_names"],
                **{key: report[key] for key in ("seed", "epochs", "channels", "dropout", "fit", "calibration", "test")},
            })
    result = {
        "requested_clips": len(clips), "eligible_clips": len(represented),
        "windows": len(payload["target"]), "sequence_steps": int(payload["features"].shape[1]),
        "window_frames": 32, "frame_stride": 2, "negative_gap_frames": 8,
        "excluded_short_clips": [{"path": clip.relative_path.as_posix(), "frame_count": clip.frame_count} for clip in omitted],
        "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
        "sequences_sha256": hashlib.sha256(args.sequences.read_bytes()).hexdigest(),
        "extraction": {key: extraction[key] for key in ("failure_count", "elapsed_sec", "runtime_proposals", "oracle_diagnostics_only")},
        "evaluation_unit": "Annotation-defined temporal windows; incident boxes enter diagnostics only",
        "comparison_scope": "Full eligible cohort, common windows, official IID/geographic partitions; distinct from the balanced 500-clip reference study",
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ("feature_set", "split", "fit_clips", "calibration_clips", "test_clips", "test")} for row in models], indent=2))


if __name__ == "__main__":
    main()
