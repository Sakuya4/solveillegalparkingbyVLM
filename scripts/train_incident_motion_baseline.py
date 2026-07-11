from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.incident_baseline import binary_classification_metrics, fit_logistic_regression


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Train an edge-friendly ACCIDENT motion baseline.")
    parser.add_argument("--features", default="data/processed/accident/motion_features.csv")
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), default="iid")
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--model-output", default="models/accident_motion_baseline.json")
    parser.add_argument("--report-output", default="outputs/accident/motion_baseline_report.json")
    parser.add_argument("--predictions-output", default="outputs/accident/motion_baseline_predictions.csv")
    args = parser.parse_args()

    feature_path = _resolve(args.features)
    rows = list(csv.DictReader(feature_path.open("r", encoding="utf-8-sig", newline="")))
    if not rows:
        raise SystemExit(f"Feature CSV has no rows: {feature_path}")
    feature_names = tuple(
        name
        for name in rows[0]
        if name == "motion_peak_position" or name.endswith(("_mean", "_max", "_std"))
    )
    split_field = "iid_split" if args.split_scheme == "iid" else "geographic_split"
    train_rows = [row for row in rows if row[split_field] == "train"]
    test_rows = [row for row in rows if row[split_field] == "test"]
    if not train_rows or not test_rows:
        raise SystemExit(f"Feature CSV must contain train and test rows for {split_field}")

    train_features, train_targets = _matrix(train_rows, feature_names)
    test_features, test_targets = _matrix(test_rows, feature_names)
    model = fit_logistic_regression(
        train_features,
        train_targets,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        threshold=args.threshold,
        feature_names=feature_names,
    )
    train_probabilities = model.predict_proba(train_features)
    test_probabilities = model.predict_proba(test_features)

    report = {
        "model": "standardized_logistic_regression",
        "split_scheme": args.split_scheme,
        "feature_count": len(feature_names),
        "feature_names": list(feature_names),
        "train": binary_classification_metrics(train_targets, train_probabilities, args.threshold),
        "test": binary_classification_metrics(test_targets, test_probabilities, args.threshold),
        "test_by_collision_type": _group_metrics(test_rows, test_targets, test_probabilities, "collision_type", args.threshold),
        "test_by_quality": _group_metrics(test_rows, test_targets, test_probabilities, "quality", args.threshold),
        "limitations": [
            "Normal windows are pre-incident segments from accident clips, not independent normal-only CCTV videos.",
            "False-positive rate is window-level and must not be reported as false alarms per camera-hour.",
        ],
    }

    model_output = _resolve(args.model_output)
    report_output = _resolve(args.report_output)
    predictions_output = _resolve(args.predictions_output)
    for output in (model_output, report_output, predictions_output):
        output.parent.mkdir(parents=True, exist_ok=True)
    model_output.write_text(json.dumps(model.to_dict(), indent=2), encoding="utf-8")
    report_output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_predictions(predictions_output, test_rows, test_probabilities, args.threshold)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _matrix(rows: list[dict[str, str]], feature_names: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray([[float(row[name]) for name in feature_names] for row in rows], dtype=np.float64)
    targets = np.asarray([int(row["target"]) for row in rows], dtype=np.int64)
    return features, targets


def _group_metrics(
    rows: list[dict[str, str]],
    targets: np.ndarray,
    probabilities: np.ndarray,
    field: str,
    threshold: float,
) -> dict[str, dict]:
    groups: dict[str, dict] = {}
    for value in sorted({row[field] for row in rows}):
        indices = np.asarray([index for index, row in enumerate(rows) if row[field] == value])
        groups[value] = binary_classification_metrics(targets[indices], probabilities[indices], threshold)
    return groups


def _write_predictions(
    output_path: Path,
    rows: list[dict[str, str]],
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    fieldnames = ["path", "label", "target", "collision_type", "region", "quality", "probability", "prediction"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, probability in zip(rows, probabilities):
            writer.writerow(
                {
                    **{field: row[field] for field in fieldnames[:6]},
                    "probability": float(probability),
                    "prediction": int(probability >= threshold),
                }
            )


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
