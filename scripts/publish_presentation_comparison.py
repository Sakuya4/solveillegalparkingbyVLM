from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from scripts.publish_full_cohort_results import verify_predictions
from scripts.train_presentation_tcn import validate_frozen_sequences
from illegal_parking.research_statistics import paired_cluster_f1_interval


def paired_contrasts(protocol, scores):
    pairs = []
    for feature in ("candidate", "global"):
        for split in ("iid", "geographic"):
            labels, fixed, fixed_threshold, clips = scores[("fixed", feature, split)]
            other_labels, legacy, legacy_threshold, other_clips = scores[("legacy", feature, split)]
            if not np.array_equal(labels, other_labels) or not np.array_equal(clips, other_clips):
                raise ValueError("Paired temporal contrasts require identical labels and source-clip order")
            pairs.append({"feature": feature, "split": split, "contrast": "fixed minus legacy",
                          **paired_cluster_f1_interval(labels, fixed, legacy, clips, threshold_a=fixed_threshold,
                                                      threshold_b=legacy_threshold,
                                                      repeats=protocol["evaluation"]["bootstrap_resamples"], seed=protocol["evaluation"]["seed"])})
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Verify all eight paired temporal fits and publish both signs of the predeclared contrasts.")
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/results/presentation_protocol.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/presentation_comparison.json")
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".csv").exists():
        raise FileExistsError("Published comparisons are immutable; use a new output path")
    import sklearn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from threadpoolctl import threadpool_limits

    threadpool_limits(2)
    models, scores, extractions = [], {}, {}
    for arm in ("legacy", "fixed"):
        directory = ROOT / "data/processed/accident" / ("presentation_" + arm)
        for split in ("iid", "geographic"):
            protocol, extraction, sequence = validate_frozen_sequences(args.protocol, directory, split)
            extractions[arm] = extraction
            with np.load(sequence, allow_pickle=False) as payload:
                test = payload[split + "_split"] == "test"
                keys = list(zip(payload["path"][test].astype(str), payload["label"][test].astype(str)))
                labels, clips = payload["target"][test], payload["path"][test]
                for feature in ("candidate", "global"):
                    stem = f"presentation_{arm}_{feature}_{split}"
                    report_path = ROOT / "outputs/accident/presentation" / (stem + "_report.json")
                    prediction_path = report_path.with_name(stem + "_predictions.csv")
                    model = ROOT / "models" / (stem + ".pt")
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    if report["protocol_sha256"] != sha256(args.protocol) or report["sequences_sha256"] != sha256(sequence):
                        raise ValueError("Model report does not match its frozen input")
                    if report["model_sha256"] != sha256(model) or report["predictions_sha256"] != sha256(prediction_path):
                        raise ValueError("Model or prediction bytes changed after fitting")
                    with prediction_path.open(encoding="utf-8", newline="") as handle:
                        predictions = list(csv.DictReader(handle))
                    verify_predictions(payload, predictions, split, report)
                    indexed = {(row["path"], row["label"]): float(row["probability"]) for row in predictions}
                    probabilities = np.asarray([indexed[key] for key in keys])
                    scores[(arm, feature, split)] = (labels, probabilities, report["decision_threshold"]["effective"], clips)
                    counts = {part + "_clips": len(set(payload["path"][payload[split + "_partition"] == part])) for part in ("fit", "calibration", "test")}
                    models.append({"arm": arm, "feature": feature, "split": split, **counts,
                                   "decision_threshold": report["decision_threshold"], "test": report["test"], "calibration": report["calibration"],
                                   "auroc": float(roc_auc_score(labels, probabilities)), "average_precision": float(average_precision_score(labels, probabilities)),
                                   "positive_prevalence": float(labels.mean()),
                                   "training_torch_version": report["training_torch_version"], "gpu_name": report["gpu_name"],
                                   "model_sha256": sha256(model), "predictions_sha256": sha256(prediction_path), "report_sha256": sha256(report_path)})
    if extractions["legacy"]["versions"] != extractions["fixed"]["versions"] or extractions["legacy"]["device"] != extractions["fixed"]["device"]:
        raise ValueError("Extraction runtime and device must match between arms")
    if len({(row["training_torch_version"], row["gpu_name"]) for row in models}) != 1:
        raise ValueError("Matched fits must use the same training runtime and hardware")
    pairs = paired_contrasts(protocol, scores)
    result = {"protocol_sha256": sha256(args.protocol), "paired_clips": protocol["paired_clips"], "paired_windows": protocol["paired_windows"],
              "models": models, "paired_f1_contrasts": pairs, "training": protocol["training"],
              "extraction_sha256": {arm: sha256(ROOT / "data/processed/accident" / ("presentation_" + arm) / "extraction.json") for arm in extractions},
              "sklearn_version": sklearn.__version__, "interpretation": protocol["interpretation"],
              "evaluation_scope": "Paired annotation-defined windows under verified presentation clocks; continuous event evaluation is a separate endpoint",
              "uncertainty_scope": "Four pointwise source-clip bootstrap contrasts conditional on fitted models and thresholds; no family-wise correction",
              "default_policy": protocol["default_policy"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["arm", "feature", "split", "test_clips", "test_windows", "f1", "false_positive_rate", "auroc", "average_precision", "positive_prevalence"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in models:
            writer.writerow({**{key: row[key] for key in ("arm", "feature", "split", "test_clips", "auroc", "average_precision", "positive_prevalence")},
                             "test_windows": row["test"]["sample_count"], "f1": row["test"]["f1"], "false_positive_rate": row["test"]["false_positive_rate"]})
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
