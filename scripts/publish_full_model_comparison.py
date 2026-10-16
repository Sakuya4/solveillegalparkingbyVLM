from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.research_statistics import paired_cluster_f1_interval
from scripts.publish_full_cohort_results import verify_predictions


def main():
    parser = argparse.ArgumentParser(description="Publish matched full-cohort temporal representations and paired uncertainty.")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/full_cohort_model_comparison.json")
    args = parser.parse_args()
    import sklearn
    from sklearn.metrics import average_precision_score, roc_auc_score
    from threadpoolctl import threadpool_limits

    threadpool_limits(2)
    sequence_path = ROOT / "data/processed/accident/candidate_roi_sequences_full.npz"
    payload = np.load(sequence_path, allow_pickle=False)
    tcn = json.loads((ROOT / "docs/results/full_cohort_temporal.json").read_text(encoding="utf-8"))
    feature_path = ROOT / "data/processed/accident/videomae_small_features_full.csv"
    feature_report = json.loads((ROOT / "outputs/accident/videomae_full_features_report.json").read_text(encoding="utf-8"))
    if feature_report["failure_count"] or feature_report["extracted_window_count"] != len(payload["target"]):
        raise ValueError("VideoMAE extraction must match the complete temporal cohort")
    specs = [("candidate", "Candidate ROI TCN", "candidate_full_tcn", ".pt"), ("global", "Global motion TCN", "global_full_tcn", ".pt"), ("videomae_frozen", "Frozen VideoMAE", "videomae_full_frozen", ".json"), ("videomae_finetuned", "Fine-tuned VideoMAE", "videomae_full_finetuned", ".pt")]
    models, scores, pairs = [], {}, []
    for split in ("iid", "geographic"):
        test = payload[split + "_split"] == "test"
        keys = list(zip(payload["path"][test].astype(str), payload["label"][test].astype(str)))
        labels = payload["target"][test]
        counts = next(row for row in tcn["models"] if row["split"] == split)
        for feature, display, prefix, extension in specs:
            stem = f"{prefix}_{split}_fpr20"
            path = ROOT / "outputs/accident" / f"{stem}_report.json"
            report = json.loads(path.read_text(encoding="utf-8"))
            with path.with_name(f"{stem}_predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            verify_predictions(payload, predictions, split, report)
            indexed = {(row["path"], row["label"]): row for row in predictions}
            probabilities = np.asarray([float(indexed[key]["probability"]) for key in keys])
            scores[(feature, split)] = (probabilities, report["decision_threshold"]["effective"])
            row = {
                "feature_set": feature, "display_name": display, "split": split, "artifact_stem": stem,
                **{key: counts[key] for key in ("fit_clips", "calibration_clips", "test_clips")},
                "test": report["test"], "calibration": report["calibration"],
                "decision_threshold": report["decision_threshold"],
                "ranking": {"auroc": float(roc_auc_score(labels, probabilities)), "average_precision": float(average_precision_score(labels, probabilities)), "positive_prevalence": float(labels.mean())},
                "model_sha256": hashlib.sha256((ROOT / "models" / (stem + extension)).read_bytes()).hexdigest(),
                "predictions_sha256": hashlib.sha256(path.with_name(f"{stem}_predictions.csv").read_bytes()).hexdigest(),
                "training": {key: report[key] for key in ("seed", "training_seed", "epochs", "best_epoch", "epochs_completed", "learning_rate", "head_learning_rate", "parameters", "model_revision", "elapsed_sec") if key in report},
            }
            models.append(row)
        for a, b in (("videomae_finetuned", "videomae_frozen"), ("candidate", "global")):
            a_scores, a_threshold = scores[(a, split)]
            b_scores, b_threshold = scores[(b, split)]
            pairs.append({"split": split, "model_a": a, "model_b": b, **paired_cluster_f1_interval(labels, a_scores, b_scores, payload["path"][test], threshold_a=a_threshold, threshold_b=b_threshold)})
    result = {
        "eligible_clips": tcn["eligible_clips"], "windows": tcn["windows"],
        "sequence_sha256": hashlib.sha256(sequence_path.read_bytes()).hexdigest(),
        "videomae_features_sha256": hashlib.sha256(feature_path.read_bytes()).hexdigest(),
        "videomae_checkpoint_revision": feature_report["model_revision"],
        "sklearn_version": sklearn.__version__, "models": models, "paired_f1_differences": pairs,
        "uncertainty": "Paired source-clip percentile bootstrap; 2,000 repetitions, seed 42; fixed trained models and thresholds",
        "scope": "Common annotation-defined windows and official partitions; AUROC/AP are window-level ranking endpoints",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["model", "split", "test_clips", "test_windows", "f1", "false_positive_rate", "auroc", "average_precision", "positive_prevalence"])
        writer.writeheader()
        for row in models:
            writer.writerow({"model": row["display_name"], "split": row["split"], "test_clips": row["test_clips"], "test_windows": row["test"]["sample_count"], "f1": row["test"]["f1"], "false_positive_rate": row["test"]["false_positive_rate"], **row["ranking"]})
    print(json.dumps(pairs, indent=2))


if __name__ == "__main__":
    main()
