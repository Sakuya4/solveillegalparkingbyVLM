from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "docs/results/full_cohort_model_comparison.json"
    report = json.loads(source.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    colors = {"candidate": "#b33849", "global": "#5b6370", "videomae_frozen": "#286eaa", "videomae_finetuned": "#16856f"}
    for column, split in enumerate(("iid", "geographic")):
        roc, pr = axes[:, column]
        for model in (row for row in report["models"] if row["split"] == split):
            path = ROOT / "outputs/accident" / f'{model["artifact_stem"]}_predictions.csv'
            if hashlib.sha256(path.read_bytes()).hexdigest() != model["predictions_sha256"]:
                raise ValueError("Prediction hash differs from the verified comparison")
            with path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            targets = [int(row["target"]) for row in rows]
            scores = [float(row["probability"]) for row in rows]
            fpr, tpr, _ = roc_curve(targets, scores)
            precision, recall, _ = precision_recall_curve(targets, scores)
            color = colors[model["feature_set"]]
            roc.plot(fpr, tpr, color=color, label=f'{model["display_name"]} ({model["ranking"]["auroc"]:.3f})')
            pr.plot(recall, precision, color=color, label=f'{model["display_name"]} ({model["ranking"]["average_precision"]:.3f})')
            roc.scatter(model["test"]["false_positive_rate"], model["test"]["recall"], color=color, s=22, zorder=5)
        roc.plot([0, 1], [0, 1], "--", color="#a5a9ac", linewidth=1)
        pr.axhline(model["ranking"]["positive_prevalence"], linestyle="--", color="#a5a9ac", linewidth=1)
        roc.set(xlabel="False-positive rate", ylabel="True-positive rate", title=f'{split.upper() if split == "iid" else "Geographic"} | ROC')
        pr.set(xlabel="Recall", ylabel="Precision", title=f'{split.upper() if split == "iid" else "Geographic"} | Precision-recall')
        for axis in (roc, pr):
            axis.set_xlim(0, 1)
            axis.set_ylim(0, 1.02)
            axis.grid(alpha=0.2)
            axis.legend(fontsize=8, loc="lower right" if axis is roc else "lower left", frameon=False)
    for extension in ("png", "pdf"):
        figure.savefig(ROOT / "docs/assets" / f"full_cohort_model_curves.{extension}", dpi=160)
    plt.close(figure)
    print(f"Plotted eight verified model/split results from {hashlib.sha256(source.read_bytes()).hexdigest()}")


if __name__ == "__main__":
    main()
