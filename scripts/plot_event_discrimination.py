from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "docs/results/event_discrimination_results.json"
    study = json.loads(source.read_text(encoding="utf-8"))
    indexed = {row["model"]: row["summary"] for row in study["models"]}
    figure, axes = plt.subplots(2, 2, figsize=(13, 8.2), layout="constrained", gridspec_kw={"width_ratios": [1, 1.7]})
    labels = ["Source / candidate", "Source / global", "Fixed / candidate", "Fixed / global"]
    categories = [("detected", "Detected", "#25816b"), ("structurally_unreachable", "No admissible pair", "#9a9fa3"),
                  ("no_target_positive_pair", "No positive target pair", "#4b7896"), ("onset_suppression", "Onset suppression", "#ba6654")]
    for row, split in enumerate(("iid", "geographic")):
        models = [f"{arm}_{feature}_{split}" for arm in ("legacy", "fixed") for feature in ("candidate", "global")]
        axis = axes[row, 0]
        for offset, field, color, label in ((-.17, "pooled", "#487b98", "Pooled contexts"), (.17, "within_clip_macro", "#2d876c", "Within-clip macro")):
            values = [indexed[key][field]["auroc"] for key in models]
            axis.barh(np.arange(4) + offset, values, height=.28, color=color, label=label)
            for index, value in enumerate(values):
                axis.text(value + .015, index + offset, f"{value:.3f}", va="center", fontsize=9)
        axis.axvline(.5, color="#59646a", linewidth=1, linestyle="--")
        axis.set(yticks=range(4), yticklabels=labels, xlim=(0, 1), xlabel="AUROC", title=f"TRAIN {'IID' if split == 'iid' else 'geographic'} ranking")
        axis.invert_yaxis()
        axis = axes[row, 1]
        groups = [(key, policy) for key in models for policy in ("onset_only", "positive_context")]
        left = np.zeros(len(groups))
        for category, label, color in categories:
            values = [indexed[key]["policy_decomposition"][policy][category] / indexed[key]["clip_count"] for key, policy in groups]
            axis.barh(range(8), values, left=left, height=.66, color=color, label=label)
            for index, (start, value) in enumerate(zip(left, values)):
                if value >= .09:
                    axis.text(start + value / 2, index, f"{value:.0%}", ha="center", va="center", color="white", fontsize=9)
            left += values
        axis.set(yticks=range(8), yticklabels=[f"{label} / {policy}" for label in labels for policy in ("onset", "capped")],
                 xlim=(0, 1), xlabel="Fraction of all assigned TRAIN-holdout clips", title=f"Deterministic controller decomposition / n={indexed[models[0]]['clip_count']}")
        axis.invert_yaxis()
        for axis in axes[row]:
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="x", alpha=.15)
            axis.set_axisbelow(True)
    handles, names = axes[0, 1].get_legend_handles_labels()
    ranking_handles, ranking_names = axes[0, 0].get_legend_handles_labels()
    figure.legend(ranking_handles + handles, ranking_names + names, loc="outside lower center", ncol=3, frameon=False, fontsize=9)
    outputs = []
    for suffix in ("png", "pdf"):
        path = ROOT / f"docs/assets/event_discrimination.{suffix}"
        figure.savefig(path, dpi=180, metadata={"CreationDate": None} if suffix == "pdf" else None)
        outputs.append(path)
    plt.close(figure)
    record = {"results_sha256": digest(source), "plotter_sha256": digest(Path(__file__)),
              "outputs": {path.name: digest(path) for path in outputs},
              "scope": "Exploratory corresponding-TRAIN-holdout diagnostics. Pooled and within-clip AUROC have different weighting. Controller fractions retain all assigned clips, including unsupported and empty contexts. No new test endpoints."}
    (ROOT / "docs/assets/event_discrimination.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
