from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "docs/results/continuous_context_results.json"
    study = json.loads(source.read_text())
    indexed = {row["model"]: row for row in study["models"]}
    groups = [(feature, split) for feature in ("candidate", "global") for split in ("iid", "geographic")]
    labels = [f"{feature.title()}\n{'IID' if split == 'iid' else 'Geographic'}" for feature, split in groups]
    colors = {"anchored": "#167D8D", "continuous": "#B94A62"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.2), layout="constrained")
    fields = [("Event recall", lambda value: value["summary"]["event_recall"], None),
              ("Off-target episodes / accident clip", lambda value: value["summary"]["off_target_episodes_per_clip"], .5),
              ("Macro prefix-positive fraction", lambda value: value["pre_event"]["macro_positive_fraction"], .2)]
    for axis, (title, getter, cap) in zip(axes.flat, fields):
        for arm, offset in (("anchored", -.18), ("continuous", .18)):
            values = [getter(indexed[f"{arm}_{feature}_{split}"]["test"]) if indexed[f"{arm}_{feature}_{split}"]["test"] else np.nan for feature, split in groups]
            bars = axis.bar(np.arange(4) + offset, values, width=.34, label=arm.title(), color=colors[arm])
            for bar, value in zip(bars, values):
                if np.isfinite(value):
                    axis.annotate(f"{value:.2f}", (bar.get_x() + bar.get_width() / 2, value), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)
        if cap is not None:
            axis.axhline(cap, color="#444444", linestyle="--", linewidth=1, label="TRAIN cap")
        axis.set_xticks(range(4), labels)
        axis.set_title(title, loc="left", fontsize=11)
        axis.set_ylim(bottom=0)
        if title != "Off-target episodes / accident clip":
            axis.set_ylim(0, 1.1)
        else:
            axis.margins(y=.2)
        axis.grid(axis="y", alpha=.15)
        axis.set_axisbelow(True)
    axes[0, 0].legend(frameon=False, ncols=2)
    axis = axes[1, 1]
    for index, row in enumerate(study["contrasts"]):
        value = row["paired_interval"]
        if value:
            center, lower, upper = (100 * value[field] for field in ("difference", "lower_95", "upper_95"))
            axis.errorbar(center, index, xerr=[[center - lower], [upper - center]], fmt="o", color=colors["continuous"], capsize=3)
        else:
            axis.text(0, index, "Unselected", va="center")
    axis.axvline(0, color="#444444", linestyle="--", linewidth=1)
    axis.set_yticks(range(4), [label.replace("\n", " / ") for label in labels])
    axis.invert_yaxis()
    axis.set_title("Paired event-recall change", loc="left", fontsize=11)
    axis.set_xlabel("Continuous minus anchored (percentage points)")
    axis.grid(axis="x", alpha=.15)
    fig.suptitle("Continuous-context training: fixed models and TRAIN-selected operating policies", fontsize=12)
    stem = ROOT / "docs/assets/continuous_context_comparison"
    fig.savefig(stem.with_suffix(".png"), dpi=170)
    fig.savefig(stem.with_suffix(".pdf"))
    plt.close(fig)
    manifest = {"results_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "plotter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "figures": {extension: hashlib.sha256(stem.with_suffix("." + extension).read_bytes()).hexdigest() for extension in ("png", "pdf")},
                "scope": "All eight predeclared policies and four pointwise paired source-clip intervals; conditional on fixed fitted models/calibration. Missing policies are not imputed as zero."}
    stem.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
