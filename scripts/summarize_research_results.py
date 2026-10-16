from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))

from illegal_parking.accidents import load_a1_a2_accident_records
from illegal_parking.deployment_coverage import load_camera_sites
from illegal_parking.research_statistics import paired_cluster_coverage_interval, wilson_interval
from illegal_parking.risk_coverage import aggregate_risk_cells, coverage_index


def main():
    study_dir = ROOT / "docs/results/government_camera_budget"
    protocol = json.loads((study_dir / "protocol.json").read_text(encoding="utf-8"))
    selections = json.loads((study_dir / "selections.json").read_text(encoding="utf-8"))
    selected = {
        row["method"]: row["selected_camera_ids"] for row in selections
        if row["a1_weight"] == 5 and row["radius_m"] == 1000 and row["budget"] == 10
    }
    records = load_a1_a2_accident_records(ROOT / "data/raw/taiwan/taiwan_accidents_a1_a2_113.zip")
    holdout = [record for record in records if record.city == protocol["city"] and record.date >= protocol["holdout_period"][0] and record.category in {"A1", "A2"}]
    cells = aggregate_risk_cells(holdout, a1_weight=5)
    valid_cells = {cell["id"] for cell in cells}
    index = coverage_index(load_camera_sites(ROOT / "data/raw/taiwan/new_taipei_illegal_parking_camera_sites.csv"), cells, 1000)
    greedy = set().union(*(index[key] for key in selected["risk_greedy"]))
    independent = set().union(*(index[key] for key in selected["independent_rank"]))
    clusters = defaultdict(lambda: np.zeros(3))
    for record in holdout:
        if record.latitude is None or record.longitude is None:
            continue
        cell = f"{record.latitude:.3f},{record.longitude:.3f}"
        if cell not in valid_cells:
            continue
        weight = 5 if record.category == "A1" else 1
        clusters[record.area or "Unspecified district"] += np.array([weight, weight * (cell in greedy), weight * (cell in independent)])
    matrix = np.stack(list(clusters.values()))
    uncertainty = {
        "camera_allocation": {
            "contrast": "risk_greedy minus independent_rank", "a1_weight": 5, "radius_m": 1000, "camera_budget": 10,
            "resampling_unit": "New Taipei district", "holdout_weight": float(matrix[:, 0].sum()),
            **paired_cluster_coverage_interval(matrix[:, 0], matrix[:, 1], matrix[:, 2]),
        },
        "event_recall": [],
    }
    with (ROOT / "docs/results/accident_event_metrics.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            lower, upper = wilson_interval(int(row["detected"]), int(row["clips"]))
            uncertainty["event_recall"].append({"profile": row["profile"], "recall": float(row["event_recall"]), "wilson_lower_95": lower, "wilson_upper_95": upper})
    (ROOT / "docs/results/research_uncertainty.json").write_text(json.dumps(uncertainty, indent=2) + "\n", encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with (study_dir / "comparison.csv").open(encoding="utf-8", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if float(row["a1_weight"]) == 5 and int(row["radius_m"]) == 1000]
    fig, axis = plt.subplots(figsize=(7.2, 4.4), layout="constrained")
    for method, label, color, marker in [
        ("risk_greedy", "Coverage-aware greedy", "#08765c", "o"),
        ("independent_rank", "Independent risk ranking", "#3269af", "s"),
        ("random", "Random selection (mean, +/- 1 SD)", "#ba6521", "^"),
    ]:
        method_rows = sorted((row for row in rows if row["method"] == method), key=lambda row: int(row["budget"]))
        x = np.array([int(row["budget"]) for row in method_rows])
        y = np.array([float(row["holdout_risk_coverage_mean"]) * 100 for row in method_rows])
        error = np.array([float(row["holdout_risk_coverage_sd"]) * 100 for row in method_rows])
        axis.errorbar(x, y, yerr=error if method == "random" else None, label=label, color=color, marker=marker, capsize=3, linewidth=1.6)
    axis.set(xlabel="Activated camera budget", ylabel="Held-out crash risk coverage (%)", xticks=[1, 3, 5, 10, 17], ylim=(0, 19))
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#dde3df", linewidth=0.6)
    axis.legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(ROOT / "docs/assets/government_camera_budget.png", dpi=220)
    fig.savefig(ROOT / "docs/assets/government_camera_budget.pdf")
    plt.close(fig)
    print(json.dumps(uncertainty, indent=2))


if __name__ == "__main__":
    main()
