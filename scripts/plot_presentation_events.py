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
    result_path = ROOT / "docs/results/presentation_event_comparison.json"
    control_path = ROOT / "docs/results/presentation_event_controls.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))
    controls = json.loads(control_path.read_text(encoding="utf-8"))
    if result["protocol_sha256"] != controls["protocol_sha256"] or result["test_batch_sha256"] != controls["test_batch_sha256"]:
        raise ValueError("Model and control figures require the same frozen test replay")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    labels = ["Source-frame / candidate", "Source-frame / global", "Fixed-time / candidate", "Fixed-time / global", "Clock-only positive", "Always negative"]
    colors = ["#437798", "#437798", "#23856e", "#23856e", "#626a68", "#adb4b1"]
    maximum_demand = max([row["test"]["off_target_episodes_per_clip"] for row in result["models"] if row["test"]]
                         + [row["off_target_episodes_per_clip"] for cohorts in controls["controls"].values() for row in cohorts["test"].values()])
    for row, split in enumerate(("iid", "geographic")):
        models = {(value["arm"], value["feature"]): value for value in result["models"] if value["split"] == split}
        outcomes = [models[(arm, feature)]["test"] for arm in ("legacy", "fixed") for feature in ("candidate", "global")]
        outcomes.extend(controls["controls"][split]["test"][name] for name in ("clock_only_positive", "always_negative"))
        for column, (field, xlabel) in enumerate((("event_recall", "Test event recall"), ("off_target_episodes_per_clip", "Off-target episodes per accident clip"))):
            axis = axes[row, column]
            values = [outcome[field] if outcome else np.nan for outcome in outcomes]
            bars = axis.barh(range(6), values, color=colors)
            for index in (1, 3):
                bars[index].set_hatch("//")
            axis.set_yticks(range(6), labels if column == 0 else [""] * 6)
            axis.invert_yaxis()
            axis.set(xlabel=xlabel, title=f"{'IID' if split == 'iid' else 'Geographic'} / n={outcomes[-1]['clip_count']}")
            axis.set_xlim(0, 1.05 if column == 0 else max(.75, maximum_demand * 1.25))
            if column == 1:
                axis.axvline(.5, color="#a34c49", linestyle="--", linewidth=1.2, label="Calibration budget 0.5")
                axis.legend(frameon=False, fontsize=9, loc="lower right")
            for index, value in enumerate(values):
                if np.isfinite(value):
                    axis.text(value + .012, index, f"{value:.3f}", va="center", fontsize=9)
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="x", alpha=.2)
            axis.set_axisbelow(True)
    outputs = []
    for suffix in ("png", "pdf"):
        output = ROOT / f"docs/assets/presentation_event_controls.{suffix}"
        figure.savefig(output, dpi=180)
        outputs.append(output)
    plt.close(figure)
    metadata = {"inputs": {path.relative_to(ROOT).as_posix(): digest(path) for path in (result_path, control_path)},
                "plotter_sha256": digest(Path(__file__)), "outputs": {path.name: digest(path) for path in outputs},
                "scope": "Point estimates on fixed event cohorts; model-minus-clock uncertainty is in the linked numerical study"}
    (ROOT / "docs/assets/presentation_event_controls.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
