from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT / "docs/results/positive_exposure_results.json"
    study = json.loads(source.read_text(encoding="utf-8"))
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.4), layout="constrained")
    models = [(arm, feature) for arm in ("legacy", "fixed") for feature in ("candidate", "global")]
    labels = ["Source-frame / candidate", "Source-frame / global", "Fixed-time / candidate", "Fixed-time / global", "Clock-only positive", "Always negative"]
    fields = [("test", "event_recall", "Event recall", None),
              ("pre_event", "macro_positive_fraction", "Macro pre-event positive fraction", .2),
              ("test", "off_target_episodes_per_clip", "Off-target episodes / accident clip", .5)]
    for row, split in enumerate(("iid", "geographic")):
        indexed = {(value["model"], value["policy"]): value for value in study["policies"]}
        for column, (section, field, title, cap) in enumerate(fields):
            axis = axes[row, column]
            for policy, offset, color, label in (("onset_only", -.17, "#487c9a", "Onset-only"), ("positive_context", .17, "#29866b", "Positive-context cap")):
                values = [indexed[(f"{arm}_{feature}_{split}", policy)][section][field] for arm, feature in models]
                axis.barh(np.arange(4) + offset, values, height=.29, color=color, label=label)
                for index, value in enumerate(values):
                    axis.text(value + .015, index + offset, f"{value:.2f}", va="center", fontsize=8)
            controls = [study["controls"][split][name][section][field] for name in ("clock_only_positive", "always_negative")]
            axis.barh([4, 5], controls, color="#8c9394", height=.5, label="Nonvisual control")
            for index, value in enumerate(controls, 4):
                axis.text(value + .015, index, f"{value:.2f}", va="center", fontsize=8)
            axis.set_yticks(range(6), labels if column == 0 else [""] * 6)
            axis.invert_yaxis()
            axis.set(xlim=(0, 1.25 if column == 2 else 1.14), xlabel=title, title=f"{'IID' if split == 'iid' else 'Geographic'} / 100 clips")
            if cap is not None:
                axis.axvline(cap, color="#b04746", linestyle="--", linewidth=1, label=f"Train constraint {cap:g}")
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="x", alpha=.2)
            axis.set_axisbelow(True)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    handles.extend([Line2D([], [], color="#b04746", linestyle="--")])
    labels.extend(["TRAIN constraints: prefix 0.2; episodes 0.5"])
    figure.legend(handles, labels, loc="outside lower center", ncol=4, frameon=False, fontsize=9)
    outputs = []
    for suffix in ("png", "pdf"):
        path = ROOT / f"docs/assets/positive_context_tradeoffs.{suffix}"
        figure.savefig(path, dpi=180, metadata={"CreationDate": None} if suffix == "pdf" else None)
        outputs.append(path)
    plt.close(figure)
    record = {"results_sha256": digest(source), "plotter_sha256": digest(Path(__file__)),
              "outputs": {path.name: digest(path) for path in outputs},
              "scope": "All sixteen policies and four controls on the separately frozen 100/100 event cohorts. Dashed constraints were used on TRAIN holdouts; test exceedances remain visible. Pointwise paired intervals are in the numerical study."}
    (ROOT / "docs/assets/positive_context_tradeoffs.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
