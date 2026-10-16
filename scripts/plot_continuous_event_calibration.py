from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout="constrained", sharey=True)
    colors = {0.25: "#437798", 0.5: "#23856e", 1.0: "#bd643a"}
    for column, split in enumerate(("iid", "geographic")):
        data = json.loads((ROOT / f"docs/results/continuous_event_calibration_{split}.json").read_text(encoding="utf-8"))
        for row, hop in enumerate((5, 15)):
            axis = axes[row, column]
            for budget, color in colors.items():
                for gate, marker in ((1, "o"), (2, "s")):
                    policy = next(p for p in data["policies"] if p["hop_steps"] == hop and p["gate"] == gate and p["budget_per_clip"] == budget)
                    if policy["test"] is not None:
                        point = policy["test"]
                        axis.scatter(point["off_target_episodes_per_clip"], point["event_recall"], color=color, marker=marker, s=58,
                                     label=f"Budget {budget:g} / gate {gate}")
            for baseline in data["baselines"]:
                if baseline["hop_steps"] == hop:
                    point = baseline["test"]
                    axis.scatter(point["off_target_episodes_per_clip"], point["event_recall"], marker="x", color="#343c3b", s=65)
                    axis.annotate(f"W{baseline['gate']}", (point["off_target_episodes_per_clip"], point["event_recall"]), xytext=(5, 4), textcoords="offset points", fontsize=9)
            axis.set(title=f"{'IID' if split == 'iid' else 'Geographic'} / hop {hop}", xlabel="Test off-target episodes per accident clip", ylim=(-0.04, 1.03), xlim=(-0.08, 3.2))
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(alpha=0.2)
            if column == 0:
                axis.set_ylabel("Test event recall")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(ROOT / "docs/assets/continuous_event_calibration.png", dpi=180)
    fig.savefig(ROOT / "docs/assets/continuous_event_calibration.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
