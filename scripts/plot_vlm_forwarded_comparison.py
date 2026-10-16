import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    data = json.loads((ROOT / "docs/results/vlm_forwarded_comparison.json").read_text(encoding="utf-8"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    keys = ("qwen3", "llava3", "qwen8")
    x = np.arange(3)
    for offset, version, color, label in ((-0.18, "legacy", "#89918f", "Preserved requested-greedy"), (0.18, "v2", "#24856e", "Explicit greedy / v2")):
        axes[0].bar(x + offset, [data["studies"][f"{key}_{version}"]["end_to_end_correct_rate"] for key in keys], width=0.36, color=color, label=label)
    axes[0].set_xticks(x, ["Qwen / 3", "LLaVA / 3", "Qwen / 8"])
    axes[0].set(ylabel="Correct structured outputs / all cases", ylim=(0, 1), title="20 paired sequences / 10 source clips")
    axes[0].legend(frameon=False, fontsize=9)
    comparisons = data["paired_all_case_correctness"]
    labels = ["Qwen minus LLaVA / 3 frames", "Qwen 8 minus 3 frames", "Qwen 3: v2 minus preserved"]
    for index, key in enumerate(("qwen3_v2_minus_llava3_v2", "qwen8_v2_minus_qwen3_v2", "qwen3_v2_minus_qwen3_legacy")):
        row = comparisons[key]
        center = row["difference"]
        axes[1].errorbar(center, index, xerr=[[center - row["lower_95"]], [row["upper_95"] - center]], fmt="o", capsize=4, color="#24856e" if index < 2 else "#a56738")
    axes[1].axvline(0, color="#89918f", linestyle="--", linewidth=1)
    axes[1].set_yticks(range(3), labels)
    axes[1].invert_yaxis()
    axes[1].set(xlabel="Paired correctness difference (95% interval)", title="Source-clip bootstrap / 2,000 resamples")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(alpha=0.2)
        axis.set_axisbelow(True)
    for suffix in ("png", "pdf"):
        fig.savefig(ROOT / f"docs/assets/vlm_forwarded_comparison.{suffix}", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
