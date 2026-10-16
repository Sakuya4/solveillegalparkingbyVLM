from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    study = json.loads((ROOT / "docs/results/review_service_variability.json").read_text(encoding="utf-8"))
    rows = [row for row in study["summaries"] if row["cameras"] == 4 and row["policy"] == "metered_100pct"]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.8), layout="constrained")
    times = [row["review_sec"] for row in study["service_samples"]]
    axes[0].hist(times, bins=np.linspace(6.5, 11, 10), color="#23856e", edgecolor="white")
    axes[0].axvline(study["measured_service_sec"]["mean"], color="#b94d31", linewidth=1.8, label="Measured mean")
    axes[0].set(xlabel="Measured review service (s)", ylabel="Sequences", title="Controlled Qwen service samples (n = 20)")
    axes[0].legend(frameon=False)
    x = np.arange(len(rows))
    baseline = [row["review_p95_sec"]["constant_mean"] for row in rows]
    median = np.asarray([row["review_p95_sec"]["empirical_median"] for row in rows])
    minimum = np.asarray([row["review_p95_sec"]["empirical_min"] for row in rows])
    maximum = np.asarray([row["review_p95_sec"]["empirical_max"] for row in rows])
    axes[1].bar(x - 0.18, baseline, width=0.36, color="#85918e", label="Constant mean")
    axes[1].bar(x + 0.18, median, width=0.36, color="#c57234", label="Empirical median")
    axes[1].errorbar(x + 0.18, median, yerr=[median - minimum, maximum - median], fmt="none", color="#303a37", capsize=4)
    axes[1].set_xticks(x, [f"{'IID' if row['split'] == 'iid' else 'Geo'} / gate {row['gate']}" for row in rows])
    axes[1].set(ylabel="p95 review completion latency (s)", title="Four-camera full-capacity admission")
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.savefig(ROOT / "docs/assets/review_service_variability.png", dpi=180)
    fig.savefig(ROOT / "docs/assets/review_service_variability.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
