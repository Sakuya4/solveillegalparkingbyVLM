from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.research_statistics import wilson_interval


def main():
    parser = argparse.ArgumentParser(description="Publish portable summaries of a completed geographic event batch.")
    parser.add_argument("--batch-dir", type=Path, default=ROOT / "outputs/accident/event_batch_geographic_100_20261016")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/geographic_event_evaluation.json")
    args = parser.parse_args()
    batch = json.loads((args.batch_dir / "batch_summary.json").read_text(encoding="utf-8"))
    if batch["failure_count"] or batch["completed_clips"] + batch["resumed_clips"] != batch["requested_clips"]:
        raise ValueError("Publish only a complete, failure-free batch")
    model = Path(batch["model"])
    result = {
        "split_scheme": batch["split_scheme"], "sampling_seed": batch["sampling_seed"],
        "clips": batch["requested_clips"], "failure_count": batch["failure_count"],
        "elapsed_sec": batch["elapsed_sec"], "threads": batch["threads"],
        "model_filename": model.name, "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
        "profiles": [], "evaluations": {},
    }
    for name in ("responsive", "conservative"):
        source = json.loads((args.batch_dir / f"{name}_evaluation.json").read_text(encoding="utf-8"))
        evaluation = source["incident_evaluation"]
        low, high = wilson_interval(evaluation["detected_clips"], evaluation["clip_count"])
        row = {
            "profile": name, "min_positive_windows": source["methodology"]["min_positive_windows"],
            "clips": evaluation["clip_count"], "detected": evaluation["detected_clips"],
            "event_recall": evaluation["event_recall"], "recall_lower_95": low, "recall_upper_95": high,
            "early_alert_episodes": evaluation["early_alert_episodes"], "late_alert_episodes": evaluation["late_alert_episodes"],
            "delay_mean_sec": evaluation["trigger_delay_sec"]["mean"],
            "delay_median_sec": evaluation["trigger_delay_sec"]["median"],
            "delay_p95_sec": evaluation["trigger_delay_sec"]["p95"],
        }
        result["profiles"].append(row)
        result["evaluations"][name] = {"methodology": source["methodology"], "summary": evaluation, "clips": source["clips"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["profiles"], indent=2))


if __name__ == "__main__":
    main()
