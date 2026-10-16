from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from scripts.replay_presentation_events import load_event_protocol, validate_batch
from illegal_parking.incident_event_calibration import select_event_threshold
from illegal_parking.incident_presentation_events import evaluate_presentation_event, presentation_operating_summary
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def model_reports(protocol, reports, key, cohort):
    definition = protocol["models"][key]
    names = protocol["cohorts"][definition["split"]][cohort + "_clip_paths"]
    indexed = {report["path"]: report for report in reports}
    return [{**indexed[name], "window_probabilities": indexed[name]["scores"][key]} for name in names]


def select_models(protocol, reports):
    policies = {}
    for key, definition in protocol["models"].items():
        selected_reports = model_reports(protocol, reports, key, "calibration")
        curve = [{"threshold": threshold, "summary": presentation_operating_summary(selected_reports, protocol["annotations"], threshold)}
                 for threshold in sorted(set([*protocol["calibration"]["threshold_grid"], definition["window_threshold"]]))]
        policies[key] = {"selected": select_event_threshold(curve, protocol["calibration"]["episode_budget_per_clip"]),
                         "calibration_clips": len(selected_reports), "curve": curve}
    return policies


def main():
    parser = argparse.ArgumentParser(description="Calibrate PTS event thresholds before test replay; publish all frozen-model outcomes.")
    parser.add_argument("stage", choices=("calibrate", "evaluate"))
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/results/presentation_event_protocol.json")
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    protocol = load_event_protocol(args.protocol)
    calibration_directory = ROOT / "outputs/accident/presentation_events_calibration"
    calibration_batch, calibration_reports = validate_batch(calibration_directory, args.protocol, protocol, "calibration")
    calibration_hash = sha256(calibration_directory / "batch.json")
    selection_path = ROOT / "outputs/accident/presentation_event_selection.json"
    if args.stage == "calibrate":
        if selection_path.exists():
            old = json.loads(selection_path.read_text(encoding="utf-8"))
            if old["protocol_sha256"] != sha256(args.protocol) or old["calibration_batch_sha256"] != calibration_hash:
                raise ValueError("Existing threshold-selection provenance changed")
            print("Frozen event threshold selection verified", flush=True)
            return
        selection = {"protocol_sha256": sha256(args.protocol), "calibration_batch_sha256": calibration_hash,
                     "selected_at_utc": datetime.now(timezone.utc).isoformat(), "policies": select_models(protocol, calibration_reports),
                     "input_scope": "Corresponding official-training holdout only; no test reports read"}
        selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({key: row["selected"] for key, row in selection["policies"].items()}, indent=2))
        return
    output = ROOT / "docs/results/presentation_event_comparison.json"
    if output.exists() or output.with_suffix(".csv").exists():
        raise FileExistsError("Published event comparisons are immutable")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["protocol_sha256"] != sha256(args.protocol) or selection["calibration_batch_sha256"] != calibration_hash:
        raise ValueError("Threshold selection no longer binds its calibration inputs")
    if selection["policies"] != select_models(protocol, calibration_reports):
        raise ValueError("Frozen policies do not reproduce from the held-out training reports")
    test_directory = ROOT / "outputs/accident/presentation_events_test"
    test_batch, test_reports = validate_batch(test_directory, args.protocol, protocol, "test", sha256(selection_path))
    if datetime.fromisoformat(test_batch["started_at_utc"]) < datetime.fromisoformat(selection["selected_at_utc"]):
        raise ValueError("Test replay must follow training-only threshold selection")
    if calibration_batch["versions"] != test_batch["versions"] or calibration_batch["device"] != test_batch["device"]:
        raise ValueError("Calibration/test feature runtimes must match")
    outcomes, models = {}, []
    for key, definition in protocol["models"].items():
        reports = model_reports(protocol, test_reports, key, "test")
        policy = selection["policies"][key]
        selected = policy["selected"]
        threshold = selected["threshold"] if selected else None
        summary = presentation_operating_summary(reports, protocol["annotations"], threshold) if selected else None
        if selected:
            outcomes[key] = np.asarray([evaluate_presentation_event(report, protocol["annotations"][report["path"]], threshold).detected for report in reports], dtype=float)
        models.append({"key": key, **definition, "calibration_clips": policy["calibration_clips"], "test_clips": len(reports),
                       "selected_threshold": threshold, "calibration": selected["summary"] if selected else None,
                       "test": summary, "test_budget_exceeded": summary["off_target_episodes_per_clip"] > .5 if summary else None})
    contrasts = []
    for feature in ("candidate", "global"):
        for split in ("iid", "geographic"):
            a, b = f"fixed_{feature}_{split}", f"legacy_{feature}_{split}"
            contrast = paired_cluster_coverage_interval(np.ones(len(outcomes[a])), outcomes[a], outcomes[b]) if a in outcomes and b in outcomes else None
            contrasts.append({"feature": feature, "split": split, "contrast": "fixed minus legacy event recall", "paired_interval": contrast})
    result = {"protocol_sha256": sha256(args.protocol), "selection_sha256": sha256(selection_path),
              "calibration_batch_sha256": calibration_hash, "test_batch_sha256": sha256(test_directory / "batch.json"),
              "models": models, "paired_recall_contrasts": contrasts,
              "scope": protocol["scope"], "clock": protocol["inference"]["clock"],
              "uncertainty": "Pointwise paired source-clip bootstrap, fixed models/selected thresholds; no refitting or family-wise coverage",
              "default_policy": protocol["default_policy"]}
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "test_clips", "threshold", "recall", "off_target_per_clip", "delay_median_sec", "budget_exceeded", "rejected_windows"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in models:
            summary = row["test"]
            writer.writerow({"model": row["key"], "test_clips": row["test_clips"], "threshold": row["selected_threshold"],
                             "recall": summary["event_recall"] if summary else None,
                             "off_target_per_clip": summary["off_target_episodes_per_clip"] if summary else None,
                             "delay_median_sec": summary["trigger_delay_sec"]["median"] if summary else None,
                             "budget_exceeded": row["test_budget_exceeded"], "rejected_windows": summary["rejected_windows"] if summary else None})
    print(json.dumps(contrasts, indent=2))


if __name__ == "__main__":
    main()
