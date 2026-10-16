from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from scripts.replay_positive_exposure import load_study, load_batch, validate_execution
from scripts.replay_presentation_events import load_event_protocol, validate_batch
from scripts.evaluate_presentation_events import model_reports
from scripts.audit_presentation_event_predictions import clock_control, verify_feature_predictions
from scripts.run_presentation_event_reporting import ScalarJSON
from illegal_parking.incident_positive_exposure import pre_event_positive_contexts, select_positive_exposure_policy
from illegal_parking.incident_presentation_events import evaluate_presentation_event, presentation_operating_summary
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def calibration_curve(protocol, reports, key):
    chosen = model_reports(protocol, reports, key, "calibration")
    thresholds = sorted(set([*protocol["threshold_grid"], protocol["models"][key]["window_threshold"]]))
    return [{"threshold": threshold, "summary": presentation_operating_summary(chosen, protocol["annotations"], threshold),
             "pre_event": pre_event_positive_contexts(chosen, protocol["annotations"], threshold)} for threshold in thresholds]


def main():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(2)
    threadpool_limits(2)
    protocol_path, protocol, selection_path, selection = load_study()
    execution_path, execution = validate_execution(protocol_path, selection_path)
    directory, batch, reports = load_batch(protocol_path, protocol, selection_path)
    if datetime.fromisoformat(batch["started_at_utc"]) < max(datetime.fromisoformat(selection["selected_at_utc"]), datetime.fromisoformat(execution["created_at_utc"])):
        raise ValueError("Test replay must follow frozen calibration and execution specification")
    original_path = ROOT / "docs/results/presentation_event_protocol.json"
    original = load_event_protocol(original_path)
    calibration_directory = ROOT / "outputs/accident/presentation_events_calibration"
    _, calibration = validate_batch(calibration_directory, original_path, original, "calibration")
    if sha256(calibration_directory / "batch.json") != protocol["calibration_batch_sha256"]:
        raise ValueError("Training holdout replay changed")
    output = ROOT / "docs/results/positive_exposure_results.json"
    trace_path = ROOT / "docs/results/positive_exposure_outcomes.csv"
    if output.exists() or output.with_suffix(".csv").exists() or trace_path.exists():
        raise FileExistsError("Published positive-context results are immutable")
    settings = protocol["constraints"]
    for key in protocol["models"]:
        curve = calibration_curve(protocol, calibration, key)
        selected = select_positive_exposure_policy(curve, episode_budget=settings["episode_budget_per_clip"],
                                                  positive_cap=settings["macro_pre_event_positive_cap"],
                                                  minimum_supported_clips=settings["minimum_supported_calibration_clips"])
        if selection["policies"][key]["curve"] != curve or selection["policies"][key]["selected"] != selected:
            raise ValueError("Training-only positive-context selections do not reproduce exactly")
    verification = verify_feature_predictions(directory, batch, reports, protocol)
    rows, traces, contrasts = [], [], []
    outcomes = {}
    for key, definition in protocol["models"].items():
        chosen = model_reports(protocol, reports, key, "test")
        policy = selection["policies"][key]
        for name, threshold in (("onset_only", policy["onset_only_threshold"]),
                                ("positive_context", policy["selected"]["threshold"] if policy["selected"] else None)):
            summary = presentation_operating_summary(chosen, protocol["annotations"], threshold) if threshold is not None else None
            prefix = pre_event_positive_contexts(chosen, protocol["annotations"], threshold) if threshold is not None else None
            rows.append({"model": key, "split": definition["split"], "policy": name, "threshold": threshold,
                         "test": summary, "pre_event": prefix,
                         "episode_budget_exceeded": summary["off_target_episodes_per_clip"] > settings["episode_budget_per_clip"] if summary else None,
                         "pre_event_cap_exceeded": prefix["macro_positive_fraction"] > settings["macro_pre_event_positive_cap"] if prefix and prefix["macro_positive_fraction"] is not None else None})
            if threshold is not None:
                values = [evaluate_incident(report, protocol, threshold) for report in chosen]
                outcomes[(key, name)] = np.asarray([value.detected for value in values], dtype=float)
                traces.extend(trace_row(report, value, key, name, definition["split"], threshold) for report, value in zip(chosen, values))
        a, b = outcomes.get((key, "positive_context")), outcomes.get((key, "onset_only"))
        contrasts.append({"model": key, "contrast": "positive-context minus onset-only recall",
                          "paired_interval": paired_cluster_coverage_interval(np.ones(len(a)), a, b) if a is not None and b is not None else None})
    controls = {}
    for split in protocol["cohorts"]:
        key = next(key for key, definition in protocol["models"].items() if definition["split"] == split)
        chosen = model_reports(protocol, reports, key, "test")
        controls[split] = {}
        for positive, name in ((True, "clock_only_positive"), (False, "always_negative")):
            nonvisual = [clock_control(report, positive) for report in chosen]
            controls[split][name] = {"test": presentation_operating_summary(nonvisual, protocol["annotations"], .5),
                                     "pre_event": pre_event_positive_contexts(nonvisual, protocol["annotations"], .5)}
            traces.extend(trace_row(report, evaluate_incident(report, protocol, .5), name, "control", split, .5) for report in nonvisual)
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    result = {"protocol_sha256": sha256(protocol_path), "selection_sha256": sha256(selection_path), "execution_sha256": sha256(execution_path),
              "test_batch_sha256": sha256(directory / "batch.json"), "trace_sha256": sha256(trace_path), "trace_rows": len(traces),
              "prediction_verification": verification, "policies": rows, "paired_recall_contrasts": contrasts, "controls": controls,
              "constraints": settings, "scope": protocol["scope"], "default_policy": protocol["default_policy"]}
    output.write_text(ScalarJSON.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["model", "policy", "threshold", "event_recall", "off_target_per_clip", "pre_event_macro_positive_fraction", "pre_event_support_clips", "median_delay_sec", "episode_budget_exceeded", "pre_event_cap_exceeded"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            summary, prefix = row["test"], row["pre_event"]
            writer.writerow({"model": row["model"], "policy": row["policy"], "threshold": row["threshold"],
                             "event_recall": summary["event_recall"] if summary else None,
                             "off_target_per_clip": summary["off_target_episodes_per_clip"] if summary else None,
                             "pre_event_macro_positive_fraction": prefix["macro_positive_fraction"] if prefix else None,
                             "pre_event_support_clips": prefix["clips_with_support"] if prefix else None,
                             "median_delay_sec": summary["trigger_delay_sec"]["median"] if summary else None,
                             "episode_budget_exceeded": row["episode_budget_exceeded"], "pre_event_cap_exceeded": row["pre_event_cap_exceeded"]})
    print(ScalarJSON.dumps({"prediction_verification": verification, "paired_recall_contrasts": contrasts}, indent=2))


def evaluate_incident(report, protocol, threshold):
    return evaluate_presentation_event(report, protocol["annotations"][report["path"]], threshold)


def trace_row(report, value, model, policy, split, threshold):
    return {"model": model, "policy": policy, "split": split, "path": report["path"], "source_sha256": report["source_sha256"],
            "threshold": threshold, "annotation_pts_sec": value.accident_time_sec, "detected": int(value.detected),
            "trigger_pts_sec": value.trigger_time_sec, "signed_delay_sec": value.trigger_delay_sec,
            "early_episodes": int(value.early_alert_episodes), "late_episodes": int(value.late_alert_episodes),
            "valid_contexts": len(report["window_starts"]), "rejected_contexts": len(report["rejected_window_starts"])}


if __name__ == "__main__":
    main()
