from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.incident_event_calibration import event_operating_summary, protocol_clips, regrid_event_report, select_event_threshold
from illegal_parking.incident_evaluation import evaluate_incident_report, match_report_to_accident_clip
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_reports(directory, protocol, protocol_hash, split, cohort):
    summary_path = directory / "batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    definition = protocol["models"][split]
    if summary["failure_count"] or summary["protocol_sha256"] != protocol_hash or summary["cohort"] != cohort or summary["model_sha256"] != definition["model_sha256"]:
        raise ValueError("Batch provenance or completion does not match the frozen calibration protocol")
    if summary["requested_clips"] != summary["completed_clips"] + summary["resumed_clips"]:
        raise ValueError("Event batch is incomplete")
    metadata = ROOT / "data/raw/accident/full/extracted/metadata-real.csv"
    if digest(metadata) != protocol["metadata_sha256"]:
        raise ValueError("Metadata no longer matches the frozen protocol")
    clips = load_accident_manifest(metadata)
    expected = {clip.relative_path.as_posix() for clip in protocol_clips(protocol, clips, split, cohort, ROOT / definition["checkpoint"])}
    reports, sources, seen = [], [], set()
    for name in summary["report_paths"]:
        path = directory / "reports" / Path(name.replace("\\", "/")).name
        report = json.loads(path.read_text(encoding="utf-8"))
        clip = match_report_to_accident_clip(report, clips)
        key = clip.relative_path.as_posix()
        if key in seen or "window_end_times_sec" not in report or report.get("batch_inference_signature") != summary["batch_inference_signature"]:
            raise ValueError("Reports must be unique, source-clock aligned, and bound to this batch")
        seen.add(key)
        reports.append((report, clip))
        sources.append({"clip": key, "sha256": digest(path)})
    if seen != expected:
        raise ValueError("Batch clips differ from the frozen protocol cohort")
    return reports, {"batch_sha256": digest(summary_path), "reports": sources}, summary


def main():
    parser = argparse.ArgumentParser(description="Select continuous-event operating points on held-out training clips, then evaluate frozen test outcomes.")
    parser.add_argument("stage", choices=("calibrate", "evaluate"))
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), required=True)
    parser.add_argument("--protocol", type=Path, default=ROOT / "docs/results/continuous_event_calibration_protocol.json")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_hash = digest(args.protocol)
    split = args.split_scheme
    calibration_directory = ROOT / f"outputs/accident/continuous_calibration_{split}"
    selection_path = ROOT / f"outputs/accident/continuous_event_{split}_selection.json"
    calibration, calibration_sources, _ = load_reports(calibration_directory, protocol, protocol_hash, split, "calibration")
    original_threshold = protocol["models"][split]["original_window_threshold"]
    if args.stage == "calibrate":
        if selection_path.exists():
            old = json.loads(selection_path.read_text(encoding="utf-8"))
            if old["protocol_sha256"] != protocol_hash or old["calibration_sources"] != calibration_sources:
                raise ValueError("Frozen selection inputs changed; use a separately versioned study")
            print("Existing frozen training-only selection verified", flush=True)
            return
        curves, policies = [], []
        thresholds = sorted(set([*protocol["threshold_grid"], original_threshold]))
        for hop in protocol["evaluated_hop_steps"]:
            for gate in protocol["persistence_gates"]:
                curve = [{"threshold": threshold, "summary": event_operating_summary(calibration, threshold=threshold, hop_steps=hop, gate=gate)} for threshold in thresholds]
                curves.extend({"hop_steps": hop, "gate": gate, **row} for row in curve)
                for budget in protocol["off_target_budgets_per_clip"]:
                    selected = select_event_threshold(curve, budget)
                    policies.append({"hop_steps": hop, "gate": gate, "budget_per_clip": budget, "selected": selected})
                print(f"Calibrated {split}: hop={hop}, gate={gate}, {len(curve)} thresholds", flush=True)
        selection = {"split_scheme": split, "selected_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": protocol_hash,
                     "calibration_clips": len(calibration), "calibration_sources": calibration_sources, "policies": policies, "curves": curves,
                     "selection_inputs": "Held-out official training clips only; no test reports are read in calibration stage"}
        selection_path.parent.mkdir(parents=True, exist_ok=True)
        selection_path.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        return
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection["protocol_sha256"] != protocol_hash or selection["calibration_sources"] != calibration_sources:
        raise ValueError("Frozen selection provenance mismatch")
    test, test_sources, test_summary = load_reports(ROOT / f"outputs/accident/continuous_test_{split}", protocol, protocol_hash, split, "test")
    if test_summary["threshold_selection_sha256"] != digest(selection_path) or datetime.fromisoformat(test_summary["started_at_utc"]) < datetime.fromisoformat(selection["selected_at_utc"]):
        raise ValueError("Test inference must follow the frozen train-only threshold selection")
    rows, baselines = [], []
    for hop in protocol["evaluated_hop_steps"]:
        for gate in protocol["persistence_gates"]:
            baselines.append({"hop_steps": hop, "gate": gate, "threshold": original_threshold, "test": event_operating_summary(test, threshold=original_threshold, hop_steps=hop, gate=gate)})
    for policy in selection["policies"]:
        row = {**policy, "test": None}
        if policy["selected"] is not None:
            threshold = policy["selected"]["threshold"]
            hop, gate = policy["hop_steps"], policy["gate"]
            row["test"] = event_operating_summary(test, threshold=threshold, hop_steps=hop, gate=gate)
            outcomes = [[evaluate_incident_report(regrid_event_report(report, hop_steps=hop, threshold=value, gate=gate), clip).detected for report, clip in test] for value in (threshold, original_threshold)]
            row["paired_recall_vs_window_threshold"] = paired_cluster_coverage_interval(np.ones(len(test)), np.asarray(outcomes[0], dtype=float), np.asarray(outcomes[1], dtype=float))
        rows.append(row)
    result = {"split_scheme": split, "training_cohort": "full eligible ACCIDENT cohort; candidate ROI TCN", "model_sha256": protocol["models"][split]["model_sha256"],
              "protocol_sha256": protocol_hash, "selection_sha256": digest(selection_path), "calibration_clips": len(calibration), "test_clips": len(test),
              "calibration_sources": calibration_sources, "test_sources": test_sources, "baselines": baselines, "policies": rows,
              "scope": protocol["scope"], "uncertainty": "Paired test-clip bootstrap, fixed models and selected thresholds; calibration-selection uncertainty is not resampled",
              "reference_policy": "All predeclared budgets, gates, and hops are reported; 500-clip desktop binaries and defaults remain unchanged"}
    output = ROOT / f"docs/results/continuous_event_calibration_{split}.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    with output.with_suffix(".csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "hop", "gate", "budget_per_clip", "threshold", "test_recall", "test_off_target_per_clip", "test_delay_median_sec"])
        writer.writeheader()
        for row in rows:
            data = row["test"]
            writer.writerow({"split": split, "hop": row["hop_steps"], "gate": row["gate"], "budget_per_clip": row["budget_per_clip"], "threshold": row["selected"]["threshold"] if data else None, "test_recall": data["event_recall"] if data else None, "test_off_target_per_clip": data["off_target_episodes_per_clip"] if data else None, "test_delay_median_sec": data["trigger_delay_sec"]["median"] if data else None})
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
