from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .incident_baseline import grouped_calibration_indices
from .incident_evaluation import evaluate_incident_report, summarize_incident_evaluations
from .incident_tcn_inference import persistent_positive_flags


def reconstruct_tcn_partitions(payload, report):
    split = report["split_scheme"]
    paths = np.asarray(payload["path"]).astype(str)
    official = np.asarray(payload[split + "_split"]).astype(str)
    train = np.flatnonzero(official == "train")
    decision = report["decision_threshold"]
    if decision["source"] != "train_holdout_fpr_calibration":
        raise ValueError("Event calibration requires an existing held-out training partition")
    fit, calibration = grouped_calibration_indices(paths[train], fraction=decision["calibration_fraction"], seed=decision["seed"])
    indices = {"fit": train[fit], "calibration": train[calibration], "test": np.flatnonzero(official == "test")}
    groups = {key: set(paths[index]) for key, index in indices.items()}
    if any(groups[a] & groups[b] for a, b in (("fit", "calibration"), ("fit", "test"), ("calibration", "test"))):
        raise ValueError("Source clips overlap between fitting, calibration, and test partitions")
    if any(len(indices[key]) != report[key]["sample_count"] for key in indices):
        raise ValueError("Sequence archive does not reconstruct the model training counts")
    return {key: sorted(value) for key, value in groups.items()}


def protocol_clips(protocol, clips, split, cohort, model_path: Path):
    if cohort not in {"calibration", "test"}:
        raise ValueError("Only held-out calibration and test cohorts can be replayed")
    definition = protocol["models"][split]
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != definition["model_sha256"]:
        raise ValueError("Model bytes differ from the calibration protocol")
    fit = set(definition["fit_clip_paths"])
    calibration = set(definition["calibration_clip_paths"])
    test = set(definition["test_clip_paths"])
    if fit & calibration or fit & test or calibration & test:
        raise ValueError("Protocol fitting, calibration, and test clips must be disjoint")
    names = definition[cohort + "_clip_paths"]
    if not names or len(names) != len(set(names)):
        raise ValueError("Protocol cohort must contain unique nonempty clip paths")
    indexed = {clip.relative_path.as_posix(): clip for clip in clips}
    expected_split = "train" if cohort == "calibration" else "test"
    if any(name not in indexed or getattr(indexed[name], split + "_split") != expected_split for name in names):
        raise ValueError("Protocol cohort violates the official dataset partition")
    return [indexed[name] for name in sorted(names)]


def regrid_event_report(report, *, hop_steps: int, threshold: float, gate: int):
    starts = report["window_starts"]
    if len(set(starts)) != len(starts) or len(starts) != len(report["window_probabilities"]) or hop_steps <= 0:
        raise ValueError("A unique aligned window grid and positive hop are required")
    desired = list(range(0, int(report["sampled_frames"]) - int(report["sequence_steps"]), hop_steps))
    index = {start: offset for offset, start in enumerate(starts)}
    if not desired or any(start not in index for start in desired):
        raise ValueError("Source probabilities do not cover the requested decision grid")
    selected = [index[start] for start in desired]
    probabilities = [report["window_probabilities"][offset] for offset in selected]
    result = {**report, "window_starts": desired, "window_probabilities": probabilities, "threshold": threshold,
              "hop_steps": hop_steps, "min_positive_windows": gate,
              "window_review_flags": persistent_positive_flags(probabilities, threshold, gate).tolist()}
    if "window_end_times_sec" in report:
        if len(report["window_end_times_sec"]) != len(starts):
            raise ValueError("Source decision timestamps must align with probabilities")
        result["window_end_times_sec"] = [report["window_end_times_sec"][offset] for offset in selected]
    return result


def event_operating_summary(reports_and_clips, *, threshold, hop_steps, gate):
    evaluations = [evaluate_incident_report(regrid_event_report(report, hop_steps=hop_steps, threshold=threshold, gate=gate), clip) for report, clip in reports_and_clips]
    summary = summarize_incident_evaluations(evaluations)
    summary["off_target_episodes_per_clip"] = (summary["early_alert_episodes"] + summary["late_alert_episodes"]) / summary["clip_count"]
    return summary


def select_event_threshold(curve, budget):
    if not np.isfinite(budget) or budget < 0:
        raise ValueError("Off-target episode budget must be finite and nonnegative")
    feasible = [row for row in curve if row["summary"]["off_target_episodes_per_clip"] <= budget]
    if not feasible:
        return None
    return max(feasible, key=lambda row: (row["summary"]["event_recall"], -row["summary"]["off_target_episodes_per_clip"], row["threshold"]))
