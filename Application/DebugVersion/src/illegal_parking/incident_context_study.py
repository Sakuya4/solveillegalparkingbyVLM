from __future__ import annotations

import numpy as np

from .incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
from .incident_positive_exposure import pre_event_positive_contexts
from .incident_presentation_events import presentation_operating_summary


def fit_indices(rows, protocol, arm, split):
    allowed = set(protocol["memberships"][split]["fit_clip_paths"])
    selected = []
    for index, row in enumerate(rows):
        if row["arm"] != arm or split not in row["fit_splits"]:
            continue
        if row["path"] not in allowed or row["target"] not in (0, 1):
            raise ValueError("Fitting row crosses its corresponding source partition")
        selected.append(index)
    if not selected or {rows[index]["target"] for index in selected} != {0, 1}:
        raise ValueError("Both fitting classes are required")
    return np.asarray(selected, dtype=int)


def validate_features(matrix, count):
    array = np.asarray(matrix)
    if array.shape != (count, 15, len(CANDIDATE_MOTION_FEATURE_NAMES)) or array.dtype != np.float32 or not np.all(np.isfinite(array)):
        raise ValueError("Feature tensors violate the finite float32 context contract")
    return array


def model_reports(protocol, reports, key, cohort):
    split = key.rsplit("_", 1)[-1]
    names = protocol["memberships"][split][cohort + "_clip_paths"]
    indexed = {report["path"]: report for report in reports}
    if len(indexed) != len(reports) or any(name not in indexed for name in names):
        raise ValueError("Every frozen source must have exactly one replay record")
    return [{**indexed[name], "window_probabilities": indexed[name]["scores"][key]} for name in names]


def operating_point(reports, annotations, threshold):
    return {"threshold": threshold,
            "summary": presentation_operating_summary(reports, annotations, threshold),
            "pre_event": pre_event_positive_contexts(reports, annotations, threshold)}


def select_policy(curve, settings):
    eligible = [row for row in curve if row["summary"]["off_target_episodes_per_clip"] <= settings["episode_budget_per_clip"]
                and row["pre_event"]["clips_with_support"] >= settings["minimum_supported_clips"]
                and row["pre_event"]["macro_positive_fraction"] is not None
                and row["pre_event"]["macro_positive_fraction"] <= settings["macro_pre_event_positive_cap"]]
    return max(eligible, key=lambda row: (row["summary"]["event_recall"],
                                        -row["summary"]["off_target_episodes_per_clip"],
                                        -row["pre_event"]["macro_positive_fraction"], row["threshold"])) if eligible else None
