from __future__ import annotations

import math

import numpy as np

from .incident_event_calibration import select_event_threshold


def pre_event_positive_contexts(reports, annotations, threshold):
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A finite probability threshold is required")
    fractions = []
    contexts, positives = 0, 0
    for report in reports:
        times = np.asarray(report["window_end_times_sec"], dtype=float)
        scores = np.asarray(report["window_probabilities"], dtype=float)
        event = float(annotations[report["path"]]["event_time_sec"])
        if (times.ndim != 1 or scores.shape != times.shape or not np.all(np.isfinite(times))
                or np.any(times < 0) or np.any(np.diff(times) <= 0) or not np.all(np.isfinite(scores))
                or np.any((scores < 0) | (scores > 1)) or not math.isfinite(event) or event < 0):
            raise ValueError("Pre-event exposure requires aligned finite scores and source times")
        eligible = times < event - 1
        count = int(eligible.sum())
        if count:
            positive = int(np.count_nonzero(scores[eligible] >= threshold))
            contexts += count
            positives += positive
            fractions.append(positive / count)
    return {"clips_with_support": len(fractions), "clips_without_support": len(reports) - len(fractions),
            "pre_event_contexts": contexts, "positive_pre_event_contexts": positives,
            "macro_positive_fraction": float(np.mean(fractions)) if fractions else None,
            "pooled_positive_fraction": positives / contexts if contexts else None}


def select_positive_exposure_policy(curve, *, episode_budget=.5, positive_cap=.2, minimum_supported_clips=20):
    if not math.isfinite(positive_cap) or not 0 <= positive_cap <= 1 or minimum_supported_clips <= 0:
        raise ValueError("Positive-context cap and support requirement must be valid")
    feasible = [row for row in curve if row["pre_event"]["clips_with_support"] >= minimum_supported_clips
                and row["pre_event"]["macro_positive_fraction"] is not None
                and row["pre_event"]["macro_positive_fraction"] <= positive_cap]
    return select_event_threshold(feasible, episode_budget)
