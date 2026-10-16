from __future__ import annotations

import math

import numpy as np

from .incident_evaluation import IncidentClipEvaluation, summarize_incident_evaluations
from .incident_presentation import normalized_presentation_times, presentation_grid, valid_fixed_window


def continuous_presentation_windows(timestamps):
    times = normalized_presentation_times(timestamps)
    indices, targets = presentation_grid(times)
    windows, rejected = [], []
    for start in range(0, len(indices) - 15, 15):
        selected, requested = indices[start:start + 16], targets[start:start + 16]
        if not valid_fixed_window(selected, requested, times):
            rejected.append(start)
            continue
        windows.append({"start": start, "indices": selected.tolist(),
                        "times_sec": times[selected].tolist(), "end_time_sec": float(times[selected[-1]])})
    return windows, rejected


def presentation_episode_indices(starts, probabilities, threshold, gate=2):
    starts = np.asarray(starts)
    scores = np.asarray(probabilities, dtype=float)
    if starts.ndim != 1 or scores.ndim != 1 or len(starts) != len(scores):
        raise ValueError("Window identities and probabilities must be aligned vectors")
    if not np.issubdtype(starts.dtype, np.integer) and len(starts):
        raise ValueError("Window starts must be integer grid slots")
    if np.any(starts < 0) or np.any(starts % 15) or np.any(np.diff(starts) <= 0):
        raise ValueError("Unique increasing starts on the frozen hop-15 grid are required")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1 or not isinstance(gate, int) or gate <= 0:
        raise ValueError("Threshold and persistence gate must be valid")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0) | (scores > 1)):
        raise ValueError("Probabilities must be finite and bounded")
    run = 0
    episodes = []
    for offset, (start, score) in enumerate(zip(starts, scores)):
        if offset and start - starts[offset - 1] != 15:
            run = 0
        run = run + 1 if score >= threshold else 0
        if run == gate:
            episodes.append(offset)
    return episodes


def evaluate_presentation_event(report, annotation, threshold, gate=2):
    if report.get("annotation_free_inference") is not True or report.get("clock") != "normalized_presentation_time":
        raise ValueError("Annotation-free presentation-clock predictions are required")
    starts, scores = report["window_starts"], report["window_probabilities"]
    episodes = presentation_episode_indices(starts, scores, threshold, gate)
    times = np.asarray(report["window_end_times_sec"], dtype=float)
    if times.ndim != 1 or len(times) != len(starts) or not np.all(np.isfinite(times)) or np.any(times < 0) or np.any(np.diff(times) <= 0):
        raise ValueError("Decision presentation times must be finite, increasing and aligned")
    event = float(annotation["event_time_sec"])
    if not math.isfinite(event) or event < 0 or report["path"] != annotation["path"]:
        raise ValueError("Event anchor must be finite and matched to the source clip")
    matching = [offset for offset in episodes if event - 1 <= times[offset] <= event + 3]
    first = matching[0] if matching else None
    trigger = float(times[first]) if first is not None else None
    return IncidentClipEvaluation(
        clip_path=report["path"], collision_type=annotation["collision_type"], accident_time_sec=event,
        detected=first is not None, trigger_time_sec=trigger,
        trigger_delay_sec=trigger - event if trigger is not None else None,
        trigger_probability=float(scores[first]) if first is not None else None,
        early_alert_episodes=sum(times[offset] < event - 1 for offset in episodes),
        late_alert_episodes=sum(times[offset] > event + 3 for offset in episodes), alert_episode_count=len(episodes))


def presentation_operating_summary(reports, annotations, threshold):
    if len(reports) != len({report["path"] for report in reports}):
        raise ValueError("Event summaries require unique source clips")
    evaluations = [evaluate_presentation_event(report, annotations[report["path"]], threshold) for report in reports]
    summary = summarize_incident_evaluations(evaluations)
    summary["off_target_episodes_per_clip"] = (summary["early_alert_episodes"] + summary["late_alert_episodes"]) / len(reports)
    summary["rejected_windows"] = sum(len(report["rejected_window_starts"]) for report in reports)
    summary["clips_without_valid_windows"] = sum(not report["window_starts"] for report in reports)
    return summary
