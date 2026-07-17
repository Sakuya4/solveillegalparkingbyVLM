from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .accident_dataset import AccidentClip


@dataclass(frozen=True)
class IncidentClipEvaluation:
    clip_path: str
    collision_type: str
    accident_time_sec: float
    detected: bool
    trigger_time_sec: float | None
    trigger_delay_sec: float | None
    trigger_probability: float | None
    early_alert_episodes: int
    late_alert_episodes: int
    alert_episode_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_incident_report(
    report: dict[str, Any],
    clip: AccidentClip,
    early_tolerance_sec: float = 1.0,
    late_tolerance_sec: float = 3.0,
) -> IncidentClipEvaluation:
    if early_tolerance_sec < 0 or late_tolerance_sec < 0:
        raise ValueError("Incident timing tolerances must be non-negative")
    if report.get("annotation_free_inference") is not True:
        raise ValueError("Incident evaluation requires annotation-free inference reports")

    starts, probabilities, review_flags = _validated_windows(report)
    sampled_fps = float(report.get("sampled_fps", 0.0))
    sequence_steps = int(report.get("sequence_steps", 0))
    if not math.isfinite(sampled_fps) or sampled_fps <= 0 or sequence_steps <= 0:
        raise ValueError("sampled_fps and sequence_steps must be positive")

    episode_indices = _episode_start_indices(review_flags)
    episode_times = [
        (starts[index] + sequence_steps) / sampled_fps
        for index in episode_indices
    ]
    lower_bound = clip.accident_time_sec - early_tolerance_sec
    upper_bound = clip.accident_time_sec + late_tolerance_sec
    matching = [
        (index, time_sec)
        for index, time_sec in zip(episode_indices, episode_times)
        if lower_bound <= time_sec <= upper_bound
    ]
    if matching:
        trigger_index, trigger_time_sec = matching[0]
        trigger_delay_sec = trigger_time_sec - clip.accident_time_sec
        trigger_probability = probabilities[trigger_index]
    else:
        trigger_time_sec = None
        trigger_delay_sec = None
        trigger_probability = None

    return IncidentClipEvaluation(
        clip_path=clip.relative_path.as_posix(),
        collision_type=clip.collision_type,
        accident_time_sec=clip.accident_time_sec,
        detected=bool(matching),
        trigger_time_sec=trigger_time_sec,
        trigger_delay_sec=trigger_delay_sec,
        trigger_probability=trigger_probability,
        early_alert_episodes=sum(time_sec < lower_bound for time_sec in episode_times),
        late_alert_episodes=sum(time_sec > upper_bound for time_sec in episode_times),
        alert_episode_count=len(episode_times),
    )


def summarize_incident_evaluations(
    evaluations: list[IncidentClipEvaluation],
) -> dict[str, Any]:
    if not evaluations:
        raise ValueError("At least one incident evaluation is required")
    detected = [item for item in evaluations if item.detected]
    delays = [item.trigger_delay_sec for item in detected if item.trigger_delay_sec is not None]
    groups: dict[str, list[IncidentClipEvaluation]] = defaultdict(list)
    for item in evaluations:
        groups[item.collision_type].append(item)

    return {
        "clip_count": len(evaluations),
        "detected_clips": len(detected),
        "missed_clips": len(evaluations) - len(detected),
        "event_recall": len(detected) / len(evaluations),
        "early_alert_episodes": sum(item.early_alert_episodes for item in evaluations),
        "late_alert_episodes": sum(item.late_alert_episodes for item in evaluations),
        "trigger_delay_sec": _delay_summary(delays),
        "by_collision_type": {
            name: {
                "clip_count": len(items),
                "detected_clips": sum(item.detected for item in items),
                "event_recall": sum(item.detected for item in items) / len(items),
            }
            for name, items in sorted(groups.items())
        },
    }


def match_report_to_accident_clip(
    report: dict[str, Any],
    clips: list[AccidentClip],
) -> AccidentClip:
    input_path = _portable_path(str(report.get("input", "")))
    matches = [
        clip
        for clip in clips
        if input_path.endswith(_portable_path(clip.relative_path.as_posix()))
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected one ACCIDENT metadata match for report input {input_path!r}, found {len(matches)}"
        )
    return matches[0]


def _validated_windows(
    report: dict[str, Any],
) -> tuple[list[int], list[float], list[bool]]:
    starts = report.get("window_starts", [])
    probabilities = report.get("window_probabilities", [])
    review_flags = report.get("window_review_flags", [])
    if not starts or len(starts) != len(probabilities) or len(starts) != len(review_flags):
        raise ValueError("Incident report window arrays must be non-empty and have equal lengths")
    if any(not isinstance(flag, bool) for flag in review_flags):
        raise ValueError("window_review_flags must contain booleans")

    parsed_starts = [int(value) for value in starts]
    parsed_probabilities = [float(value) for value in probabilities]
    if parsed_starts != sorted(parsed_starts) or any(value < 0 for value in parsed_starts):
        raise ValueError("window_starts must be sorted non-negative values")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in parsed_probabilities):
        raise ValueError("window_probabilities must be finite values between zero and one")
    return parsed_starts, parsed_probabilities, list(review_flags)


def _episode_start_indices(flags: list[bool]) -> list[int]:
    return [
        index
        for index, flag in enumerate(flags)
        if flag and (index == 0 or not flags[index - 1])
    ]


def _delay_summary(delays: list[float]) -> dict[str, float | None]:
    if not delays:
        return {"mean": None, "median": None, "p95": None, "minimum": None, "maximum": None}
    values = np.asarray(delays, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "minimum": float(values.min()),
        "maximum": float(values.max()),
    }


def _portable_path(value: str) -> str:
    return value.replace("\\", "/").casefold()
