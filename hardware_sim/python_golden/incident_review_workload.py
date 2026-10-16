from __future__ import annotations

import math

import numpy as np

from illegal_parking.incident_tcn_inference import persistent_positive_flags


def build_incident_review_workload(reports_and_clips, min_positive_windows: int):
    """Build causal episode arrivals; attach annotation matches only for scoring."""
    flags, matching = [], {}
    seen = set()
    for report, clip in reports_and_clips:
        key = clip.relative_path.as_posix()
        if key in seen:
            raise ValueError("Duplicate clip in review workload")
        seen.add(key)
        if report.get("annotation_free_inference") is not True:
            raise ValueError("Workload requires annotation-free inference")
        count, fps = int(report["sampled_frames"]), float(report["sampled_fps"])
        if count <= 0 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("Positive sampled frame count and FPS are required")
        starts = report["window_starts"]
        reviews = persistent_positive_flags(report["window_probabilities"], report["threshold"], min_positive_windows)
        if len(starts) != len(reviews):
            raise ValueError("Window starts and probabilities must have equal lengths")
        current = np.zeros(count, dtype=bool)
        offset = len(flags)
        previous_decision = -1
        for index, active in enumerate(reviews):
            decision = int(starts[index]) + int(report["sequence_steps"])
            if not 0 <= decision < count or decision <= previous_decision:
                raise ValueError("Window decisions must be increasing and inside the clip")
            previous_decision = decision
            if not active or (index and reviews[index - 1]):
                continue
            current[decision] = True
            if clip.accident_time_sec - 1 <= decision / fps <= clip.accident_time_sec + 3:
                matching[offset + decision] = key
        flags.extend(current.tolist())
    if not flags:
        raise ValueError("At least one clip is required")
    return np.asarray(flags, dtype=bool), matching, len(seen)


def summarize_review_retention(raw, admitted, matching, review_events, *, clips: int, fps: float):
    raw, admitted = np.asarray(raw, dtype=bool), np.asarray(admitted, dtype=bool)
    if raw.ndim != 2 or raw.shape != admitted.shape or np.any(admitted & ~raw):
        raise ValueError("Admission must be a subset of the time-by-camera workload")
    cameras = raw.shape[1]

    def identities(indices):
        return {(matching[index // cameras], index % cameras) for index in indices if index // cameras in matching}

    detected = identities(np.flatnonzero(raw.reshape(-1)))
    after_admission = identities(np.flatnonzero(admitted.reshape(-1)))
    completed = identities([event["source_index"] for event in review_events if event["status"] == "completed"])
    queued = identities([event["source_index"] for event in review_events if event["status"] != "dropped"])
    latency = [(event["finish_ms"] - (event["source_index"] // cameras) * 1000 / fps) / 1000 for event in review_events if event["status"] == "completed"]
    return {
        "clip_camera_events": clips * cameras, "detected_before_routing": len(detected),
        "retained_after_admission": len(after_admission), "retained_after_queue_acceptance": len(queued),
        "retained_after_completed_service": len(completed),
        "admission_retention": len(after_admission) / len(detected) if detected else None,
        "completed_service_retention": len(completed) / len(detected) if detected else None,
        "completed_service_event_coverage": len(completed) / (clips * cameras),
        "completed_review_latency_sec": {"median": float(np.median(latency)) if latency else None, "p95": float(np.percentile(latency, 95)) if latency else None},
    }
