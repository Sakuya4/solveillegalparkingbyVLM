from __future__ import annotations

from dataclasses import replace
import math

import numpy as np

from .accident_dataset import build_temporal_windows


TARGET_FPS = 7.5
SAMPLE_COUNT = 16
TIMESTAMP_TOLERANCE_SEC = 1e-4


def normalized_presentation_times(timestamps):
    times = np.asarray(timestamps, dtype=np.float64)
    if times.ndim != 1 or not len(times) or not np.all(np.isfinite(times)) or np.any(np.diff(times) <= 0):
        raise ValueError("Presentation timestamps must be finite, nonempty and strictly increasing")
    return times - times[0]


def presentation_grid(timestamps, target_fps=TARGET_FPS):
    times = normalized_presentation_times(timestamps)
    if not math.isfinite(target_fps) or target_fps <= 0:
        raise ValueError("Target cadence must be finite and positive")
    targets = np.arange(math.floor((times[-1] + 1e-6) * target_fps) + 1) / target_fps
    indices = np.searchsorted(times, targets - 1e-6)
    return indices, targets


def valid_fixed_window(indices, targets, times):
    if len(indices) != SAMPLE_COUNT or len(targets) != SAMPLE_COUNT or np.any(np.diff(indices) <= 0):
        return False
    if indices[0] < 0 or indices[-1] >= len(times):
        return False
    lateness = times[indices] - targets
    return bool(np.all(lateness >= -1e-6) and np.all(lateness <= 1 / TARGET_FPS + 1e-6))


def paired_presentation_windows(clip, timestamps):
    times = normalized_presentation_times(timestamps)
    event_frame = clip.accident_frame
    if not 0 <= event_frame < len(times):
        return [], "supplied_annotation_outside_decoded_support"
    actual_clip = replace(clip, frame_count=len(times))
    legacy = build_temporal_windows(actual_clip, window_frames=32, negative_gap_frames=8)
    grid, targets = presentation_grid(times)
    if not legacy or len(grid) < SAMPLE_COUNT:
        return [], "insufficient_paired_context"
    incident = next(window for window in legacy if window.label == "incident")
    if incident.end_frame - 2 < event_frame:
        return [], "legacy_event_after_last_sampled_frame"
    event_time = float(times[event_frame])
    first = max(0, min(int(np.searchsorted(targets, event_time)) - 8, len(grid) - SAMPLE_COUNT))
    fixed = {"incident": (grid[first:first + SAMPLE_COUNT], targets[first:first + SAMPLE_COUNT])}
    normal_end = first - 4
    if normal_end >= SAMPLE_COUNT:
        fixed["normal"] = (grid[normal_end - SAMPLE_COUNT:normal_end], targets[normal_end - SAMPLE_COUNT:normal_end])
    positive_indices, positive_targets = fixed["incident"]
    if not valid_fixed_window(positive_indices, positive_targets, times):
        return [], "fixed_event_cadence_or_lateness_violation"
    if not times[positive_indices[0]] <= event_time <= times[positive_indices[-1]]:
        return [], "fixed_event_outside_sampled_context"
    rows = []
    for window in legacy:
        if window.label not in fixed:
            continue
        indices, requested = fixed[window.label]
        if not valid_fixed_window(indices, requested, times):
            continue
        legacy_indices = list(range(window.start_frame, window.end_frame, 2))
        rows.append({"path": clip.relative_path.as_posix(), "label": window.label,
                     "target": int(window.label == "incident"), "collision_type": clip.collision_type,
                     "region": clip.region, "quality": clip.quality, "iid_split": clip.iid_split,
                     "geographic_split": clip.geographic_split, "source_frame_count": len(times),
                     "supplied_event_frame": event_frame, "legacy_effective_event_frame": clip.effective_accident_frame,
                     "event_time_sec": event_time, "legacy_indices": legacy_indices,
                     "fixed_indices": indices.tolist(), "fixed_target_times_sec": requested.tolist(),
                     "legacy_times_sec": times[legacy_indices].tolist(), "fixed_times_sec": times[indices].tolist()})
    return rows, None


def read_presentation_frames(path, indices, expected_times, target_width=320):
    import cv2

    indices = list(indices)
    if target_width <= 0 or not np.all(np.isfinite(expected_times)):
        raise ValueError("Width and expected timestamps must be valid")
    if len(indices) != len(expected_times) or not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        raise ValueError("Selected frames must be strictly increasing and paired with timestamps")
    capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    frames = []
    try:
        success, _ = capture.read()
        if not success:
            raise OSError(f"Cannot decode timestamp origin: {path}")
        origin = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000
        if not math.isfinite(origin):
            raise ValueError("Decoder must expose a finite timestamp origin")
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, indices[0]):
            raise OSError("Cannot seek to the frozen source-frame index")
        selected = dict(zip(indices, expected_times))
        for index in range(indices[0], indices[-1] + 1):
            success, frame = capture.read()
            if not success:
                raise OSError("Frozen source-frame support is not decodable")
            if index not in selected:
                continue
            actual = capture.get(cv2.CAP_PROP_POS_MSEC) / 1000 - origin
            position = capture.get(cv2.CAP_PROP_POS_FRAMES)
            if not math.isfinite(actual) or not math.isfinite(position) or abs(actual - selected[index]) > TIMESTAMP_TOLERANCE_SEC or abs(position - index - 1) > .5:
                raise ValueError("Decoded frame identity or PTS differs from the frozen manifest")
            if frame.shape[1] > target_width:
                height = max(1, round(frame.shape[0] * target_width / frame.shape[1]))
                frame = cv2.resize(frame, (target_width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
    finally:
        capture.release()
    return frames
