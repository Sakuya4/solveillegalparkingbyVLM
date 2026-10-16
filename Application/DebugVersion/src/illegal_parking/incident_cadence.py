from __future__ import annotations

import math

from .accident_dataset import build_temporal_windows


TARGET_FPS = 7.5
SAMPLE_COUNT = 16


def source_time_grid(frame_count, source_fps, target_fps=TARGET_FPS):
    """Constant reported-FPS frame-index grid, not a presentation-time sampler."""
    if not isinstance(frame_count, int) or frame_count < 1:
        raise ValueError("Frame count must be a positive integer")
    if not math.isfinite(source_fps) or not math.isfinite(target_fps) or min(source_fps, target_fps) <= 0:
        raise ValueError("Sampling rates must be finite and positive")
    if source_fps < target_fps:
        raise ValueError("Fixed-cadence extraction does not duplicate low-FPS frames")
    # First available source frame at or after each global target timestamp.
    indices = []
    slot = 0
    while True:
        index = math.ceil(slot * source_fps / target_fps - 1e-9)
        if index >= frame_count:
            break
        indices.append(index)
        slot += 1
    return indices


def paired_cadence_windows(clip, source_fps, frame_count):
    """Audit nominal-clock pairing without changing deployment or cohort defaults."""
    from bisect import bisect_left

    if not math.isfinite(source_fps) or source_fps <= 0 or frame_count < 1:
        raise ValueError("Container timing must be valid")
    if source_fps < TARGET_FPS:
        return [], "source_fps_below_target"
    if frame_count not in (clip.frame_count, clip.frame_count + 1):
        return [], "container_metadata_frame_count_mismatch"
    grid = source_time_grid(clip.frame_count, source_fps)
    legacy = build_temporal_windows(clip, window_frames=32, negative_gap_frames=8)
    if not legacy or len(grid) < SAMPLE_COUNT:
        return [], "insufficient_paired_context"
    first = max(0, min(bisect_left(grid, clip.effective_accident_frame) - 8, len(grid) - SAMPLE_COUNT))
    fixed = {"incident": grid[first:first + SAMPLE_COUNT]}
    legacy_incident = next(window for window in legacy if window.label == "incident")
    if min(fixed["incident"][-1], legacy_incident.end_frame - 2) < clip.effective_accident_frame:
        return [], "event_after_last_sampled_frame"
    normal_end = first - 4
    if normal_end >= SAMPLE_COUNT:
        fixed["normal"] = grid[normal_end - SAMPLE_COUNT:normal_end]
    rows = []
    for window in legacy:
        if window.label not in fixed:
            continue
        rows.append({"path": clip.relative_path.as_posix(), "label": window.label,
                     "target": int(window.label == "incident"), "collision_type": clip.collision_type,
                     "region": clip.region, "quality": clip.quality,
                     "iid_split": clip.iid_split, "geographic_split": clip.geographic_split,
                     "source_fps": source_fps, "source_frame_count": frame_count,
                     "support_frame_count": clip.frame_count,
                     "legacy_indices": list(range(window.start_frame, window.end_frame, 2)),
                     "fixed_indices": fixed[window.label]})
    return rows, None
