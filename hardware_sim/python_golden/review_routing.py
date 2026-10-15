from __future__ import annotations

import math

import numpy as np


def coalesce_candidate_flags(
    records: list[dict], flags: np.ndarray, *, max_gap_steps: int,
    minimum_iou: float = 0.1, source_frame_stride: int = 2,
) -> np.ndarray:
    """Emit the first overlapping proposal in each causal source-local episode."""
    flags = np.asarray(flags, dtype=bool).reshape(-1)
    if len(records) != len(flags) or max_gap_steps < 0 or source_frame_stride <= 0:
        raise ValueError("Matching records, nonnegative gap, and positive source stride are required")
    if not math.isfinite(minimum_iou) or not 0 <= minimum_iou <= 1:
        raise ValueError("minimum_iou must be finite and in [0, 1]")
    emitted = np.zeros(len(flags), dtype=bool)
    previous = None
    last_box = None
    last_candidate = -max_gap_steps - 1
    for index, (record, positive) in enumerate(zip(records, flags)):
        source = (str(record["path"]), int(record["frame_index"]))
        if previous is None or source[0] != previous[0] or source[1] - previous[1] != source_frame_stride:
            last_box = None
        previous = source
        if not positive:
            continue
        box = record.get("candidate_bbox")
        if box is not None:
            box = np.asarray(box, dtype=float)
            if box.shape != (4,) or not np.all(np.isfinite(box)) or np.any(box[2:] <= box[:2]):
                raise ValueError("Candidate boxes must be finite positive-area xyxy coordinates")
        duplicate = False
        if box is not None and last_box is not None and index - last_candidate <= max_gap_steps:
            intersection = np.maximum(0, np.minimum(box[2:], last_box[2:]) - np.maximum(box[:2], last_box[:2])).prod()
            union = np.prod(box[2:] - box[:2]) + np.prod(last_box[2:] - last_box[:2]) - intersection
            duplicate = intersection / union >= minimum_iou
        emitted[index] = not duplicate
        last_box = box
        last_candidate = index
    return emitted


def meter_review_flags(
    flags: np.ndarray, *, camera_fps: float, rate_per_sec: float, burst: int = 1,
) -> np.ndarray:
    """Strict token-bucket admission, with rotating priority for simultaneous cameras."""
    flags = np.asarray(flags, dtype=bool)
    if flags.ndim != 2 or flags.shape[1] == 0:
        raise ValueError("flags must have shape (time_steps, cameras)")
    if not all(math.isfinite(value) and value > 0 for value in (camera_fps, rate_per_sec)) or burst < 1 or int(burst) != burst:
        raise ValueError("Finite positive rates and a positive integer burst are required")
    admitted = np.zeros_like(flags)
    credits = float(burst)
    next_camera = 0
    for frame_index, proposals in enumerate(flags):
        if frame_index:
            credits = min(float(burst), credits + rate_per_sec / camera_fps)
        order = [(next_camera + offset) % flags.shape[1] for offset in range(flags.shape[1])]
        for camera in order:
            if proposals[camera] and credits + 1e-12 >= 1:
                admitted[frame_index, camera] = True
                credits = max(0.0, credits - 1)
                next_camera = (camera + 1) % flags.shape[1]
    return admitted
