from __future__ import annotations

from typing import Any

import numpy as np

from .incident_trajectory import TrackBox, build_track_boxes


VEHICLE_CLASS_IDS = (2, 3, 5, 7)


def track_vehicle_frames(
    model: Any,
    frames: list[np.ndarray],
    frame_indices: list[int],
    tracker: str = "bytetrack.yaml",
    device: str = "0",
    image_size: int = 640,
    confidence: float = 0.15,
) -> dict[int, list[TrackBox]]:
    if not frames or len(frames) != len(frame_indices):
        raise ValueError("frames and frame_indices must have equal non-zero lengths")
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between zero and one")

    tracked = {frame_index: [] for frame_index in frame_indices}
    try:
        results = list(model.track(
            source=frames,
            persist=True,
            tracker=tracker,
            device=device,
            imgsz=image_size,
            conf=confidence,
            classes=list(VEHICLE_CLASS_IDS),
            verbose=False,
        ))
        if len(results) != len(frames):
            raise RuntimeError(
                f"Tracker returned {len(results)} results for {len(frames)} frames"
            )
        for frame_index, result in zip(frame_indices, results):
            boxes = result.boxes
            if boxes is None or boxes.id is None or len(boxes) == 0:
                continue
            height, width = result.orig_shape
            tracked[frame_index] = build_track_boxes(
                frame_index=frame_index,
                boxes_xyxy=_as_numpy(boxes.xyxy),
                track_ids=_as_numpy(boxes.id),
                confidences=_as_numpy(boxes.conf),
                frame_width=width,
                frame_height=height,
            )
        return tracked
    finally:
        reset_ultralytics_trackers(model)


def reset_ultralytics_trackers(model: Any) -> None:
    predictor = getattr(model, "predictor", None)
    for tracker in getattr(predictor, "trackers", ()) or ():
        tracker.reset()


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)
