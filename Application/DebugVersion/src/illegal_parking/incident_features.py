from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


MOTION_FEATURE_NAMES = (
    "global_diff_mean",
    "roi_diff_mean",
    "global_change_ratio",
    "roi_change_ratio",
    "global_flow_mean",
    "roi_flow_mean",
    "global_flow_p95",
    "roi_flow_p95",
)


def compute_frame_motion(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    roi_normalized: tuple[float, float, float, float],
    change_threshold: int = 20,
) -> dict[str, float]:
    previous = _to_gray(previous_frame)
    current = _to_gray(current_frame)
    if previous.shape != current.shape:
        raise ValueError("previous and current frames must have the same shape")

    ys, xs = _roi_slices(roi_normalized, current.shape[1], current.shape[0])
    absolute_difference = cv2.absdiff(previous, current)
    flow = cv2.calcOpticalFlowFarneback(
        previous,
        current,
        None,
        0.5,
        3,
        15,
        3,
        5,
        1.2,
        0,
    )
    flow_magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
    roi_difference = absolute_difference[ys, xs]
    roi_flow = flow_magnitude[ys, xs]

    return {
        "global_diff_mean": float(np.mean(absolute_difference)),
        "roi_diff_mean": float(np.mean(roi_difference)),
        "global_change_ratio": float(np.mean(absolute_difference >= change_threshold)),
        "roi_change_ratio": float(np.mean(roi_difference >= change_threshold)),
        "global_flow_mean": float(np.mean(flow_magnitude)),
        "roi_flow_mean": float(np.mean(roi_flow)),
        "global_flow_p95": float(np.percentile(flow_magnitude, 95)),
        "roi_flow_p95": float(np.percentile(roi_flow, 95)),
    }


def compute_motion_sequence(
    frames: list[np.ndarray],
    roi_normalized: tuple[float, float, float, float],
) -> list[dict[str, float]]:
    return [
        compute_frame_motion(previous, current, roi_normalized)
        for previous, current in zip(frames, frames[1:])
    ]


def aggregate_motion_features(sequence: list[dict[str, float]]) -> dict[str, float]:
    if not sequence:
        raise ValueError("Cannot aggregate an empty motion sequence")

    feature_names = tuple(sequence[0])
    result: dict[str, float] = {}
    for feature_name in feature_names:
        values = np.asarray([frame[feature_name] for frame in sequence], dtype=np.float64)
        result[f"{feature_name}_mean"] = float(np.mean(values))
        result[f"{feature_name}_max"] = float(np.max(values))
        result[f"{feature_name}_std"] = float(np.std(values))

    peak_feature = "roi_diff_mean" if "roi_diff_mean" in feature_names else feature_names[0]
    peak_index = int(np.argmax([frame[peak_feature] for frame in sequence]))
    result["motion_peak_position"] = peak_index / max(1, len(sequence) - 1)
    return result


def extract_video_window_features(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    roi_normalized: tuple[float, float, float, float],
    frame_stride: int = 1,
    target_width: int = 320,
) -> dict[str, float]:
    frames = read_video_window(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        frame_stride=frame_stride,
        target_width=target_width,
    )
    return aggregate_motion_features(compute_motion_sequence(frames, roi_normalized))


def read_video_window(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    frame_stride: int = 1,
    target_width: int = 320,
) -> list[np.ndarray]:
    if frame_stride <= 0:
        raise ValueError("frame_stride must be positive")
    if end_frame <= start_frame:
        raise ValueError("end_frame must be greater than start_frame")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames: list[np.ndarray] = []
    try:
        for frame_index in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_index - start_frame) % frame_stride == 0:
                frames.append(_resize(frame, target_width))
    finally:
        capture.release()

    if len(frames) < 2:
        raise ValueError(f"Video window has fewer than two readable frames: {video_path}")
    return frames


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")


def _roi_slices(
    roi: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[slice, slice]:
    x1, y1, x2, y2 = roi
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"Invalid normalized ROI: {roi}")
    left = min(width - 1, int(x1 * width))
    top = min(height - 1, int(y1 * height))
    right = max(left + 1, min(width, int(np.ceil(x2 * width))))
    bottom = max(top + 1, min(height, int(np.ceil(y2 * height))))
    return slice(top, bottom), slice(left, right)


def _resize(frame: np.ndarray, target_width: int) -> np.ndarray:
    if target_width <= 0 or frame.shape[1] <= target_width:
        return frame
    scale = target_width / frame.shape[1]
    target_height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
