from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from math import hypot

import numpy as np


TRAJECTORY_FRAME_FEATURE_NAMES = (
    "track_count",
    "pair_count",
    "total_bbox_area",
    "min_center_distance",
    "max_iou",
    "max_track_speed",
    "max_area_growth_rate",
    "max_closing_speed",
    "min_ttc_sec",
)


@dataclass(frozen=True)
class TrackBox:
    frame_index: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0

    def __post_init__(self) -> None:
        values = np.asarray((self.x1, self.y1, self.x2, self.y2, self.confidence))
        if not np.all(np.isfinite(values)):
            raise ValueError("TrackBox values must be finite")
        if self.frame_index < 0 or self.track_id < 0:
            raise ValueError("frame_index and track_id must be non-negative")
        if not (0.0 <= self.x1 < self.x2 <= 1.0 and 0.0 <= self.y1 < self.y2 <= 1.0):
            raise ValueError("TrackBox coordinates must be normalized and ordered")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between zero and one")

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


def build_track_boxes(
    frame_index: int,
    boxes_xyxy: np.ndarray,
    track_ids: np.ndarray,
    confidences: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> list[TrackBox]:
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    ids = np.asarray(track_ids).reshape(-1)
    scores = np.asarray(confidences, dtype=np.float64).reshape(-1)
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes_xyxy must have shape (N, 4)")
    if len(boxes) != len(ids) or len(boxes) != len(scores):
        raise ValueError("boxes, track_ids, and confidences must have equal lengths")
    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("frame dimensions must be positive")

    normalized: list[TrackBox] = []
    for coordinates, track_id, confidence in zip(boxes, ids, scores):
        x1, y1, x2, y2 = coordinates
        x1 = float(np.clip(x1 / frame_width, 0.0, 1.0))
        y1 = float(np.clip(y1 / frame_height, 0.0, 1.0))
        x2 = float(np.clip(x2 / frame_width, 0.0, 1.0))
        y2 = float(np.clip(y2 / frame_height, 0.0, 1.0))
        if x2 <= x1 or y2 <= y1:
            continue
        normalized.append(
            TrackBox(
                frame_index=frame_index,
                track_id=int(track_id),
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                confidence=float(confidence),
            )
        )
    return normalized


def compute_trajectory_sequence(
    observations: list[TrackBox],
    fps: float,
    max_ttc_sec: float = 10.0,
    frame_indices: list[int] | None = None,
) -> list[dict[str, float]]:
    if fps <= 0:
        raise ValueError("fps must be positive")
    if max_ttc_sec <= 0:
        raise ValueError("max_ttc_sec must be positive")
    if frame_indices is not None and (not frame_indices or min(frame_indices) < 0):
        raise ValueError("frame_indices must contain non-negative frame numbers")
    if not observations and frame_indices is None:
        raise ValueError("observations must not be empty when frame_indices are omitted")

    boxes_by_frame: dict[int, list[TrackBox]] = defaultdict(list)
    for box in observations:
        boxes_by_frame[box.frame_index].append(box)

    previous_tracks: dict[int, TrackBox] = {}
    previous_pairs: dict[tuple[int, int], tuple[int, float]] = {}
    sequence: list[dict[str, float]] = []
    sequence_frames = set(boxes_by_frame)
    if frame_indices is not None:
        sequence_frames.update(frame_indices)
    for frame_index in sorted(sequence_frames):
        boxes = sorted(boxes_by_frame[frame_index], key=lambda box: box.track_id)
        track_speeds: list[float] = []
        growth_rates: list[float] = []
        for box in boxes:
            previous = previous_tracks.get(box.track_id)
            if previous is None or previous.frame_index >= frame_index:
                track_speeds.append(0.0)
                growth_rates.append(0.0)
                continue
            delta_sec = (frame_index - previous.frame_index) / fps
            track_speeds.append(_center_distance(previous, box) / delta_sec)
            growth_rates.append(max(0.0, (box.area - previous.area) / delta_sec))

        pair_distances: list[float] = []
        pair_ious: list[float] = []
        closing_speeds: list[float] = []
        ttcs: list[float] = []
        for first, second in combinations(boxes, 2):
            pair_key = (first.track_id, second.track_id)
            distance = _center_distance(first, second)
            pair_distances.append(distance)
            pair_ious.append(_intersection_over_union(first, second))
            previous_pair = previous_pairs.get(pair_key)
            closing_speed = 0.0
            if previous_pair is not None and previous_pair[0] < frame_index:
                delta_sec = (frame_index - previous_pair[0]) / fps
                closing_speed = max(0.0, (previous_pair[1] - distance) / delta_sec)
            closing_speeds.append(closing_speed)
            ttc = distance / closing_speed if closing_speed > 1e-9 else max_ttc_sec
            ttcs.append(min(max_ttc_sec, ttc))
            previous_pairs[pair_key] = (frame_index, distance)

        sequence.append(
            {
                "track_count": float(len(boxes)),
                "pair_count": float(len(pair_distances)),
                "total_bbox_area": float(sum(box.area for box in boxes)),
                "min_center_distance": min(pair_distances, default=1.0),
                "max_iou": max(pair_ious, default=0.0),
                "max_track_speed": max(track_speeds, default=0.0),
                "max_area_growth_rate": max(growth_rates, default=0.0),
                "max_closing_speed": max(closing_speeds, default=0.0),
                "min_ttc_sec": min(ttcs, default=max_ttc_sec),
            }
        )
        previous_tracks.update((box.track_id, box) for box in boxes)
    return sequence


def aggregate_trajectory_features(sequence: list[dict[str, float]]) -> dict[str, float]:
    if not sequence:
        raise ValueError("Cannot aggregate an empty trajectory sequence")

    features: dict[str, float] = {}
    for feature_name in TRAJECTORY_FRAME_FEATURE_NAMES:
        values = np.asarray([frame[feature_name] for frame in sequence], dtype=np.float64)
        features[f"{feature_name}_mean"] = float(np.mean(values))
        features[f"{feature_name}_max"] = float(np.max(values))
        features[f"{feature_name}_min"] = float(np.min(values))
        features[f"{feature_name}_std"] = float(np.std(values))

    max_ttc = max(item["min_ttc_sec"] for item in sequence)
    risk_scores = [
        (1.0 - frame["min_ttc_sec"] / max(1e-9, max_ttc))
        + frame["max_iou"]
        + frame["max_closing_speed"] / (1.0 + frame["max_closing_speed"])
        for frame in sequence
    ]
    peak_index = int(np.argmax(risk_scores))
    features["risk_peak_position"] = peak_index / max(1, len(sequence) - 1)
    return features


def _center_distance(first: TrackBox, second: TrackBox) -> float:
    return hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])


def _intersection_over_union(first: TrackBox, second: TrackBox) -> float:
    intersection_width = max(0.0, min(first.x2, second.x2) - max(first.x1, second.x1))
    intersection_height = max(0.0, min(first.y2, second.y2) - max(first.y1, second.y1))
    intersection = intersection_width * intersection_height
    union = first.area + second.area - intersection
    return intersection / union if union > 0 else 0.0
