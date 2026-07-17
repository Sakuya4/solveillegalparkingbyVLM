from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from math import hypot
from typing import Protocol

import cv2
import numpy as np

from .incident_features import compute_frame_motion
from .incident_trajectory import TrackBox


CANDIDATE_MOTION_FEATURE_NAMES = (
    "global_diff_mean",
    "candidate_diff_mean",
    "global_change_ratio",
    "candidate_change_ratio",
    "global_flow_mean",
    "candidate_flow_mean",
    "global_flow_p95",
    "candidate_flow_p95",
    "candidate_available",
    "candidate_score",
    "candidate_area_ratio",
    "candidate_from_track",
    "candidate_from_motion",
    "candidate_from_sam2",
)


@dataclass(frozen=True)
class CandidateRegion:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    source: str
    track_ids: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        values = np.asarray((self.x1, self.y1, self.x2, self.y2, self.score))
        if not np.all(np.isfinite(values)):
            raise ValueError("Candidate values must be finite")
        if not (0.0 <= self.x1 < self.x2 <= 1.0 and 0.0 <= self.y1 < self.y2 <= 1.0):
            raise ValueError("Candidate coordinates must be normalized and ordered")
        if not 0.0 <= self.score <= 1.0:
            raise ValueError("Candidate score must be between zero and one")
        if not self.source:
            raise ValueError("Candidate source must not be empty")
        if any(track_id < 0 for track_id in self.track_ids):
            raise ValueError("Candidate track IDs must be non-negative")

    @property
    def bbox(self) -> tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        return (self.x2 - self.x1) * (self.y2 - self.y1)


@dataclass(frozen=True)
class CandidateMotionStep:
    features: dict[str, float]
    candidates: tuple[CandidateRegion, ...]
    selected: CandidateRegion | None


@dataclass(frozen=True)
class ProposalDiagnostics:
    candidate_count: int
    best_iou: float
    hit_at_0_1: bool
    hit_at_0_3: bool


class CandidateMaskRefiner(Protocol):
    source_name: str

    def refine(self, frame_bgr: np.ndarray, candidate: CandidateRegion) -> CandidateRegion:
        ...


def propose_track_regions(
    tracks: list[TrackBox],
    pair_distance_threshold: float = 0.35,
    expansion_ratio: float = 0.1,
    max_tracks: int = 32,
) -> list[CandidateRegion]:
    if pair_distance_threshold <= 0:
        raise ValueError("pair_distance_threshold must be positive")
    if expansion_ratio < 0:
        raise ValueError("expansion_ratio must be non-negative")
    if max_tracks <= 0:
        raise ValueError("max_tracks must be positive")

    selected_tracks = sorted(
        tracks,
        key=lambda track: (-track.confidence, track.track_id),
    )[:max_tracks]

    candidates = [
        _candidate_from_bbox(
            _expand_bbox((track.x1, track.y1, track.x2, track.y2), expansion_ratio),
            score=min(1.0, 0.45 + 0.45 * track.confidence),
            source="track",
            track_ids=(track.track_id,),
        )
        for track in selected_tracks
    ]
    for first, second in combinations(sorted(selected_tracks, key=lambda track: track.track_id), 2):
        distance = hypot(first.center[0] - second.center[0], first.center[1] - second.center[1])
        if distance > pair_distance_threshold and _bbox_iou(
            (first.x1, first.y1, first.x2, first.y2),
            (second.x1, second.y1, second.x2, second.y2),
        ) == 0.0:
            continue
        proximity = max(0.0, 1.0 - distance / pair_distance_threshold)
        confidence = (first.confidence + second.confidence) / 2.0
        score = min(1.0, 0.55 + 0.25 * confidence + 0.2 * proximity)
        candidates.append(
            _candidate_from_bbox(
                _expand_bbox(_union_bbox(
                    (first.x1, first.y1, first.x2, first.y2),
                    (second.x1, second.y1, second.x2, second.y2),
                ), expansion_ratio),
                score=score,
                source="track_pair",
                track_ids=(first.track_id, second.track_id),
            )
        )
    return sorted(candidates, key=_candidate_sort_key)


def propose_motion_regions(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    change_threshold: int = 20,
    min_area_ratio: float = 0.002,
    max_regions: int = 8,
    expansion_ratio: float = 0.05,
) -> list[CandidateRegion]:
    previous = _to_gray(previous_frame)
    current = _to_gray(current_frame)
    if previous.shape != current.shape:
        raise ValueError("previous and current frames must have the same shape")
    if not 0 <= change_threshold <= 255:
        raise ValueError("change_threshold must be between zero and 255")
    if not 0.0 < min_area_ratio <= 1.0:
        raise ValueError("min_area_ratio must be in (0, 1]")
    if max_regions <= 0:
        raise ValueError("max_regions must be positive")

    difference = cv2.absdiff(previous, current)
    changed = (difference >= change_threshold).astype(np.uint8) * 255
    kernel = np.ones((3, 3), dtype=np.uint8)
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, kernel)
    changed = cv2.dilate(changed, kernel, iterations=1)
    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = current.shape
    frame_area = float(height * width)
    candidates: list[CandidateRegion] = []
    for contour in contours:
        contour_area = float(cv2.contourArea(contour))
        area_ratio = contour_area / frame_area
        if area_ratio < min_area_ratio:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        bbox = _expand_bbox(
            (x / width, y / height, (x + box_width) / width, (y + box_height) / height),
            expansion_ratio,
        )
        fill_ratio = min(1.0, contour_area / max(1.0, box_width * box_height))
        score = min(1.0, 0.35 + 0.35 * fill_ratio + 0.3 * min(1.0, area_ratio / 0.05))
        candidates.append(_candidate_from_bbox(bbox, score=score, source="motion"))
    return sorted(candidates, key=_candidate_sort_key)[:max_regions]


def fuse_candidate_regions(
    track_candidates: list[CandidateRegion],
    motion_candidates: list[CandidateRegion],
    overlap_threshold: float = 0.2,
    max_regions: int = 8,
) -> list[CandidateRegion]:
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be between zero and one")
    if max_regions <= 0:
        raise ValueError("max_regions must be positive")

    fused: list[CandidateRegion] = []
    matched_motion: set[int] = set()
    for track in track_candidates:
        best_index = None
        best_overlap = 0.0
        for index, motion in enumerate(motion_candidates):
            overlap = _intersection_over_smaller(track.bbox, motion.bbox)
            if overlap >= overlap_threshold and overlap > best_overlap:
                best_index = index
                best_overlap = overlap
        if best_index is None:
            fused.append(track)
            continue
        motion = motion_candidates[best_index]
        matched_motion.add(best_index)
        fused.append(
            _candidate_from_bbox(
                _union_bbox(track.bbox, motion.bbox),
                score=1.0 - (1.0 - track.score) * (1.0 - motion.score),
                source="track_motion",
                track_ids=track.track_ids,
            )
        )
    fused.extend(
        motion for index, motion in enumerate(motion_candidates) if index not in matched_motion
    )
    return _non_maximum_suppression(sorted(fused, key=_candidate_sort_key), max_regions)


def compute_candidate_motion_step(
    previous_frame: np.ndarray,
    current_frame: np.ndarray,
    tracks: list[TrackBox],
    mask_refiner: CandidateMaskRefiner | None = None,
) -> CandidateMotionStep:
    track_candidates = propose_track_regions(tracks)
    motion_candidates = propose_motion_regions(previous_frame, current_frame)
    candidates = fuse_candidate_regions(track_candidates, motion_candidates)
    selected = candidates[0] if candidates else None
    if selected is not None and mask_refiner is not None:
        selected = mask_refiner.refine(current_frame, selected)

    roi = selected.bbox if selected is not None else (0.0, 0.0, 1.0, 1.0)
    motion = compute_frame_motion(previous_frame, current_frame, roi)
    source = selected.source if selected is not None else ""
    features = {
        "global_diff_mean": motion["global_diff_mean"],
        "candidate_diff_mean": motion["roi_diff_mean"],
        "global_change_ratio": motion["global_change_ratio"],
        "candidate_change_ratio": motion["roi_change_ratio"],
        "global_flow_mean": motion["global_flow_mean"],
        "candidate_flow_mean": motion["roi_flow_mean"],
        "global_flow_p95": motion["global_flow_p95"],
        "candidate_flow_p95": motion["roi_flow_p95"],
        "candidate_available": float(selected is not None),
        "candidate_score": selected.score if selected is not None else 0.0,
        "candidate_area_ratio": selected.area if selected is not None else 1.0,
        "candidate_from_track": float("track" in source),
        "candidate_from_motion": float("motion" in source),
        "candidate_from_sam2": float("sam2" in source),
    }
    return CandidateMotionStep(
        features=features,
        candidates=tuple(candidates),
        selected=selected,
    )


def aggregate_candidate_motion_features(
    sequence: list[CandidateMotionStep],
) -> dict[str, float]:
    if not sequence:
        raise ValueError("Cannot aggregate an empty candidate motion sequence")
    result: dict[str, float] = {}
    for feature_name in CANDIDATE_MOTION_FEATURE_NAMES:
        values = np.asarray([step.features[feature_name] for step in sequence], dtype=np.float64)
        result[f"{feature_name}_mean"] = float(np.mean(values))
        result[f"{feature_name}_max"] = float(np.max(values))
        result[f"{feature_name}_std"] = float(np.std(values))
    peak_index = int(np.argmax([step.features["candidate_diff_mean"] for step in sequence]))
    result["candidate_peak_position"] = peak_index / max(1, len(sequence) - 1)
    return result


def candidate_motion_sequence_matrix(sequence: list[CandidateMotionStep]) -> np.ndarray:
    if not sequence:
        raise ValueError("Cannot convert an empty candidate motion sequence")
    matrix = np.asarray(
        [
            [step.features[feature_name] for feature_name in CANDIDATE_MOTION_FEATURE_NAMES]
            for step in sequence
        ],
        dtype=np.float32,
    )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("Candidate motion sequence contains non-finite values")
    return matrix


def evaluate_candidate_proposals(
    candidates: list[CandidateRegion] | tuple[CandidateRegion, ...],
    ground_truth_bbox: tuple[float, float, float, float],
) -> ProposalDiagnostics:
    _validate_bbox(ground_truth_bbox)
    best_iou = max((_bbox_iou(candidate.bbox, ground_truth_bbox) for candidate in candidates), default=0.0)
    return ProposalDiagnostics(
        candidate_count=len(candidates),
        best_iou=best_iou,
        hit_at_0_1=best_iou >= 0.1,
        hit_at_0_3=best_iou >= 0.3,
    )


def _candidate_from_bbox(
    bbox: tuple[float, float, float, float],
    score: float,
    source: str,
    track_ids: tuple[int, ...] = (),
) -> CandidateRegion:
    return CandidateRegion(*bbox, score=score, source=source, track_ids=track_ids)


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    ratio: float,
) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    margin_x = (x2 - x1) * ratio
    margin_y = (y2 - y1) * ratio
    return (
        max(0.0, x1 - margin_x),
        max(0.0, y1 - margin_y),
        min(1.0, x2 + margin_x),
        min(1.0, y2 + margin_y),
    )


def _union_bbox(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(first[0], second[0]),
        min(first[1], second[1]),
        max(first[2], second[2]),
        max(first[3], second[3]),
    )


def _bbox_iou(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    intersection = _intersection_area(first, second)
    union = _bbox_area(first) + _bbox_area(second) - intersection
    return intersection / union if union > 0 else 0.0


def _intersection_over_smaller(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    smaller = min(_bbox_area(first), _bbox_area(second))
    return _intersection_area(first, second) / smaller if smaller > 0 else 0.0


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    return width * height


def _bbox_area(bbox: tuple[float, float, float, float]) -> float:
    return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


def _non_maximum_suppression(
    candidates: list[CandidateRegion],
    max_regions: int,
    iou_threshold: float = 0.8,
) -> list[CandidateRegion]:
    kept: list[CandidateRegion] = []
    for candidate in candidates:
        if all(_bbox_iou(candidate.bbox, existing.bbox) < iou_threshold for existing in kept):
            kept.append(candidate)
        if len(kept) >= max_regions:
            break
    return kept


def _candidate_sort_key(candidate: CandidateRegion) -> tuple[float, float, str]:
    return (-candidate.score, candidate.area, candidate.source)


def _validate_bbox(bbox: tuple[float, float, float, float]) -> None:
    if len(bbox) != 4 or not (0.0 <= bbox[0] < bbox[2] <= 1.0 and 0.0 <= bbox[1] < bbox[3] <= 1.0):
        raise ValueError("Bounding box must be normalized and ordered")


def _to_gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    raise ValueError(f"Unsupported frame shape: {frame.shape}")
