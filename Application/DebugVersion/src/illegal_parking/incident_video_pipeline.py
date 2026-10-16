from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .incident_candidate_roi import (
    CANDIDATE_MOTION_FEATURE_NAMES,
    candidate_motion_sequence_matrix,
    compute_candidate_motion_step,
)
from .incident_tcn_inference import (
    IncidentTcnPredictor,
    build_sequence_windows,
    persistent_positive_flags,
)
from .incident_ultralytics import track_vehicle_frames


@dataclass
class CandidateIncidentInference:
    report: dict
    frames: list[np.ndarray]
    source_indices: list[int]
    tracked: dict
    steps: list
    frame_probabilities: np.ndarray
    frame_reviews: np.ndarray


def run_candidate_incident_inference(
    input_path: Path,
    detector,
    predictor: IncidentTcnPredictor,
    *,
    detector_path: Path,
    model_path: Path,
    device: str,
    target_fps: float,
    target_width: int,
    image_size: int,
    confidence: float,
    sequence_steps: int,
    hop_steps: int,
    min_positive_windows: int,
    feature_width: int | None = None,
) -> CandidateIncidentInference:
    frames, feature_frames, source_indices, source_fps = _read_sampled_video_views(
        input_path,
        target_fps=target_fps,
        target_width=target_width,
        feature_width=feature_width,
    )
    tracked = track_vehicle_frames(
        detector,
        feature_frames,
        source_indices,
        device=device,
        image_size=image_size,
        confidence=confidence,
    )

    steps = []
    source_counts: Counter[str] = Counter()
    for index in range(1, len(frames)):
        step = compute_candidate_motion_step(
            feature_frames[index - 1],
            feature_frames[index],
            tracked[source_indices[index]],
        )
        steps.append(step)
        source_counts[step.selected.source if step.selected else "global_fallback"] += 1

    step_features = candidate_motion_sequence_matrix(steps)
    unknown_features = sorted(set(predictor.feature_names) - set(CANDIDATE_MOTION_FEATURE_NAMES))
    if unknown_features:
        raise ValueError(
            "Checkpoint requests features not produced by the candidate ROI extractor: "
            f"{unknown_features}"
        )
    feature_indices = [CANDIDATE_MOTION_FEATURE_NAMES.index(name) for name in predictor.feature_names]
    windows, starts = build_sequence_windows(
        step_features[:, feature_indices],
        sequence_steps=sequence_steps,
        hop_steps=hop_steps,
    )
    probabilities = predictor.predict_probabilities(windows)
    review_windows = persistent_positive_flags(
        probabilities,
        predictor.threshold,
        min_positive_windows,
    )
    frame_probabilities = expand_window_probabilities(
        len(frames), starts, probabilities, sequence_steps
    )
    frame_reviews = expand_window_flags(len(frames), starts, review_windows, sequence_steps)
    sampled_duration_sec = (source_indices[-1] - source_indices[0] + 1) / source_fps
    sampled_fps = len(frames) / sampled_duration_sec
    report = {
        "input": str(input_path),
        "model": str(model_path),
        "detector": str(detector_path),
        "annotation_free_inference": True,
        "source_fps": source_fps,
        "target_fps": target_fps,
        "display_width": target_width,
        "feature_width_limit": feature_width,
        "feature_frame_width": feature_frames[0].shape[1],
        "feature_resolution_protocol": "legacy_display_resolution" if feature_width is None else "source_downsample_no_upscale",
        "sampled_fps": sampled_fps,
        "sampled_frames": len(frames),
        "observed_duration_sec": sampled_duration_sec,
        "sequence_steps": sequence_steps,
        "hop_steps": hop_steps,
        "threshold": predictor.threshold,
        "min_positive_windows": min_positive_windows,
        "window_starts": starts.tolist(),
        "window_probabilities": probabilities.tolist(),
        "window_review_flags": review_windows.tolist(),
        "candidate_sources": dict(source_counts),
        "privacy": "Heuristic lower-center blur is applied inside every tracked vehicle box.",
    }
    return CandidateIncidentInference(
        report=report,
        frames=frames,
        source_indices=source_indices,
        tracked=tracked,
        steps=steps,
        frame_probabilities=frame_probabilities,
        frame_reviews=frame_reviews,
    )


def read_sampled_video(
    path: Path,
    target_fps: float,
    target_width: int,
) -> tuple[list[np.ndarray], list[int], float]:
    frames, _, indices, source_fps = _read_sampled_video_views(path, target_fps, target_width)
    return frames, indices, source_fps


def _read_sampled_video_views(
    path: Path,
    target_fps: float,
    target_width: int,
    feature_width: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], float]:
    if not math.isfinite(target_fps) or target_fps <= 0 or target_width <= 0 or (feature_width is not None and feature_width <= 0):
        raise ValueError("Sampling cadence and frame widths must be positive")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not math.isfinite(source_fps) or source_fps <= 0:
        capture.release()
        raise ValueError("Video does not report a valid FPS")
    sample_period = 1.0 / min(target_fps, source_fps)
    next_sample_sec = 0.0
    frames: list[np.ndarray] = []
    feature_frames: list[np.ndarray] = []
    indices: list[int] = []
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_sec = frame_index / source_fps
            if timestamp_sec + 1e-9 >= next_sample_sec:
                height, width = frame.shape[:2]
                target_height = max(1, round(height * target_width / width))
                frames.append(cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA))
                if feature_width is None:
                    feature_frames.append(frames[-1])
                elif width <= feature_width:
                    feature_frames.append(frame)
                else:
                    feature_height = max(1, round(height * feature_width / width))
                    feature_frames.append(cv2.resize(frame, (feature_width, feature_height), interpolation=cv2.INTER_AREA))
                indices.append(frame_index)
                next_sample_sec += sample_period
            frame_index += 1
    finally:
        capture.release()
    if len(frames) < 16:
        raise ValueError("Video must provide at least 16 sampled frames")
    return frames, feature_frames, indices, source_fps


def expand_window_probabilities(
    frame_count: int,
    starts: np.ndarray,
    probabilities: np.ndarray,
    sequence_steps: int,
) -> np.ndarray:
    expanded = np.full(frame_count, np.nan, dtype=np.float64)
    decisions = np.asarray(starts, dtype=np.int64) + sequence_steps
    for index, (decision, probability) in enumerate(zip(decisions, probabilities)):
        end = int(decisions[index + 1]) if index + 1 < len(decisions) else frame_count
        expanded[int(decision):min(frame_count, end)] = probability
    return expanded


def expand_window_flags(
    frame_count: int,
    starts: np.ndarray,
    flags: np.ndarray,
    sequence_steps: int,
) -> np.ndarray:
    expanded = np.zeros(frame_count, dtype=np.bool_)
    decisions = np.asarray(starts, dtype=np.int64) + sequence_steps
    for index, (decision, flag) in enumerate(zip(decisions, flags)):
        end = int(decisions[index + 1]) if index + 1 < len(decisions) else frame_count
        expanded[int(decision):min(frame_count, end)] = flag
    return expanded
