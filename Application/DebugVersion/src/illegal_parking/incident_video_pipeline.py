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
) -> CandidateIncidentInference:
    frames, source_indices, source_fps = read_sampled_video(
        input_path,
        target_fps=target_fps,
        target_width=target_width,
    )
    tracked = track_vehicle_frames(
        detector,
        frames,
        source_indices,
        device=device,
        image_size=image_size,
        confidence=confidence,
    )

    steps = []
    source_counts: Counter[str] = Counter()
    for index in range(1, len(frames)):
        step = compute_candidate_motion_step(
            frames[index - 1],
            frames[index],
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
                indices.append(frame_index)
                next_sample_sec += sample_period
            frame_index += 1
    finally:
        capture.release()
    if len(frames) < 16:
        raise ValueError("Video must provide at least 16 sampled frames")
    return frames, indices, source_fps


def expand_window_probabilities(
    frame_count: int,
    starts: np.ndarray,
    probabilities: np.ndarray,
    sequence_steps: int,
) -> np.ndarray:
    expanded = np.full(frame_count, np.nan, dtype=np.float64)
    for start, probability in zip(starts, probabilities):
        first_frame = int(start) + 1
        end_frame = min(frame_count, first_frame + sequence_steps)
        current = expanded[first_frame:end_frame]
        expanded[first_frame:end_frame] = np.where(
            np.isnan(current), probability, np.maximum(current, probability)
        )
    return expanded


def expand_window_flags(
    frame_count: int,
    starts: np.ndarray,
    flags: np.ndarray,
    sequence_steps: int,
) -> np.ndarray:
    expanded = np.zeros(frame_count, dtype=np.bool_)
    for start, flag in zip(starts, flags):
        first_frame = int(start) + 1
        expanded[first_frame : min(frame_count, first_frame + sequence_steps)] = flag
    return expanded
