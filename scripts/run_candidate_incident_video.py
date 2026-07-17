from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.incident_candidate_roi import (
    CANDIDATE_MOTION_FEATURE_NAMES,
    candidate_motion_sequence_matrix,
    compute_candidate_motion_step,
)
from illegal_parking.incident_tcn_inference import (
    IncidentTcnPredictor,
    build_sequence_windows,
    persistent_positive_flags,
    summarize_normal_video_predictions,
)
from illegal_parking.incident_ultralytics import track_vehicle_frames


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run annotation-free candidate ROI and calibrated TCN inference on a video."
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--model", default="models/candidate500_tcn_only_iid_fpr20.pt")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    parser.add_argument("--target-fps", type=float, default=7.5)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--sequence-steps", type=int, default=15)
    parser.add_argument("--hop-steps", type=int, default=15)
    parser.add_argument("--normal-only", action="store_true")
    parser.add_argument("--min-positive-windows", type=int, default=2)
    parser.add_argument("--output-video", default="outputs/accident/candidate_incident_demo.mp4")
    parser.add_argument("--output-report", default="outputs/accident/candidate_incident_demo.json")
    args = parser.parse_args()

    if args.target_fps <= 0 or args.target_width <= 0 or args.min_positive_windows <= 0:
        raise SystemExit("target FPS, width, and min-positive-windows must be positive")

    from ultralytics import YOLO

    input_path = _resolve(args.input)
    output_video = _resolve(args.output_video)
    output_report = _resolve(args.output_report)
    frames, source_indices, source_fps = _read_sampled_video(
        input_path,
        target_fps=args.target_fps,
        target_width=args.target_width,
    )
    detector = YOLO(str(_resolve(args.detector)))
    tracked = track_vehicle_frames(
        detector,
        frames,
        source_indices,
        device=args.device,
        image_size=args.image_size,
        confidence=args.confidence,
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
    predictor_device = "cuda" if args.device not in {"cpu", "-1"} else "cpu"
    predictor = IncidentTcnPredictor.from_checkpoint(_resolve(args.model), predictor_device)
    unknown_features = sorted(set(predictor.feature_names) - set(CANDIDATE_MOTION_FEATURE_NAMES))
    if unknown_features:
        raise SystemExit(
            "Checkpoint requests features not produced by the candidate ROI extractor: "
            f"{unknown_features}"
        )
    feature_indices = [CANDIDATE_MOTION_FEATURE_NAMES.index(name) for name in predictor.feature_names]
    step_features = step_features[:, feature_indices]
    windows, starts = build_sequence_windows(
        step_features,
        sequence_steps=args.sequence_steps,
        hop_steps=args.hop_steps,
    )
    probabilities = predictor.predict_probabilities(windows)
    review_windows = persistent_positive_flags(
        probabilities,
        predictor.threshold,
        args.min_positive_windows,
    )
    frame_probabilities = _expand_window_probabilities(
        len(frames), starts, probabilities, args.sequence_steps
    )
    frame_reviews = _expand_window_flags(
        len(frames), starts, review_windows, args.sequence_steps
    )

    sampled_duration_sec = (source_indices[-1] - source_indices[0] + 1) / source_fps
    sampled_fps = len(frames) / sampled_duration_sec
    output_video.parent.mkdir(parents=True, exist_ok=True)
    _write_annotated_video(
        output_video,
        frames,
        source_indices,
        tracked,
        steps,
        frame_probabilities,
        frame_reviews,
        predictor.threshold,
        sampled_fps,
    )

    report = {
        "input": str(input_path),
        "model": str(_resolve(args.model)),
        "detector": str(_resolve(args.detector)),
        "annotation_free_inference": True,
        "source_fps": source_fps,
        "target_fps": args.target_fps,
        "sampled_fps": sampled_fps,
        "sampled_frames": len(frames),
        "observed_duration_sec": sampled_duration_sec,
        "sequence_steps": args.sequence_steps,
        "hop_steps": args.hop_steps,
        "threshold": predictor.threshold,
        "min_positive_windows": args.min_positive_windows,
        "window_starts": starts.tolist(),
        "window_probabilities": probabilities.tolist(),
        "window_review_flags": review_windows.tolist(),
        "candidate_sources": dict(source_counts),
        "privacy": "Heuristic lower-center blur is applied inside every tracked vehicle box.",
    }
    if args.normal_only:
        report["normal_only_evaluation"] = summarize_normal_video_predictions(
            probabilities,
            predictor.threshold,
            sampled_duration_sec,
            min_consecutive_windows=args.min_positive_windows,
        )
        report["limitations"] = [
            "The normal-only label was assigned by manual visual inspection, not an incident annotation file.",
            "This is a short public-CCTV pilot; the hourly rate is an exposure-normalized observation, not a stable field estimate.",
            "The public stream frame cadence differs from ACCIDENT training clips and is a domain-shift test.",
        ]
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_sampled_video(
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


def _expand_window_probabilities(
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


def _expand_window_flags(
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


def _write_annotated_video(
    path: Path,
    frames: list[np.ndarray],
    source_indices: list[int],
    tracked,
    steps,
    probabilities: np.ndarray,
    review_flags: np.ndarray,
    threshold: float,
    fps: float,
) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Cannot create output video: {path}")
    try:
        for index, original in enumerate(frames):
            frame = original.copy()
            for track in tracked[source_indices[index]]:
                _blur_plate_region(frame, track)
            probability = probabilities[index]
            is_positive = bool(np.isfinite(probability) and probability >= threshold)
            is_review = bool(review_flags[index])
            if index > 0 and steps[index - 1].selected is not None:
                candidate = steps[index - 1].selected
                x1, y1, x2, y2 = _pixel_bbox(candidate.bbox, width, height)
                color = (30, 50, 235) if is_review else (0, 210, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    frame,
                    f"candidate: {candidate.source} {candidate.score:.2f}",
                    (x1, max(22, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            label = "WARMUP" if not np.isfinite(probability) else (
                "REVIEW" if is_review else ("PENDING" if is_positive else "NORMAL")
            )
            score_text = "--" if not np.isfinite(probability) else f"{probability:.3f}"
            cv2.rectangle(frame, (0, 0), (width, 58), (18, 18, 18), -1)
            cv2.putText(
                frame,
                f"MODEL OUTPUT  incident={score_text}  threshold={threshold:.3f}  {label}",
                (16, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (245, 245, 245),
                2,
                cv2.LINE_AA,
            )
            writer.write(frame)
    finally:
        writer.release()


def _blur_plate_region(frame: np.ndarray, bbox) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = _pixel_bbox((bbox.x1, bbox.y1, bbox.x2, bbox.y2), width, height)
    box_width = x2 - x1
    box_height = y2 - y1
    left = x1 + round(box_width * 0.05)
    right = x2 - round(box_width * 0.05)
    top = y1 + round(box_height * 0.5)
    bottom = y1 + round(box_height * 0.95)
    if right <= left or bottom <= top:
        return
    region = frame[top:bottom, left:right]
    if region.size:
        frame[top:bottom, left:right] = cv2.GaussianBlur(region, (31, 31), 0)


def _pixel_bbox(bbox, width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    return (
        max(0, min(width - 1, round(x1 * width))),
        max(0, min(height - 1, round(y1 * height))),
        max(1, min(width, round(x2 * width))),
        max(1, min(height, round(y2 * height))),
    )


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
