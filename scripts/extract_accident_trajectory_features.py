from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accident_dataset import (
    build_temporal_windows,
    load_accident_manifest,
    stratified_sample_clips,
)
from illegal_parking.incident_trajectory import (
    aggregate_trajectory_features,
    compute_trajectory_sequence,
)
from illegal_parking.incident_ultralytics import track_vehicle_frames


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract ByteTrack trajectory and image-plane TTC features from ACCIDENT clips."
    )
    parser.add_argument("--metadata", default="data/raw/accident/metadata-real.csv")
    parser.add_argument("--dataset-root", default="data/raw/accident/full")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), default="iid")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--negative-gap-frames", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--max-ttc-sec", type=float, default=10.0)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/accident/trajectory_features.csv")
    parser.add_argument("--report", default="data/processed/accident/trajectory_feature_report.json")
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between zero and one")

    from ultralytics import YOLO

    metadata_path = _resolve(args.metadata)
    dataset_root = _resolve(args.dataset_root)
    output_path = _resolve(args.output)
    report_path = _resolve(args.report)
    model_path = _resolve(args.model)

    clips = load_accident_manifest(metadata_path)
    if args.split != "all":
        clips = [clip for clip in clips if _split_value(clip, args.split_scheme) == args.split]
    if args.max_clips is not None:
        clips = stratified_sample_clips(clips, args.max_clips, seed=args.sampling_seed)

    model = YOLO(str(model_path))
    rows: list[dict] = []
    failures: list[dict[str, str]] = []
    zero_detection_windows = 0
    started = time.perf_counter()
    for clip_index, clip in enumerate(clips, start=1):
        video_path = dataset_root / clip.relative_path
        if not video_path.is_file():
            failures.append({"path": clip.relative_path.as_posix(), "error": "missing video"})
            continue
        for window in build_temporal_windows(
            clip,
            window_frames=args.window_frames,
            negative_gap_frames=args.negative_gap_frames,
        ):
            try:
                frames, frame_indices = _read_window(
                    video_path,
                    window.start_frame,
                    window.end_frame,
                    args.frame_stride,
                )
                tracked_by_frame = track_vehicle_frames(
                    model,
                    frames,
                    frame_indices,
                    tracker=args.tracker,
                    device=args.device,
                    image_size=args.image_size,
                    confidence=args.confidence,
                )
                observations = [
                    box
                    for frame_index in frame_indices
                    for box in tracked_by_frame[frame_index]
                ]
                if not observations:
                    zero_detection_windows += 1
                sequence = compute_trajectory_sequence(
                    observations,
                    fps=clip.frames_per_second,
                    max_ttc_sec=args.max_ttc_sec,
                    frame_indices=frame_indices,
                )
                features = aggregate_trajectory_features(sequence)
                rows.append(
                    {
                        "path": clip.relative_path.as_posix(),
                        "label": window.label,
                        "target": int(window.label == "incident"),
                        "collision_type": clip.collision_type,
                        "region": clip.region,
                        "quality": clip.quality,
                        "day_time": clip.day_time,
                        "iid_split": clip.iid_split,
                        "geographic_split": clip.geographic_split,
                        "start_frame": window.start_frame,
                        "end_frame": window.end_frame,
                        **features,
                    }
                )
            except (OSError, RuntimeError, ValueError) as exc:
                failures.append({"path": clip.relative_path.as_posix(), "error": str(exc)})
        if clip_index % 10 == 0:
            print(f"processed {clip_index}/{len(clips)} clips, rows={len(rows)}, failures={len(failures)}")

    if not rows:
        raise SystemExit("No trajectory features were extracted. Check the model and dataset paths.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    elapsed_sec = time.perf_counter() - started
    report = {
        "metadata": str(metadata_path),
        "dataset_root": str(dataset_root),
        "model": str(model_path),
        "tracker": args.tracker,
        "device": args.device,
        "requested_clip_count": len(clips),
        "feature_row_count": len(rows),
        "zero_detection_window_count": zero_detection_windows,
        "failure_count": len(failures),
        "elapsed_sec": elapsed_sec,
        "clips_per_sec": len(clips) / elapsed_sec if elapsed_sec else 0.0,
        "frame_stride": args.frame_stride,
        "image_size": args.image_size,
        "confidence": args.confidence,
        "max_ttc_sec": args.max_ttc_sec,
        "sampling_seed": args.sampling_seed,
        "failures": failures[:50],
        "ttc_interpretation": "Image-plane risk proxy; not metric distance or physical TTC.",
        "sources": [
            "https://github.com/accidentbench/ACCIDENT/blob/main/baselines/heuristic/bbox_dynamics.py",
            "https://docs.ultralytics.com/modes/track/",
        ],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_window(
    video_path: Path,
    start_frame: int,
    end_frame: int,
    frame_stride: int,
) -> tuple[list, list[int]]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frames = []
    frame_indices: list[int] = []
    try:
        for frame_index in range(start_frame, end_frame):
            ok, frame = capture.read()
            if not ok:
                break
            if (frame_index - start_frame) % frame_stride == 0:
                frames.append(frame)
                frame_indices.append(frame_index)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Video window has no readable frames: {video_path}")
    return frames, frame_indices


def _split_value(clip, split_scheme: str) -> str:
    return clip.iid_split if split_scheme == "iid" else clip.geographic_split


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
