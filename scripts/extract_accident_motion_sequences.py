from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accident_dataset import (
    build_temporal_windows,
    load_accident_manifest,
    stratified_sample_clips,
)
from illegal_parking.incident_features import (
    MOTION_FEATURE_NAMES,
    compute_motion_sequence,
    motion_sequence_matrix,
    read_video_window,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Build fixed-length ACCIDENT motion tensors for temporal models.")
    parser.add_argument("--metadata", default="data/raw/accident/metadata-real.csv")
    parser.add_argument("--dataset-root", default="data/raw/accident/full")
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), default="iid")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--negative-gap-frames", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--target-width", type=int, default=160)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--output", default="data/processed/accident/motion_sequences.npz")
    parser.add_argument("--report", default="data/processed/accident/motion_sequence_report.json")
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise SystemExit("--frame-stride must be positive")
    metadata_path = _resolve(args.metadata)
    dataset_root = _resolve(args.dataset_root)
    output_path = _resolve(args.output)
    report_path = _resolve(args.report)

    clips = load_accident_manifest(metadata_path)
    if args.split != "all":
        clips = [clip for clip in clips if _split_value(clip, args.split_scheme) == args.split]
    if args.max_clips is not None:
        clips = stratified_sample_clips(clips, args.max_clips, seed=args.sampling_seed)

    expected_steps = (args.window_frames + args.frame_stride - 1) // args.frame_stride - 1
    feature_tensors: list[np.ndarray] = []
    records: list[dict] = []
    failures: list[dict[str, str]] = []
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
                frames = read_video_window(
                    video_path,
                    start_frame=window.start_frame,
                    end_frame=window.end_frame,
                    frame_stride=args.frame_stride,
                    target_width=args.target_width,
                )
                tensor = motion_sequence_matrix(compute_motion_sequence(frames, clip.bbox_normalized))
                if tensor.shape != (expected_steps, len(MOTION_FEATURE_NAMES)):
                    raise ValueError(f"Unexpected motion tensor shape: {tensor.shape}")
                feature_tensors.append(tensor)
                records.append(
                    {
                        "path": clip.relative_path.as_posix(),
                        "label": window.label,
                        "target": int(window.label == "incident"),
                        "collision_type": clip.collision_type,
                        "region": clip.region,
                        "quality": clip.quality,
                        "iid_split": clip.iid_split,
                        "geographic_split": clip.geographic_split,
                        "start_frame": window.start_frame,
                        "end_frame": window.end_frame,
                    }
                )
            except (OSError, ValueError) as exc:
                failures.append({"path": clip.relative_path.as_posix(), "error": str(exc)})
        if clip_index % 25 == 0:
            print(f"processed {clip_index}/{len(clips)} clips, rows={len(records)}, failures={len(failures)}")

    if not records:
        raise SystemExit("No motion sequences were extracted. Check the dataset paths.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray([record[name] for record in records]) for name in records[0]}
    np.savez_compressed(
        output_path,
        features=np.stack(feature_tensors).astype(np.float32),
        feature_names=np.asarray(MOTION_FEATURE_NAMES),
        **payload,
    )

    elapsed_sec = time.perf_counter() - started
    report = {
        "metadata": str(metadata_path),
        "dataset_root": str(dataset_root),
        "requested_clip_count": len(clips),
        "sequence_count": len(records),
        "sequence_shape": [len(records), expected_steps, len(MOTION_FEATURE_NAMES)],
        "failure_count": len(failures),
        "elapsed_sec": elapsed_sec,
        "clips_per_sec": len(clips) / elapsed_sec if elapsed_sec else 0.0,
        "window_frames": args.window_frames,
        "frame_stride": args.frame_stride,
        "target_width": args.target_width,
        "sampling_seed": args.sampling_seed,
        "failures": failures[:50],
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _split_value(clip, split_scheme: str) -> str:
    return clip.iid_split if split_scheme == "iid" else clip.geographic_split


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
