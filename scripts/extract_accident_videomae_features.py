from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accident_dataset import (
    build_temporal_windows,
    load_accident_manifest,
    stratified_sample_clips,
)
from illegal_parking.incident_videomae import (
    load_compatible_videomae_classifier,
    read_uniform_video_window,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract frozen VideoMAE embeddings from ACCIDENT temporal windows."
    )
    parser.add_argument("--metadata", default="data/raw/accident/full/extracted/metadata-real.csv")
    parser.add_argument("--dataset-root", default="data/raw/accident/full/extracted")
    parser.add_argument("--model-id", default="MCG-NJU/videomae-small-finetuned-kinetics")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-frames", type=int, default=16)
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--negative-gap-frames", type=int, default=8)
    parser.add_argument("--max-clips", type=int, default=500)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/processed/accident/videomae_small_features_500.csv",
    )
    parser.add_argument(
        "--report",
        default="data/processed/accident/videomae_small_feature_report_500.json",
    )
    args = parser.parse_args()
    if args.batch_size <= 0 or args.num_frames <= 0 or args.window_frames <= 1:
        raise SystemExit("batch size, num frames, and window frames must be positive")
    if args.num_frames > args.window_frames:
        raise SystemExit("--num-frames cannot exceed --window-frames")

    from transformers import VideoMAEImageProcessor

    metadata_path = _resolve(args.metadata)
    dataset_root = _resolve(args.dataset_root)
    output_path = _resolve(args.output)
    report_path = _resolve(args.report)
    clips = load_accident_manifest(metadata_path)
    if args.max_clips is not None:
        clips = stratified_sample_clips(clips, args.max_clips, seed=args.sampling_seed)
    window_records = [
        (clip, window)
        for clip in clips
        for window in build_temporal_windows(
            clip,
            window_frames=args.window_frames,
            negative_gap_frames=args.negative_gap_frames,
        )
    ]
    if not window_records:
        raise SystemExit("No temporal windows were produced")

    processor = VideoMAEImageProcessor.from_pretrained(args.model_id)
    model, revision = load_compatible_videomae_classifier(args.model_id, args.device)
    if model.config.num_frames != args.num_frames:
        raise SystemExit(
            f"Model expects {model.config.num_frames} frames, got --num-frames {args.num_frames}"
        )

    rows: list[dict] = []
    failures: list[dict[str, str | int]] = []
    started = time.perf_counter()
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for batch_start in range(0, len(window_records), args.batch_size):
        batch_records = window_records[batch_start : batch_start + args.batch_size]
        batch_videos: list[list[np.ndarray]] = []
        successful_records = []
        for clip, window in batch_records:
            try:
                batch_videos.append(
                    read_uniform_video_window(
                        dataset_root / window.relative_path,
                        window.start_frame,
                        window.end_frame,
                        num_frames=args.num_frames,
                    )
                )
                successful_records.append((clip, window))
            except ValueError as exc:
                failures.append({
                    "path": window.relative_path.as_posix(),
                    "start_frame": window.start_frame,
                    "error": str(exc),
                })
        if not batch_videos:
            continue
        pixel_values = processor(batch_videos, return_tensors="pt")["pixel_values"]
        pixel_values = pixel_values.to(args.device, non_blocking=True)
        autocast_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if args.device.startswith("cuda")
            else nullcontext()
        )
        with torch.inference_mode(), autocast_context:
            hidden = model.videomae(pixel_values).last_hidden_state
            embeddings = model.fc_norm(hidden.mean(dim=1))
        embeddings_np = embeddings.float().cpu().numpy()
        for (clip, window), embedding in zip(successful_records, embeddings_np):
            row = {
                "path": window.relative_path.as_posix(),
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
            }
            row.update({f"videomae_{index:03d}": float(value) for index, value in enumerate(embedding)})
            rows.append(row)
        completed = min(batch_start + args.batch_size, len(window_records))
        if completed % 80 == 0 or completed == len(window_records):
            print(f"processed {completed}/{len(window_records)} windows, failures={len(failures)}")

    if not rows:
        raise SystemExit("No VideoMAE embeddings were extracted")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    elapsed_sec = time.perf_counter() - started
    report = {
        "model": "frozen_videomae_small_embedding",
        "model_id": args.model_id,
        "model_revision": revision,
        "source": f"https://huggingface.co/{args.model_id}",
        "checkpoint_compatibility": "legacy q/v biases remapped; strict state-dict load",
        "clip_count": len(clips),
        "requested_window_count": len(window_records),
        "extracted_window_count": len(rows),
        "failure_count": len(failures),
        "failures": failures,
        "embedding_features": len([name for name in rows[0] if name.startswith("videomae_")]),
        "num_frames": args.num_frames,
        "window_frames": args.window_frames,
        "negative_gap_frames": args.negative_gap_frames,
        "sampling_seed": args.sampling_seed,
        "batch_size": args.batch_size,
        "device": args.device,
        "elapsed_sec": elapsed_sec,
        "windows_per_sec": len(rows) / elapsed_sec,
        "peak_cuda_memory_mb": (
            torch.cuda.max_memory_allocated() / 1024**2 if args.device.startswith("cuda") else None
        ),
        "annotation_contract": "Labels define windows; accident bbox/type/region are not model inputs.",
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
