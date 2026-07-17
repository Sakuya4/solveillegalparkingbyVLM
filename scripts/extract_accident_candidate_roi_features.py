from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
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
from illegal_parking.incident_candidate_roi import (
    CANDIDATE_MOTION_FEATURE_NAMES,
    aggregate_candidate_motion_features,
    candidate_motion_sequence_matrix,
    compute_candidate_motion_step,
    evaluate_candidate_proposals,
)
from illegal_parking.incident_features import read_video_window
from illegal_parking.incident_sam2 import UltralyticsSam2Refiner
from illegal_parking.incident_ultralytics import track_vehicle_frames


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(
        description="Extract annotation-free tracker/motion candidate ROI features from ACCIDENT clips."
    )
    parser.add_argument("--metadata", default="data/raw/accident/metadata-real.csv")
    parser.add_argument("--dataset-root", default="data/raw/accident/full")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--device", default="0")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--sam2-model", default=None)
    parser.add_argument("--sam2-device", default=None)
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), default="iid")
    parser.add_argument("--split", choices=("all", "train", "test"), default="all")
    parser.add_argument("--window-frames", type=int, default=32)
    parser.add_argument("--negative-gap-frames", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=2)
    parser.add_argument("--target-width", type=int, default=320)
    parser.add_argument("--max-clips", type=int, default=None)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument(
        "--output",
        default="data/processed/accident/candidate_roi_features.csv",
    )
    parser.add_argument(
        "--sequence-output",
        default="data/processed/accident/candidate_roi_sequences.npz",
    )
    parser.add_argument(
        "--trace-output",
        default="outputs/accident/candidate_roi_trace.jsonl",
    )
    parser.add_argument(
        "--report",
        default="data/processed/accident/candidate_roi_feature_report.json",
    )
    args = parser.parse_args()

    if args.frame_stride <= 0 or args.target_width <= 0 or args.image_size <= 0:
        raise SystemExit("frame stride, target width, and image size must be positive")
    if not 0.0 <= args.confidence <= 1.0:
        raise SystemExit("--confidence must be between zero and one")

    from ultralytics import YOLO

    metadata_path = _resolve(args.metadata)
    dataset_root = _resolve(args.dataset_root)
    model_path = _resolve(args.model)
    output_path = _resolve(args.output)
    sequence_path = _resolve(args.sequence_output)
    trace_path = _resolve(args.trace_output)
    report_path = _resolve(args.report)

    clips = load_accident_manifest(metadata_path)
    if args.split != "all":
        clips = [clip for clip in clips if _split_value(clip, args.split_scheme) == args.split]
    if args.max_clips is not None:
        clips = stratified_sample_clips(clips, args.max_clips, seed=args.sampling_seed)

    detector = YOLO(str(model_path))
    mask_refiner = (
        UltralyticsSam2Refiner(checkpoint=args.sam2_model, device=args.sam2_device)
        if args.sam2_model
        else None
    )
    expected_steps = (args.window_frames + args.frame_stride - 1) // args.frame_stride - 1
    rows: list[dict] = []
    records: list[dict] = []
    tensors: list[np.ndarray] = []
    failures: list[dict[str, str]] = []
    source_counts: Counter[str] = Counter()
    incident_best_ious: list[float] = []
    incident_hits_0_1 = 0
    incident_hits_0_3 = 0
    proposal_count_total = 0
    step_count = 0
    available_step_count = 0

    trace_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    with trace_path.open("w", encoding="utf-8", newline="\n") as trace_handle:
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
                    frame_indices = [
                        window.start_frame + index * args.frame_stride
                        for index in range(len(frames))
                    ]
                    tracked_by_frame = track_vehicle_frames(
                        detector,
                        frames,
                        frame_indices,
                        tracker=args.tracker,
                        device=args.device,
                        image_size=args.image_size,
                        confidence=args.confidence,
                    )
                    steps = []
                    for index in range(1, len(frames)):
                        frame_index = frame_indices[index]
                        step = compute_candidate_motion_step(
                            frames[index - 1],
                            frames[index],
                            tracked_by_frame[frame_index],
                            mask_refiner=mask_refiner,
                        )
                        steps.append(step)
                        step_count += 1
                        proposal_count_total += len(step.candidates)
                        if step.selected is None:
                            source_counts["global_fallback"] += 1
                        else:
                            available_step_count += 1
                            source_counts[step.selected.source] += 1

                        if window.label == "incident":
                            diagnostic_candidates = list(step.candidates)
                            if step.selected is not None and "sam2" in step.selected.source:
                                diagnostic_candidates.append(step.selected)
                            diagnostics = evaluate_candidate_proposals(
                                diagnostic_candidates,
                                clip.bbox_normalized,
                            )
                            incident_best_ious.append(diagnostics.best_iou)
                            incident_hits_0_1 += int(diagnostics.hit_at_0_1)
                            incident_hits_0_3 += int(diagnostics.hit_at_0_3)

                        trace_handle.write(json.dumps(
                            {
                                "path": clip.relative_path.as_posix(),
                                "label": window.label,
                                "frame_index": frame_index,
                                "candidate_source": (
                                    step.selected.source if step.selected is not None else "global_fallback"
                                ),
                                "candidate_bbox": (
                                    list(step.selected.bbox) if step.selected is not None else None
                                ),
                                **step.features,
                            },
                            ensure_ascii=False,
                        ) + "\n")

                    tensor = candidate_motion_sequence_matrix(steps)
                    if tensor.shape != (expected_steps, len(CANDIDATE_MOTION_FEATURE_NAMES)):
                        raise ValueError(f"Unexpected candidate sequence shape: {tensor.shape}")
                    record = {
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
                    }
                    rows.append({**record, **aggregate_candidate_motion_features(steps)})
                    records.append(record)
                    tensors.append(tensor)
                except (OSError, RuntimeError, ValueError) as exc:
                    failures.append({"path": clip.relative_path.as_posix(), "error": str(exc)})
            if clip_index % 10 == 0:
                print(
                    f"processed {clip_index}/{len(clips)} clips, "
                    f"rows={len(rows)}, failures={len(failures)}"
                )

    if not rows:
        failure_preview = json.dumps(failures[:3], ensure_ascii=False)
        raise SystemExit(
            "No candidate ROI features were extracted. Check model, device, and dataset paths. "
            f"First failures: {failure_preview}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    sequence_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: np.asarray([record[name] for record in records]) for name in records[0]}
    np.savez_compressed(
        sequence_path,
        features=np.stack(tensors).astype(np.float32),
        feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES),
        **payload,
    )

    elapsed_sec = time.perf_counter() - started
    incident_step_count = len(incident_best_ious)
    report = {
        "metadata": str(metadata_path),
        "dataset_root": str(dataset_root),
        "detector": str(model_path),
        "tracker": args.tracker,
        "device": args.device,
        "sam2_model": args.sam2_model,
        "requested_clip_count": len(clips),
        "feature_row_count": len(rows),
        "sequence_shape": [len(rows), expected_steps, len(CANDIDATE_MOTION_FEATURE_NAMES)],
        "failure_count": len(failures),
        "elapsed_sec": elapsed_sec,
        "clips_per_sec": len(clips) / elapsed_sec if elapsed_sec else 0.0,
        "frame_stride": args.frame_stride,
        "target_width": args.target_width,
        "image_size": args.image_size,
        "confidence": args.confidence,
        "sampling_seed": args.sampling_seed,
        "runtime_proposals": {
            "step_count": step_count,
            "candidate_availability_rate": available_step_count / step_count if step_count else 0.0,
            "mean_candidate_count": proposal_count_total / step_count if step_count else 0.0,
            "selected_source_counts": dict(sorted(source_counts.items())),
        },
        "oracle_diagnostics_only": {
            "incident_step_count": incident_step_count,
            "mean_best_iou": float(np.mean(incident_best_ious)) if incident_best_ious else 0.0,
            "proposal_recall_at_0_1": incident_hits_0_1 / incident_step_count if incident_step_count else 0.0,
            "proposal_recall_at_0_3": incident_hits_0_3 / incident_step_count if incident_step_count else 0.0,
            "written_to_model_features": False,
        },
        "failures": failures[:50],
        "sources": [
            "https://docs.ultralytics.com/modes/track/",
            "https://docs.ultralytics.com/models/sam-2",
            "https://github.com/accidentbench/ACCIDENT",
        ],
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
