from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.incident_batch import report_filename, select_evaluation_clips
from illegal_parking.incident_tcn_inference import IncidentTcnPredictor
from illegal_parking.incident_video_pipeline import run_candidate_incident_inference
from illegal_parking.incident_event_calibration import protocol_clips


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Batch annotation-free candidate ROI/TCN inference on an ACCIDENT test split."
    )
    parser.add_argument("--metadata", default="data/raw/accident/full/extracted/metadata-real.csv")
    parser.add_argument("--dataset-root", default="data/raw/accident/full/extracted")
    parser.add_argument("--model", default="models/candidate500_tcn_only_iid_fpr20.pt")
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--split-scheme", choices=("iid", "geographic"), default="iid")
    parser.add_argument("--max-clips", type=int)
    parser.add_argument("--sampling-seed", type=int, default=42)
    parser.add_argument("--device", default="0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--target-fps", type=float, default=7.5)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--feature-width", type=int, help="Separate source-downsampled feature width; omitted preserves legacy resolution")
    parser.add_argument("--source-clock", action="store_true", help="Record actual sampled source-frame times for event decisions")
    parser.add_argument("--protocol", type=Path, help="Frozen train-only event calibration protocol")
    parser.add_argument("--cohort", choices=("calibration", "test"), default="test")
    parser.add_argument("--threshold-selection", type=Path, help="Frozen train-only threshold selection required before protocol test replay")
    parser.add_argument("--image-size", type=int, default=320)
    parser.add_argument("--confidence", type=float, default=0.15)
    parser.add_argument("--sequence-steps", type=int, default=15)
    parser.add_argument("--hop-steps", type=int, default=15)
    parser.add_argument("--min-positive-windows", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-dir", default="outputs/accident/event_batch_iid")
    args = parser.parse_args()
    if args.max_clips is not None and args.max_clips <= 0:
        raise SystemExit("--max-clips must be positive")
    if args.threads <= 0:
        raise SystemExit("--threads must be positive")
    if args.cohort == "calibration" and args.protocol is None:
        parser.error("A frozen --protocol is required for calibration clips")
    if args.protocol is not None and args.max_clips is not None:
        parser.error("Frozen protocol cohorts cannot be subsampled with --max-clips")
    if args.protocol is not None and args.cohort == "test" and args.threshold_selection is None:
        parser.error("Protocol test replay requires a frozen --threshold-selection")

    import cv2
    import torch
    from ultralytics import YOLO

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)

    metadata_path = _resolve(args.metadata)
    dataset_root = _resolve(args.dataset_root)
    model_path = _resolve(args.model)
    detector_path = _resolve(args.detector)
    output_dir = _resolve(args.output_dir)
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    metadata_clips = load_accident_manifest(metadata_path)
    protocol_hash = None
    selection_hash = None
    inference_keys = ("target_fps", "target_width", "feature_width", "image_size", "confidence", "sequence_steps", "hop_steps", "source_clock", "min_positive_windows")
    if args.protocol is not None:
        protocol_path = _resolve(str(args.protocol))
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if any(getattr(args, key) != value for key, value in protocol["inference"].items()):
            parser.error("Inference arguments must match the frozen protocol")
        if hashlib.sha256(metadata_path.read_bytes()).hexdigest() != protocol["metadata_sha256"]:
            raise ValueError("Metadata bytes differ from the calibration protocol")
        clips = protocol_clips(protocol, metadata_clips, args.split_scheme, args.cohort, model_path)
        protocol_hash = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
        if args.threshold_selection is not None:
            selection_path = _resolve(str(args.threshold_selection))
            selection = json.loads(selection_path.read_text(encoding="utf-8"))
            if selection["protocol_sha256"] != protocol_hash or selection["split_scheme"] != args.split_scheme:
                raise ValueError("Threshold selection does not match this protocol and split")
            selection_hash = hashlib.sha256(selection_path.read_bytes()).hexdigest()
    else:
        clips = select_evaluation_clips(
            metadata_clips,
            split_scheme=args.split_scheme,
            max_clips=args.max_clips,
            seed=args.sampling_seed,
        )
    implementation = {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__).resolve(), *(SRC / "illegal_parking" / name for name in ("incident_video_pipeline.py", "incident_candidate_roi.py", "incident_features.py", "incident_ultralytics.py", "incident_tcn_inference.py"))]}
    inference_signature = hashlib.sha256(json.dumps({"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(), "detector_sha256": hashlib.sha256(detector_path.read_bytes()).hexdigest(), "implementation_sha256": implementation, "protocol_sha256": protocol_hash, "threshold_selection_sha256": selection_hash, "cohort": args.cohort, **{key: getattr(args, key) for key in inference_keys}}, sort_keys=True).encode()).hexdigest()
    detector = YOLO(str(detector_path))
    predictor_device = "cuda" if args.device not in {"cpu", "-1"} else "cpu"
    predictor = IncidentTcnPredictor.from_checkpoint(model_path, predictor_device)

    started = time.perf_counter()
    started_utc = datetime.now(timezone.utc).isoformat()
    completed = 0
    skipped = 0
    failures: list[dict[str, str]] = []
    report_paths: list[str] = []
    for index, clip in enumerate(clips, start=1):
        report_path = reports_dir / report_filename(clip.relative_path)
        report_paths.append(str(report_path))
        if report_path.is_file() and not args.overwrite:
            if protocol_hash and json.loads(report_path.read_text(encoding="utf-8")).get("batch_inference_signature") != inference_signature:
                raise ValueError(f"Existing report has a different protocol: {report_path}; use a new output directory")
            skipped += 1
            print(f"[{index}/{len(clips)}] resume {clip.relative_path.as_posix()}", flush=True)
            continue
        try:
            result = run_candidate_incident_inference(
                dataset_root / clip.relative_path,
                detector,
                predictor,
                detector_path=detector_path,
                model_path=model_path,
                device=args.device,
                target_fps=args.target_fps,
                target_width=args.target_width,
                image_size=args.image_size,
                confidence=args.confidence,
                sequence_steps=args.sequence_steps,
                hop_steps=args.hop_steps,
                min_positive_windows=args.min_positive_windows,
                feature_width=args.feature_width,
                source_clock=args.source_clock,
            )
            result.report["batch_inference_signature"] = inference_signature
            report_path.write_text(
                json.dumps(result.report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            completed += 1
            print(f"[{index}/{len(clips)}] done {clip.relative_path.as_posix()}", flush=True)
        except (RuntimeError, ValueError, OSError) as exc:
            failures.append({"path": clip.relative_path.as_posix(), "error": str(exc)})
            print(f"[{index}/{len(clips)}] failed {clip.relative_path.as_posix()}: {exc}", flush=True)

    elapsed_sec = time.perf_counter() - started
    summary = {
        "split_scheme": args.split_scheme,
        "cohort": args.cohort,
        "protocol_sha256": protocol_hash,
        "threshold_selection_sha256": selection_hash,
        "started_at_utc": started_utc,
        "implementation_sha256": implementation,
        "batch_inference_signature": inference_signature,
        "sampling_seed": args.sampling_seed,
        "threads": args.threads,
        "requested_clips": len(clips),
        "completed_clips": completed,
        "resumed_clips": skipped,
        "failure_count": len(failures),
        "failures": failures,
        "elapsed_sec": elapsed_sec,
        "clips_per_minute": completed / elapsed_sec * 60.0 if elapsed_sec else None,
        "model": str(model_path),
        "detector": str(detector_path),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "detector_sha256": hashlib.sha256(detector_path.read_bytes()).hexdigest(),
        "metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        "inference_protocol": {key: getattr(args, key) for key in inference_keys},
        "report_paths": report_paths,
    }
    (output_dir / "batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if failures else 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
