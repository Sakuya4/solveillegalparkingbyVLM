from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application" / "DebugVersion" / "src"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze one video for the native traffic research workbench.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--provider", choices=("CPUExecutionProvider", "QNNExecutionProvider"), default="CPUExecutionProvider")
    parser.add_argument("--model", type=Path, default=ROOT / "Application/desktop/models/incident_tcn.onnx")
    parser.add_argument("--metadata", type=Path, default=ROOT / "Application/desktop/models/incident_tcn.json")
    parser.add_argument("--detector", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--qnn-backend")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-duration-sec", type=float, default=120)
    args = parser.parse_args()
    if args.threads <= 0 or args.max_duration_sec <= 0:
        parser.error("Thread count and duration limit must be positive")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from illegal_parking.incident_video_pipeline import run_candidate_incident_inference
    from illegal_parking.onnx_runtime import OnnxIncidentPredictor
    from run_candidate_incident_video import _write_annotated_video

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(args.threads)
    capture = cv2.VideoCapture(str(args.input))
    fps = capture.get(cv2.CAP_PROP_FPS)
    frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
    opened = capture.isOpened()
    capture.release()
    if not opened or fps <= 0 or frame_count <= 0:
        raise ValueError("Input must be a readable video with a known duration")
    if frame_count / fps > args.max_duration_sec:
        raise ValueError(f"Select a clip of at most {args.max_duration_sec:g} seconds for interactive analysis")
    if not args.detector.is_file():
        raise FileNotFoundError(f"Detector checkpoint required: {args.detector}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    print("Loading calibrated ONNX model and vehicle tracker", flush=True)
    started = time.perf_counter()
    predictor = OnnxIncidentPredictor(args.model, args.metadata, provider=args.provider, backend_path=args.qnn_backend, threads=args.threads)
    result = run_candidate_incident_inference(
        args.input, YOLO(str(args.detector)), predictor,
        detector_path=args.detector, model_path=args.model, device="cpu",
        target_fps=7.5, target_width=640, image_size=320, confidence=0.15,
        sequence_steps=15, hop_steps=15, min_positive_windows=2,
    )
    print("Rendering privacy-treated model output", flush=True)
    video_path = args.output_dir / "incident_output.mp4"
    _write_annotated_video(video_path, result.frames, result.source_indices, result.tracked, result.steps, result.frame_probabilities, result.frame_reviews, predictor.threshold, result.report["sampled_fps"])
    flags = result.report["window_review_flags"]
    events = []
    active = False
    for index, flag in enumerate(flags):
        if flag and not active:
            start = result.report["window_starts"][index]
            decision_frame = min(start + result.report["sequence_steps"], len(result.source_indices) - 1)
            events.append({"window_index": index, "trigger_time_sec": result.source_indices[decision_frame] / result.report["source_fps"], "probability": result.report["window_probabilities"][index]})
        active = flag
    report = {
        **result.report, "runtime": predictor.runtime, "events": events,
        "elapsed_sec": time.perf_counter() - started,
        "output_video": str(video_path.resolve()),
        "model_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "detector_sha256": hashlib.sha256(args.detector.read_bytes()).hexdigest(),
        "software": {"python": sys.version.split()[0], "torch": torch.__version__, "opencv": cv2.__version__, "ultralytics": ultralytics.__version__},
        "detector_device": "cpu", "detector_role": "vehicle proposal and tracking",
        "temporal_model_role": "calibrated incident probability and persistence gate",
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {len(events)} review events in {report['elapsed_sec']:.2f} s | report.json", flush=True)


if __name__ == "__main__":
    main()
