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
    parser.add_argument("--model", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--detector", type=Path, default=ROOT / "yolov8n.pt")
    parser.add_argument("--qnn-backend")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--max-duration-sec", type=float, default=120)
    parser.add_argument("--vlm-review", action="store_true")
    parser.add_argument("--max-vlm-events", type=int, default=3)
    parser.add_argument("--min-positive-windows", type=int, choices=(1, 2), default=2)
    parser.add_argument("--capture-report", type=Path)
    args = parser.parse_args()
    if args.threads <= 0 or args.max_duration_sec <= 0 or args.max_vlm_events <= 0:
        parser.error("Thread count, duration limit, and VLM event budget must be positive")
    if args.model is None:
        suffix = ".qdq.onnx" if args.provider == "QNNExecutionProvider" else ".onnx"
        args.model = ROOT / "Application/desktop/models" / ("incident_tcn" + suffix)
    if args.metadata is None:
        args.metadata = args.model.with_suffix(".json")
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO
    from illegal_parking.incident_video_pipeline import run_candidate_incident_inference
    from illegal_parking.onnx_runtime import OnnxIncidentPredictor
    from illegal_parking.incident_vlm_evidence import REVIEW_POLICY_VERSION, write_clean_review_package, review_disposition
    from illegal_parking.cctv_capture import load_camera_observation
    from run_candidate_incident_video import _write_annotated_video, _blur_plate_region

    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
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
    capture_metadata = load_camera_observation(args.capture_report, args.input, int(frame_count)) if args.capture_report else None

    def source_time(index):
        return capture_metadata["frame_times_sec"][index] if capture_metadata is not None else index / fps
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
        sequence_steps=15, hop_steps=15, min_positive_windows=args.min_positive_windows,
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
            events.append({"window_index": index, "trigger_time_sec": source_time(result.source_indices[decision_frame]), "output_frame_index": decision_frame, "input_frame_index": result.source_indices[decision_frame], "probability": result.report["window_probabilities"][index]})
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
        "event_clock": "captured observation timestamps" if capture_metadata is not None else "source video frame clock",
    }
    if capture_metadata is not None:
        report["capture"] = capture_metadata
    vlm_pipeline = None
    for index, event in enumerate(events):
        window_start = result.report["window_starts"][event["window_index"]]
        decision = min(window_start + 15, len(result.frames) - 1)
        frame_indices = (max(0, decision - 15), decision, min(len(result.frames) - 1, decision + 15))
        review_frames = []
        for frame_index in frame_indices:
            frame = result.frames[frame_index].copy()
            for track in result.tracked[result.source_indices[frame_index]]:
                _blur_plate_region(frame, track)
            review_frames.append(frame)
        times = tuple(source_time(result.source_indices[frame]) for frame in frame_indices)
        event["evidence"] = write_clean_review_package(tuple(review_frames), times, args.output_dir / f"event_{index + 1:03d}", privacy_method=result.report["privacy"])
        if args.vlm_review and index < args.max_vlm_events:
            from illegal_parking.vlm_review import VlmReviewRequest, parse_vlm_review_result_text, build_unparsed_vlm_review_result
            from illegal_parking.vlm_transformers import TransformersVlmConfig, run_transformers_image_text_to_text
            print(f"Reviewing event {index + 1} with Qwen2.5-VL-3B", flush=True)
            request = VlmReviewRequest.from_dict(json.loads(Path(event["evidence"]["request_json"]).read_text(encoding="utf-8")))
            config = TransformersVlmConfig("Qwen/Qwen2.5-VL-3B-Instruct", "qwen2_5_vl_3b", device=0 if torch.cuda.is_available() else "cpu")
            if vlm_pipeline is None:
                from transformers import pipeline
                vlm_pipeline = pipeline("image-text-to-text", model=config.model_id, device=config.device, dtype="auto")
            review_started = time.perf_counter()
            raw = run_transformers_image_text_to_text(request, config, pipeline_factory=lambda **kwargs: vlm_pipeline)
            try:
                review = parse_vlm_review_result_text(raw, provider=config.provider_name)
                schema_valid = True
            except (ValueError, json.JSONDecodeError) as exc:
                review = build_unparsed_vlm_review_result(raw, provider=config.provider_name, error=str(exc))
                schema_valid = False
            event["vlm_review"] = {
                **review.to_dict(), "schema_valid": schema_valid,
                "elapsed_sec": time.perf_counter() - review_started,
                "model_id": config.model_id,
                "model_revision": getattr(vlm_pipeline.model.config, "_commit_hash", None),
                "parameter_dtype": str(next(vlm_pipeline.model.parameters()).dtype),
                "do_sample": config.do_sample, "max_new_tokens": config.max_new_tokens,
                "seed": 42,
            }
            event["review_disposition"] = review_disposition(True, event["vlm_review"])
            event["review_disposition_policy_version"] = REVIEW_POLICY_VERSION
            Path(event["evidence"]["request_json"]).with_name("vlm_raw.txt").write_text(raw, encoding="utf-8")
    report["elapsed_sec"] = time.perf_counter() - started
    report["vlm_requested"] = args.vlm_review
    report["vlm_event_budget"] = args.max_vlm_events
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Completed: {len(events)} review events in {report['elapsed_sec']:.2f} s | report.json", flush=True)


if __name__ == "__main__":
    main()
