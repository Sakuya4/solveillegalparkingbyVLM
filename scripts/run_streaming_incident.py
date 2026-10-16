from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))


def main():
    worker_started = time.perf_counter()
    parser = argparse.ArgumentParser(description="Run bounded-memory causal incident inference on a video or persistent camera stream.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--source-mode", choices=("file", "page", "mjpeg", "snapshot"), default="file")
    parser.add_argument("--duration-sec", type=float, default=60)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path)
    parser.add_argument("--feature-width", type=int, help="Optional training-aligned width; default preserves the reference event protocol")
    parser.add_argument("--min-positive-windows", type=int, choices=(1, 2), default=2)
    parser.add_argument("--provider", choices=("CPUExecutionProvider", "QNNExecutionProvider"), default="CPUExecutionProvider")
    parser.add_argument("--qnn-backend")
    parser.add_argument("--detector", type=Path, default=ROOT / "yolov8n.pt")
    args = parser.parse_args()
    if not math.isfinite(args.duration_sec) or args.duration_sec <= 0 or (args.feature_width is not None and args.feature_width <= 0):
        parser.error("Duration and optional feature width must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = "2"
    import cv2
    import numpy as np
    import torch
    from ultralytics import YOLO
    from illegal_parking.cctv_capture import redacted_camera_url, _write_image
    from illegal_parking.frame_sources import MJPEGFrameSource, HTTPImageSource
    from illegal_parking.live_cctv import resolve_first_media_url
    from illegal_parking.incident_streaming import StreamingIncidentProcessor, LatestFrameReader
    from illegal_parking.onnx_runtime import OnnxIncidentPredictor
    from illegal_parking.incident_vlm_evidence import write_clean_review_package
    from illegal_parking.observation_review import MAX_OBSERVATION_GAP_SEC
    from run_candidate_incident_video import _blur_plate_region, _pixel_bbox

    cv2.setNumThreads(2)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    suffix = ".qdq.onnx" if args.provider == "QNNExecutionProvider" else ".onnx"
    model = ROOT / "Application/desktop/models" / ("incident_tcn" + suffix)
    predictor = OnnxIncidentPredictor(model, model.with_suffix(".json"), provider=args.provider, backend_path=args.qnn_backend, threads=2)
    detector = YOLO(str(args.detector))
    processor = StreamingIncidentProcessor(detector, predictor, feature_width=args.feature_width, min_positive_windows=args.min_positive_windows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    camera = args.source_mode != "file"
    reader = capture = writer = clean_writer = None
    context = deque(maxlen=16)
    pending, events, windows, frame_times, processing_times = [], [], [], [], []
    context_gaps = []
    source = redacted_camera_url(args.input) if camera else str(Path(args.input).resolve())
    started_utc = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()
    next_sample = next_preview = 0.0
    decoded = processed = 0
    cancelled = False
    end_reason, source_error = "duration_limit", None
    output_video = args.output_dir / "incident_output.mp4"
    clean_video = args.output_dir / "observation_clean.mp4"
    privacy = "Heuristic lower-center blur inside each tracked vehicle box"

    def publish_json(path, payload):
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def finish_evidence(item, after, after_time, truncated):
        event, before, trigger = item
        event["evidence"] = write_clean_review_package((before, trigger, after), (event.pop("before_time_sec"), event["trigger_time_sec"], after_time), args.output_dir / f"event_{event['event_id']:03d}", privacy_method=privacy)
        event["evidence_available_wall_sec"] = time.perf_counter() - started
        event["post_context_truncated"] = truncated

    try:
        if camera:
            media = resolve_first_media_url(args.input, timeout_sec=5) if args.source_mode == "page" else args.input
            if not media:
                raise ValueError("No camera endpoint found")
            redacted_camera_url(media)
            transport = HTTPImageSource(media, max_frames=None, timeout_sec=5, poll_interval_sec=0.5) if args.source_mode == "snapshot" else MJPEGFrameSource(media, timeout_sec=5)
            reader = LatestFrameReader(transport, max_retries=3)
        else:
            capture = cv2.VideoCapture(args.input)
            fps = float(capture.get(cv2.CAP_PROP_FPS))
            if not capture.isOpened() or not math.isfinite(fps) or fps <= 0:
                raise ValueError("Streaming replay requires a readable video with known FPS")
        with (args.output_dir / "decisions.jsonl").open("w", encoding="utf-8") as decision_log:
            while True:
                if args.stop_file and args.stop_file.exists():
                    cancelled = True
                    end_reason = "cancelled"
                    break
                if camera:
                    if time.perf_counter() - reader.started >= args.duration_sec:
                        break
                    try:
                        source_index, timestamp, frame = reader.read(timeout_sec=0.5)
                    except TimeoutError:
                        continue
                    except ConnectionError as exc:
                        end_reason, source_error = "source_error", str(exc)
                        break
                else:
                    ok, frame = capture.read()
                    if not ok:
                        end_reason = "source_end"
                        break
                    source_index, timestamp = decoded, decoded / fps
                    decoded += 1
                if timestamp >= args.duration_sec:
                    break
                if timestamp + 1e-9 < next_sample:
                    continue
                next_sample += (math.floor(max(0, timestamp - next_sample) * 7.5) + 1) / 7.5
                if frame_times and timestamp - frame_times[-1] > MAX_OBSERVATION_GAP_SEC:
                    for item in pending:
                        finish_evidence(item, context[-1][0], context[-1][1], True)
                    pending.clear()
                    context.clear()
                    processor.reset_temporal_context()
                    context_gaps.append({"before_sec": frame_times[-1], "after_sec": timestamp, "duration_sec": timestamp - frame_times[-1], "temporal_context_reset": True})
                update = processor.process(frame, source_index, timestamp)
                available = time.perf_counter() - started
                height, width = frame.shape[:2]
                clean = cv2.resize(frame, (640, max(1, round(height * 640 / width))), interpolation=cv2.INTER_AREA)
                for track in update.tracks:
                    _blur_plate_region(clean, track)
                context.append((clean, timestamp))
                frame_times.append(timestamp)
                processing_times.append(update.processing_sec)
                if update.window_start is not None:
                    window = {"window_start": update.window_start, "source_index": source_index, "source_time_sec": timestamp, "available_wall_sec": available, "probability": update.probability, "review_active": update.review_active}
                    if camera:
                        window["arrival_to_decision_sec"] = time.perf_counter() - reader.started - timestamp
                    windows.append(window)
                    decision_log.write(json.dumps(window) + "\n")
                    decision_log.flush()
                if update.new_event:
                    event = {"event_id": len(events) + 1, "trigger_time_sec": timestamp, "decision_available_wall_sec": available, "input_frame_index": source_index, "output_frame_index": processed, "probability": update.probability, "before_time_sec": context[0][1]}
                    events.append(event)
                    pending.append((event, context[0][0], clean.copy()))
                for item in list(pending):
                    if processed - item[0]["output_frame_index"] >= 15:
                        finish_evidence(item, clean, timestamp, False)
                        pending.remove(item)
                display = clean.copy()
                if update.step is not None and update.step.selected is not None:
                    box = _pixel_bbox(update.step.selected.bbox, display.shape[1], display.shape[0])
                    cv2.rectangle(display, box[:2], box[2:], (30, 50, 235) if update.review_active else (0, 210, 255), 2)
                label = "WARMUP" if update.probability is None else ("REVIEW" if update.review_active else "MONITOR")
                score = "--" if update.probability is None else f"{update.probability:.3f}"
                cv2.rectangle(display, (0, 0), (640, 52), (18, 18, 18), -1)
                cv2.putText(display, f"STREAM  {timestamp:.2f}s  incident={score}  {label}", (12, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
                if writer is None:
                    writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (display.shape[1], display.shape[0]))
                    clean_writer = cv2.VideoWriter(str(clean_video), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (clean.shape[1], clean.shape[0]))
                    if not writer.isOpened() or not clean_writer.isOpened():
                        raise OSError("Could not create streaming playback")
                writer.write(display)
                clean_writer.write(clean)
                processed += 1
                if update.new_event or time.perf_counter() - started >= next_preview:
                    temporary = args.output_dir / "preview_pending.jpg"
                    _write_image(temporary, display)
                    try:
                        temporary.replace(args.output_dir / "preview.jpg")
                    except PermissionError:
                        temporary.unlink(missing_ok=True)
                    for event in events:
                        event.setdefault("notification_published_wall_sec", time.perf_counter() - started)
                    publish_json(args.output_dir / "progress.json", {"sampled_frames": processed, "source_time_sec": timestamp, "window_count": len(windows), "events": events})
                    print(json.dumps({"stage": "streaming", "frames": processed, "windows": len(windows), "events": len(events), "source_time_sec": timestamp}), flush=True)
                    next_preview = time.perf_counter() - started + 0.5
        for item in pending:
            finish_evidence(item, context[-1][0], context[-1][1], True)
    finally:
        processor.close()
        if reader is not None:
            reader.close()
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if clean_writer is not None:
            clean_writer.release()
    if not processed:
        raise ValueError("No frames were processed")
    report = {"input": source, "output_video": str(output_video.resolve()), "source_mode": args.source_mode,
              "initialization_before_observation_sec": started - worker_started, "worker_elapsed_sec": time.perf_counter() - worker_started,
              "requested_duration_sec": args.duration_sec, "context_gap_limit_sec": MAX_OBSERVATION_GAP_SEC, "context_gaps": context_gaps,
              "clean_video": str(clean_video.resolve()), "clean_video_sha256": hashlib.sha256(clean_video.read_bytes()).hexdigest(),
              "clean_video_contract": "Privacy-treated sampled frames without model boxes, probabilities, event state, or artificial timestamp overlays",
              "annotation_free_inference": True, "streaming_inference": True, "started_at_utc": started_utc,
              "elapsed_sec": time.perf_counter() - started, "cancelled": cancelled, "end_reason": end_reason, "source_error": source_error, "events": events,
              "runtime": predictor.runtime, "min_positive_windows": args.min_positive_windows,
              "threshold": predictor.threshold, "window_starts": [item["window_start"] for item in windows],
              "window_probabilities": [item["probability"] for item in windows], "window_review_flags": [item["review_active"] for item in windows],
              "sequence_steps": 15, "hop_steps": 15, "sampled_frames": processed, "frame_times_sec": frame_times,
              "feature_width_limit": args.feature_width, "display_width": 640,
              "event_clock": "decoder observation clock" if camera else "source video frame clock",
              "playback_clock": "7.5 FPS presentation; actual frame_times_sec determine event timing",
              "buffers": {"feature_steps": 15, "context_frames": 16, "camera_mailbox_frames": 1},
              "processing_sec": {"median": float(np.median(processing_times)), "p95": float(np.percentile(processing_times, 95))},
              "model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "detector_sha256": hashlib.sha256(args.detector.read_bytes()).hexdigest(),
              "input_sha256": hashlib.sha256(args.input.encode() if camera else Path(args.input).read_bytes()).hexdigest(),
              "decisions_sha256": hashlib.sha256((args.output_dir / "decisions.jsonl").read_bytes()).hexdigest(),
              "privacy": privacy, "vlm_requested": False, "review_architecture": "Immediate event notification; post-event evidence export is deferred until context arrives, independently of subsequent VLM review"}
    if reader is not None:
        report["acquisition"] = {"decoded_frames": reader.decoded_frames, "mailbox_overwrites": reader.overwritten_frames,
                                 "read_failures": reader.read_failures, "retry_attempts": reader.retry_attempts,
                                 "max_consecutive_retries": reader.max_retries,
                                 "connections": transport.get_metadata().get("connections"),
                                 "clock": "Local decoder observation, not a source-camera sensor timestamp"}
        arrivals = [item["arrival_to_decision_sec"] for item in windows]
        report["arrival_to_decision_sec"] = {"median": float(np.median(arrivals)) if arrivals else None, "p95": float(np.percentile(arrivals, 95)) if arrivals else None}
    publish_json(args.output_dir / "report.json", report)
    print(json.dumps({"stage": "complete", "frames": processed, "events": len(events), "elapsed_sec": report["elapsed_sec"]}), flush=True)


if __name__ == "__main__":
    main()
