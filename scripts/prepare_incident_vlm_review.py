from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.incident_vlm_evidence import select_review_context
from illegal_parking.vlm_review import build_traffic_incident_review_request


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a privacy-safe temporal VLM review package from incident inference."
    )
    parser.add_argument("--inference-report", required=True)
    parser.add_argument("--annotated-video", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--blind-model-context", action="store_true")
    parser.add_argument("--request-filename", default="vlm_review_request.json")
    parser.add_argument(
        "--clean-video",
        help="Original video used to create overlay-free privacy-redacted evidence.",
    )
    parser.add_argument("--detector", default="yolov8n.pt")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    report = json.loads(Path(args.inference_report).read_text(encoding="utf-8"))
    if not report.get("privacy"):
        raise ValueError("Inference report must document privacy redaction before evidence export.")

    selection = select_review_context(report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = ("before.jpg", "trigger.jpg", "after.jpg")
    output_paths = tuple(output_dir / name for name in artifact_names)
    if args.clean_video:
        _extract_clean_privacy_frames(
            Path(args.clean_video),
            selection.frame_indices,
            float(report["target_fps"]),
            Path(args.detector),
            args.device,
            output_paths,
        )
    else:
        _extract_frames(Path(args.annotated_video), selection.frame_indices, output_paths)

    evidence = {
        "event_type": "traffic_incident",
        "model_probability": selection.model_probability,
        "decision_threshold": float(report.get("threshold", 0.5)),
        "consecutive_positive_windows": selection.consecutive_positive_windows,
        "required_consecutive_windows": int(report.get("min_positive_windows", 1)),
        "confirmed_window_index": selection.window_index,
        "frame_indices": list(selection.frame_indices),
        "privacy_redacted": True,
        "privacy_method": str(report["privacy"]),
        "annotation_free_inference": bool(report.get("annotation_free_inference", False)),
        "artifacts": {
            "before": artifact_names[0],
            "trigger": artifact_names[1],
            "after": artifact_names[2],
        },
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    request = build_traffic_incident_review_request(
        evidence,
        output_dir,
        include_model_context=not args.blind_model_context,
    )
    request_path = output_dir / args.request_filename
    request_path.write_text(
        json.dumps(request.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"evidence": str(evidence_path), "request": str(request_path)}, indent=2))
    return 0


def sampled_source_indices(
    source_frame_count: int, source_fps: float, target_fps: float
) -> list[int]:
    if source_frame_count <= 0 or source_fps <= 0 or target_fps <= 0:
        raise ValueError("frame count and FPS values must be positive")
    sample_period = 1.0 / min(target_fps, source_fps)
    next_sample_sec = 0.0
    indices = []
    for frame_index in range(source_frame_count):
        timestamp_sec = frame_index / source_fps
        if timestamp_sec + 1e-9 >= next_sample_sec:
            indices.append(frame_index)
            next_sample_sec += sample_period
    return indices


def _extract_clean_privacy_frames(
    video_path: Path,
    sampled_frame_indices: tuple[int, int, int],
    target_fps: float,
    detector_path: Path,
    device: str,
    output_paths: tuple[Path, Path, Path],
) -> None:
    from ultralytics import YOLO

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open clean video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    mapping = sampled_source_indices(frame_count, source_fps, target_fps)
    try:
        frames = []
        for sampled_index in sampled_frame_indices:
            if sampled_index >= len(mapping):
                raise ValueError(f"Sampled frame {sampled_index} is outside clean video")
            capture.set(cv2.CAP_PROP_POS_FRAMES, mapping[sampled_index])
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read source frame {mapping[sampled_index]}")
            frames.append(frame)
    finally:
        capture.release()

    detector = YOLO(str(detector_path))
    results = detector.predict(frames, device=device, imgsz=640, conf=0.15, verbose=False)
    for frame, result, output_path in zip(frames, results, output_paths):
        boxes = getattr(result, "boxes", None)
        if boxes is not None:
            for box, class_id in zip(boxes.xyxy.cpu().numpy(), boxes.cls.cpu().numpy()):
                if int(class_id) in {1, 2, 3, 5, 7}:
                    _blur_vehicle_plate_region(frame, box)
        if not cv2.imwrite(str(output_path), frame):
            raise ValueError(f"Could not write review frame: {output_path}")


def _blur_vehicle_plate_region(frame, box) -> None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = (int(value) for value in box)
    x1, x2 = max(0, x1), min(width, x2)
    y1, y2 = max(0, y1), min(height, y2)
    plate_top = y1 + round((y2 - y1) * 0.55)
    left = x1 + round((x2 - x1) * 0.2)
    right = x2 - round((x2 - x1) * 0.2)
    region = frame[plate_top:y2, left:right]
    if region.size:
        frame[plate_top:y2, left:right] = cv2.GaussianBlur(region, (17, 17), 0)


def _extract_frames(
    video_path: Path,
    frame_indices: tuple[int, int, int],
    output_paths: tuple[Path, Path, Path],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open annotated video: {video_path}")
    try:
        for frame_index, output_path in zip(frame_indices, output_paths):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {frame_index} from {video_path}")
            if not cv2.imwrite(str(output_path), frame):
                raise ValueError(f"Could not write review frame: {output_path}")
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
