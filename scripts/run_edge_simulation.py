from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.detectors import DetectionRecord, StaticDetector, UltralyticsYoloDetector
from illegal_parking.event_models import BBox
from illegal_parking.frame_sources import (
    EdgeProfile,
    HTTPImageSource,
    ImageFolderSource,
    NetworkStreamSource,
    VideoFileSource,
    WebcamSource,
)
from illegal_parking.live_cctv import resolve_first_media_url
from illegal_parking.pipeline import EventPipeline
from illegal_parking.scene import RedlineSceneAnalyzer, StaticSceneAnalyzer
from illegal_parking.simulation import run_frame_source
from illegal_parking.violation_engine import ViolationEngine, ViolationEngineConfig


def build_source(
    source_type: str,
    path: str | None,
    camera_index: int,
    max_frames: int | None,
    read_timeout_sec: float,
):
    if source_type == "image-folder":
        if not path:
            raise ValueError("--path is required for image-folder source")
        return ImageFolderSource(path)
    if source_type == "video":
        if not path:
            raise ValueError("--path is required for video source")
        return VideoFileSource(path)
    if source_type == "webcam":
        return WebcamSource(camera_index)
    if source_type == "stream-url":
        if not path:
            raise ValueError("--path is required for stream-url source")
        return NetworkStreamSource(path)
    if source_type == "image-url":
        if not path:
            raise ValueError("--path is required for image-url source")
        return HTTPImageSource(path, max_frames=max_frames, timeout_sec=read_timeout_sec)
    if source_type == "cctv-page":
        if not path:
            raise ValueError("--path is required for cctv-page source")
        media_url = resolve_first_media_url(path, timeout_sec=read_timeout_sec)
        if not media_url:
            raise ValueError(f"No media URL found in CCTV page: {path}")
        return HTTPImageSource(media_url, camera_id="cctv_page", max_frames=max_frames, timeout_sec=read_timeout_sec)
    raise ValueError(f"Unsupported source type: {source_type}")


def build_detector(detector_name: str, weights: str, imgsz: int, conf: float):
    if detector_name == "none":
        return None
    if detector_name == "static-demo":
        return StaticDetector([
            DetectionRecord(
                bbox=BBox(20, 20, 160, 160),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ])
    if detector_name == "yolo":
        return UltralyticsYoloDetector(weights=weights, imgsz=imgsz, conf=conf)
    raise ValueError(f"Unsupported detector: {detector_name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an edge-source simulation without requiring camera hardware.")
    parser.add_argument("--source", choices=["image-folder", "video", "webcam", "stream-url", "image-url", "cctv-page"], required=True)
    parser.add_argument("--path", help="Folder, video path, stream URL, image URL, or CCTV page URL.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after this many frames, useful for live sources.")
    parser.add_argument("--read-timeout-sec", type=float, default=10.0, help="Network read timeout for live image/CCTV sources.")
    parser.add_argument("--profile", default="jetson_or_mini_pc_sim")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    parser.add_argument("--detector", choices=["none", "static-demo", "yolo"], default="none")
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--output-dir", default="outputs/edge_events")
    parser.add_argument("--dwell-threshold-sec", type=float, default=30.0)
    parser.add_argument("--min-stable-frames", type=int, default=3)
    parser.add_argument("--static-redline-ratio", type=float, default=None)
    args = parser.parse_args()

    source = build_source(args.source, args.path, args.camera_index, args.max_frames, args.read_timeout_sec)
    profile = EdgeProfile(name=args.profile, source_fps=args.source_fps, target_fps=args.target_fps)
    detector = build_detector(args.detector, args.weights, args.imgsz, args.conf)

    if detector is None:
        summary = run_frame_source(source, profile, max_frames=args.max_frames)
    else:
        if args.static_redline_ratio is None:
            scene_analyzer = RedlineSceneAnalyzer()
        else:
            scene_analyzer = StaticSceneAnalyzer(redline_overlap_ratio=args.static_redline_ratio)

        summary = EventPipeline(
            source=source,
            detector=detector,
            scene_analyzer=scene_analyzer,
            violation_engine=ViolationEngine(
                ViolationEngineConfig(
                    dwell_threshold_sec=args.dwell_threshold_sec,
                    min_stable_frames=args.min_stable_frames,
                    min_confidence=args.conf,
                )
            ),
            output_dir=args.output_dir,
            profile=profile,
            max_frames=args.max_frames,
        ).run()

    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
