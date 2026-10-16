from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.cctv_capture import capture_camera_observation, redacted_camera_url
from illegal_parking.frame_sources import HTTPImageSource, MJPEGFrameSource
from illegal_parking.live_cctv import resolve_first_media_url


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture a timed public CCTV observation with source timestamps.")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--page-url")
    inputs.add_argument("--source-url")
    parser.add_argument("--transport", choices=("snapshot", "mjpeg"), default="snapshot")
    parser.add_argument("--duration-sec", type=float, default=300)
    parser.add_argument("--poll-interval-sec", type=float, default=0.5)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--target-fps", type=float, default=7.5)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--preview", type=Path)
    parser.add_argument("--stop-file", type=Path)
    args = parser.parse_args()
    source_url = args.page_url or args.source_url
    redacted_camera_url(source_url)
    media = resolve_first_media_url(args.page_url, timeout_sec=5) if args.page_url else args.source_url
    if not media:
        raise ValueError("No camera endpoint was found in the source page")
    redacted_camera_url(media)
    cv2.setNumThreads(2)
    source = MJPEGFrameSource(media, timeout_sec=5) if args.transport == "mjpeg" else HTTPImageSource(media, max_frames=None, timeout_sec=5)
    report = capture_camera_observation(source, args.output_video, source_url=source_url, transport=args.transport, duration_sec=args.duration_sec, poll_interval_sec=args.poll_interval_sec, target_width=args.target_width, target_fps=args.target_fps, stop_file=args.stop_file, preview_path=args.preview, progress=lambda row: print(json.dumps(row), flush=True))
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key not in {"frame_times_sec", "frame_sha256"}}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
