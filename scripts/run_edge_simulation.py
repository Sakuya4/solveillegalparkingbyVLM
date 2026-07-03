from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.frame_sources import EdgeProfile, ImageFolderSource, VideoFileSource, WebcamSource
from illegal_parking.simulation import run_frame_source


def build_source(source_type: str, path: str | None, camera_index: int):
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
    raise ValueError(f"Unsupported source type: {source_type}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an edge-source simulation without requiring camera hardware.")
    parser.add_argument("--source", choices=["image-folder", "video", "webcam"], required=True)
    parser.add_argument("--path", help="Folder or video path for simulated input.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--profile", default="jetson_or_mini_pc_sim")
    parser.add_argument("--source-fps", type=float, default=30.0)
    parser.add_argument("--target-fps", type=float, default=10.0)
    args = parser.parse_args()

    source = build_source(args.source, args.path, args.camera_index)
    profile = EdgeProfile(name=args.profile, source_fps=args.source_fps, target_fps=args.target_fps)
    summary = run_frame_source(source, profile)
    print(json.dumps(summary.to_dict(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
