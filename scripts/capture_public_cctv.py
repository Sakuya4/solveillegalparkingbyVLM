from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.frame_sources import HTTPImageSource
from illegal_parking.live_cctv import resolve_first_media_url


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture a wall-clock-timed public CCTV image endpoint to video."
    )
    parser.add_argument("--page-url", required=True)
    parser.add_argument("--duration-sec", type=float, default=300.0)
    parser.add_argument("--poll-interval-sec", type=float, default=0.5)
    parser.add_argument("--target-width", type=int, default=640)
    parser.add_argument("--output-video", required=True)
    parser.add_argument("--output-report", required=True)
    args = parser.parse_args()
    if args.duration_sec <= 0 or args.poll_interval_sec <= 0 or args.target_width <= 0:
        raise SystemExit("duration, poll interval, and target width must be positive")

    media_url = resolve_first_media_url(args.page_url)
    if not media_url:
        raise SystemExit(f"No image endpoint found in {args.page_url}")
    source = HTTPImageSource(media_url, max_frames=None, timeout_sec=10.0)
    frames = []
    frame_times: list[float] = []
    hashes: list[str] = []
    failures = 0
    started = time.perf_counter()
    next_poll = started
    while time.perf_counter() - started < args.duration_sec:
        delay = next_poll - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        ok, frame = source.read()
        observed = time.perf_counter()
        next_poll = observed + args.poll_interval_sec
        if not ok or frame is None:
            failures += 1
            continue
        height, width = frame.shape[:2]
        if width != args.target_width:
            frame = cv2.resize(
                frame,
                (args.target_width, round(height * args.target_width / width)),
                interpolation=cv2.INTER_AREA,
            )
        frames.append(frame)
        frame_times.append(observed - started)
        hashes.append(hashlib.sha1(frame.tobytes()).hexdigest())

    elapsed_sec = time.perf_counter() - started
    if len(frames) < 16:
        raise SystemExit(f"Only captured {len(frames)} frames; at least 16 are required")
    effective_fps = (len(frames) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
    output_video = _resolve(args.output_video)
    output_video.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, (width, height)
    )
    if not writer.isOpened():
        raise SystemExit(f"Cannot create {output_video}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()

    report = {
        "source_page": args.page_url,
        "media_endpoint_redacted": True,
        "requested_duration_sec": args.duration_sec,
        "observed_wall_duration_sec": elapsed_sec,
        "successful_frames": len(frames),
        "failed_reads": failures,
        "effective_fps": effective_fps,
        "unique_exact_frames": len(set(hashes)),
        "exact_duplicate_rate": 1.0 - len(set(hashes)) / len(hashes),
        "output_video": output_video.name,
    }
    output_report = _resolve(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
