from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import tempfile
import time
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import cv2
import numpy as np


def _write_image(path: Path, frame, parameters=()) -> None:
    ok, encoded = cv2.imencode(path.suffix, frame, list(parameters))
    if not ok:
        raise OSError(f"Could not encode camera image: {path.name}")
    path.write_bytes(encoded.tobytes())


def redacted_camera_url(url: str) -> str:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError("Camera source must be an HTTP or HTTPS URL")
    host = parsed.hostname
    if ":" in host:
        host = f"[{host}]"
    if parsed.port is not None:
        host += f":{parsed.port}"
    query = [(key, value if key.lower() in {"id", "camera_id"} else "REDACTED") for key, value in parse_qsl(parsed.query)]
    return urlunsplit((parsed.scheme, host, parsed.path, urlencode(query), ""))


def load_camera_observation(path: Path, video_path: Path, expected_frames: int) -> dict:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError("Camera report must be a JSON object")
    if report.get("output_video_sha256") != hashlib.sha256(video_path.read_bytes()).hexdigest():
        raise ValueError("Camera report does not match the selected video")
    times = np.asarray(report.get("frame_times_sec", []), dtype=float)
    if times.ndim != 1 or len(times) != expected_frames or report.get("successful_frames") != expected_frames:
        raise ValueError("Camera timestamps do not match the selected video frame count")
    if len(times) < 2 or not np.all(np.isfinite(times)) or times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Camera timestamps must be finite, nonnegative, and strictly increasing")
    try:
        fps = float(report.get("effective_fps", float("nan")))
        duplicates = float(report.get("exact_duplicate_rate", float("nan")))
    except (TypeError, ValueError) as exc:
        raise ValueError("Camera cadence and duplicate fraction must be numeric") from exc
    expected_fps = (len(times) - 1) / (times[-1] - times[0])
    if not math.isfinite(fps) or not np.isclose(fps, expected_fps, rtol=0.01) or not math.isfinite(duplicates) or not 0 <= duplicates <= 1:
        raise ValueError("Camera cadence or duplicate fraction is inconsistent with the observation")
    return report


def capture_camera_observation(
    source, output_video: Path, *, source_url: str, transport: str,
    duration_sec: float, target_width: int = 640, target_fps: float = 7.5,
    poll_interval_sec: float = 0.5, stop_file: Path | None = None,
    preview_path: Path | None = None, progress=None, min_frames: int = 16,
    clock=time.perf_counter, sleep=time.sleep,
) -> dict:
    if not all(math.isfinite(value) and value > 0 for value in (duration_sec, target_fps, poll_interval_sec)) or target_width <= 0 or min_frames < 2:
        raise ValueError("Duration, cadence, polling interval, and image width must be positive")
    public_source = redacted_camera_url(source_url)
    if transport not in {"snapshot", "mjpeg"}:
        raise ValueError("Unsupported camera transport")
    output_video = output_video.resolve()
    output_video.parent.mkdir(parents=True, exist_ok=True)
    times, hashes, paths = [], [], []
    failures = decoded = consecutive_failures = 0
    started_utc = datetime.now(timezone.utc).isoformat()
    started = clock()
    next_poll = next_sample = next_preview = next_progress = started
    with tempfile.TemporaryDirectory(prefix="camera_frames_", dir=output_video.parent) as temporary:
        temporary_path = Path(temporary).resolve()
        if not temporary_path.is_relative_to(output_video.parent):
            raise ValueError("Capture scratch directory escaped the output directory")
        try:
            while clock() - started < duration_sec:
                if stop_file is not None and stop_file.exists():
                    raise InterruptedError("Camera observation cancelled")
                if transport == "snapshot":
                    delay = min(next_poll - clock(), duration_sec - (clock() - started))
                    if delay > 0:
                        sleep(delay)
                    if clock() - started >= duration_sec:
                        break
                ok, frame = source.read()
                observed = clock()
                next_poll = observed + poll_interval_sec
                if observed - started >= duration_sec:
                    break
                if not ok or frame is None:
                    failures += 1
                    consecutive_failures += 1
                    if consecutive_failures >= 3:
                        raise ConnectionError("Camera source failed three consecutive reads")
                    sleep(min(1.0, max(0.0, duration_sec - (clock() - started))))
                    continue
                consecutive_failures = 0
                decoded += 1
                if observed + 1e-9 < next_sample:
                    continue
                next_sample += (math.floor(max(0, observed - next_sample) * target_fps) + 1) / target_fps
                height, width = frame.shape[:2]
                frame = cv2.resize(frame, (target_width, max(1, round(height * target_width / width))), interpolation=cv2.INTER_AREA)
                if paths and frame.shape != shape:
                    failures += 1
                    continue
                shape = frame.shape
                path = temporary_path / f"{len(paths):08d}.png"
                _write_image(path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
                paths.append(path)
                times.append(observed - started)
                hashes.append(hashlib.sha256(frame.tobytes()).hexdigest())
                if preview_path is not None and observed >= next_preview:
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    pending = preview_path.with_name("preview_pending.jpg")
                    _write_image(pending, frame)
                    try:
                        pending.replace(preview_path)
                    except PermissionError:
                        pending.unlink(missing_ok=True)
                    next_preview = observed + 0.5
                if progress is not None and observed >= next_progress:
                    progress({"stage": "capture", "frames": len(paths), "elapsed_sec": observed - started})
                    next_progress = observed + 2
        finally:
            source.release()
        elapsed = clock() - started
        if len(paths) < min_frames:
            raise ValueError(f"Captured {len(paths)} frames; at least {min_frames} are required")
        span = times[-1] - times[0]
        if span <= 0:
            raise ValueError("Camera timestamps must span positive wall-clock time")
        effective_fps = (len(paths) - 1) / span
        height, width = shape[:2]
        writer = cv2.VideoWriter(str(output_video), cv2.VideoWriter_fourcc(*"mp4v"), effective_fps, (width, height))
        if not writer.isOpened():
            raise OSError("Could not create captured video")
        try:
            for path in paths:
                if stop_file is not None and stop_file.exists():
                    raise InterruptedError("Camera observation cancelled")
                frame = cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise OSError("Could not decode a spooled camera frame")
                writer.write(frame)
        finally:
            writer.release()
    return {
        "source_page": public_source, "source_url_sha256": hashlib.sha256(source_url.encode()).hexdigest(),
        "transport": transport, "started_at_utc": started_utc,
        "requested_duration_sec": duration_sec, "observed_wall_duration_sec": elapsed,
        "valid_observation_span_sec": span,
        "successful_frames": len(paths), "decoded_frames": decoded, "failed_reads": failures,
        "target_fps": target_fps, "effective_fps": effective_fps,
        "unique_exact_frames": len(set(hashes)), "exact_duplicate_rate": 1 - len(set(hashes)) / len(hashes),
        "new_frame_transitions_per_sec": sum(a != b for a, b in zip(hashes, hashes[1:])) / span,
        "frame_times_sec": times, "frame_sha256": hashes,
        "source_metadata": {key: value for key, value in source.get_metadata().items() if key in {"source_type", "frame_index", "connections", "last_error"}},
        "output_video": output_video.name, "output_video_sha256": hashlib.sha256(output_video.read_bytes()).hexdigest(),
        "encoding": "Lossless PNG scratch frames; mp4v playback at measured mean cadence; original observation timestamps retained",
    }
