import cv2
import hashlib
import json
import numpy as np
import pytest

from illegal_parking.cctv_capture import capture_camera_observation, load_camera_observation, redacted_camera_url


class Clock:
    now = 0.0

    def read(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class Source:
    def __init__(self, clock):
        self.clock = clock
        self.released = False

    def read(self):
        self.clock.now += 0.1
        return True, np.full((24, 32, 3), 80, dtype=np.uint8)

    def release(self):
        self.released = True

    def get_metadata(self):
        return {"source_type": "mjpeg_stream", "connections": 1, "source": "secret"}


def test_capture_retains_timestamps_and_streams_scratch_to_video(tmp_path):
    clock = Clock()
    source = Source(clock)
    output = tmp_path / "中文目錄" / "capture.mp4"
    report = capture_camera_observation(source, output, source_url="https://camera.test/feed", transport="mjpeg", duration_sec=2, target_fps=7.5, target_width=32, min_frames=2, clock=clock.read, sleep=clock.sleep)
    assert source.released
    assert report["source_metadata"] == {"source_type": "mjpeg_stream", "connections": 1}
    assert report["successful_frames"] < report["decoded_frames"]
    assert report["frame_times_sec"] == sorted(report["frame_times_sec"])
    assert report["exact_duplicate_rate"] > 0.8
    capture = cv2.VideoCapture(str(output))
    assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == report["successful_frames"]
    capture.release()
    assert not list(tmp_path.rglob("camera_frames_*"))


def test_capture_cancellation_releases_source_and_scratch(tmp_path):
    clock = Clock()
    source = Source(clock)
    stop = tmp_path / "stop"
    stop.touch()
    with pytest.raises(InterruptedError):
        capture_camera_observation(source, tmp_path / "capture.mp4", source_url="https://camera.test/feed", transport="mjpeg", duration_sec=2, stop_file=stop, clock=clock.read, sleep=clock.sleep)
    assert source.released
    assert not list(tmp_path.glob("camera_frames_*"))


def test_camera_provenance_redacts_credentials_and_tokens():
    assert redacted_camera_url("https://user:password@camera.test:8443/feed?id=C01&token=secret#fragment") == "https://camera.test:8443/feed?id=C01&token=REDACTED"
    with pytest.raises(ValueError):
        redacted_camera_url("file:///private/camera.jpg")


def test_camera_report_requires_matching_video_and_monotone_clock(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"test fixture")
    report = {"output_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "successful_frames": 3, "frame_times_sec": [0.2, 0.4, 0.8], "effective_fps": 2 / 0.6, "exact_duplicate_rate": 0.0}
    path = tmp_path / "capture.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert load_camera_observation(path, video, 3) == report
    with pytest.raises(ValueError, match="frame count"):
        load_camera_observation(path, video, 4)
    report["frame_times_sec"] = [0.2, 0.2, 0.8]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="increasing"):
        load_camera_observation(path, video, 3)
