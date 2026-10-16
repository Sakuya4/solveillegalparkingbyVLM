import hashlib
import json
from pathlib import Path
import subprocess
import sys

import cv2
import pytest


def test_positive_reference_event_uses_observation_clock_without_changing_frame_location(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root / "yolov8n.pt").is_file():
        pytest.skip("The local detector checkpoint is required for full inference integration")
    video = root / "docs/assets/incident_input.mp4"
    capture = cv2.VideoCapture(str(video))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = capture.get(cv2.CAP_PROP_FPS)
    capture.release()
    offset = 10.0
    times = [offset + index / fps for index in range(count)]
    report = {"source_type": "synthetic_clock_offset_fixture", "output_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(), "successful_frames": count, "frame_times_sec": times, "effective_fps": fps, "exact_duplicate_rate": 0.0}
    source_report = tmp_path / "capture.json"
    source_report.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "analysis"
    result = subprocess.run([sys.executable, "scripts/analyze_desktop_video.py", "--input", str(video), "--capture-report", str(source_report), "--output-dir", str(output)], cwd=root, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    analysis = json.loads((output / "report.json").read_text(encoding="utf-8"))
    assert analysis["event_clock"] == "captured observation timestamps"
    assert analysis["events"]
    event = analysis["events"][0]
    assert event["trigger_time_sec"] == pytest.approx(times[event["input_frame_index"]])
    assert event["trigger_time_sec"] >= offset
    assert event["output_frame_index"] < analysis["sampled_frames"]
    evidence = json.loads(Path(event["evidence"]["evidence_json"]).read_text(encoding="utf-8"))
    assert evidence["timestamps_sec"][1] == pytest.approx(event["trigger_time_sec"])
