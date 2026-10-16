import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest


def test_real_cpu_streaming_matches_batch_reference_and_defers_post_context(tmp_path):
    root = Path(__file__).resolve().parents[1]
    if not (root / "yolov8n.pt").is_file():
        pytest.skip("Local detector checkpoint required for full streaming parity")
    reports = []
    for script, extra in (("analyze_desktop_video.py", []), ("run_streaming_incident.py", ["--duration-sec", "60"])):
        output = tmp_path / script.removesuffix(".py")
        result = subprocess.run([sys.executable, f"scripts/{script}", "--input", "docs/assets/incident_input.mp4", "--output-dir", str(output), *extra], cwd=root, capture_output=True, timeout=180)
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        reports.append(json.loads((output / "report.json").read_text(encoding="utf-8")))
    batch, stream = reports
    assert batch["sampled_frames"] == stream["sampled_frames"]
    assert batch["window_starts"] == stream["window_starts"]
    np.testing.assert_allclose(batch["window_probabilities"], stream["window_probabilities"], atol=1e-5, rtol=0)
    assert batch["window_review_flags"] == stream["window_review_flags"]
    assert [event["input_frame_index"] for event in batch["events"]] == [event["input_frame_index"] for event in stream["events"]]
    assert stream["model_sha256"] == batch["model_sha256"]
    assert stream["buffers"] == {"feature_steps": 15, "context_frames": 16, "camera_mailbox_frames": 1}
    for event in stream["events"]:
        evidence = json.loads(Path(event["evidence"]["evidence_json"]).read_text(encoding="utf-8"))
        assert evidence["timestamps_sec"][1] == event["trigger_time_sec"]
        assert event["decision_available_wall_sec"] < event["evidence_available_wall_sec"]
