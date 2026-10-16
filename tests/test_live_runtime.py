import copy
import importlib.util
import hashlib
import json
import statistics
from pathlib import Path

import pytest


def publisher():
    path = Path(__file__).resolve().parents[1] / "scripts/publish_live_runtime.py"
    spec = importlib.util.spec_from_file_location("live_publisher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    inference = dict(display_width=640, feature_width_limit=None, sequence_steps=15, hop_steps=15, min_positive_windows=2, threshold=.8)
    report = {**inference, "streaming_inference": True, "source_mode": "page", "event_clock": "decoder observation clock",
              "input": "https://camera.example/page", "requested_duration_sec": 120,
              "model_sha256": "model", "detector_sha256": "detector", "clean_video_sha256": "video",
              "frame_times_sec": [.1, .3, 2, 2.2], "sampled_frames": 4, "window_starts": [], "window_probabilities": [],
              "window_review_flags": [], "events": [], "context_gaps": [{"duration_sec": 1.7}], "started_at_utc": "time",
              "initialization_before_observation_sec": 5, "camera_model_warmup_sec": 2, "elapsed_sec": 2.3,
              "end_reason": "source_error", "source_error": "Connection closed", "acquisition": {},
              "processing_sec": {}, "arrival_to_decision_sec": {}, "runtime": {}}
    protocol = {"camera_page": report["input"], "requested_observation_sec": 120, "inference": inference,
                "reference_model_sha256": "model", "detector_sha256": "detector", "clock": "arrival", "media_policy": "local"}
    return report, protocol


def test_runtime_summary_preserves_failure_and_never_assigns_normal_exposure():
    report, protocol = fixture()
    result = publisher().runtime_summary(report, protocol, [])
    assert result["eligible_acquisition_sec"] == pytest.approx(.4)
    assert result["excluded_gap_sec"] == pytest.approx(1.7)
    assert result["end_reason"] == "source_error" and result["requested_observation_sec"] == 120
    assert result["normal_exposure_added_hours"] == 0 and result["observation_label"] == "UNASSIGNED"
    assert "accuracy" not in result and "clean_video" not in result


@pytest.mark.parametrize("change", ["clock", "request", "model", "frames", "decisions", "gaps"])
def test_runtime_publication_rejects_contract_or_trace_mismatch(change):
    report, protocol = fixture()
    report = copy.deepcopy(report)
    if change == "clock":
        report["event_clock"] = "source video frame clock"
    elif change == "request":
        report["requested_duration_sec"] = 30
    elif change == "model":
        report["model_sha256"] = "other"
    elif change == "frames":
        report["sampled_frames"] = 10
    elif change == "decisions":
        report["window_starts"] = [0]
    else:
        report["context_gaps"] = []
    with pytest.raises(ValueError):
        publisher().runtime_summary(report, protocol, [])


def test_published_camera_trace_reconstructs_timing_without_assigning_ground_truth():
    root = Path(__file__).resolve().parents[1]
    protocol_path = root / "docs/results/live_runtime_20261017_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    path = root / "docs/results/live_runtime_20261017.json"
    data = json.loads(path.read_text())
    assert data["protocol_sha256"] == hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    assert data["publisher_sha256"] == hashlib.sha256((root / "scripts/publish_live_runtime.py").read_bytes()).hexdigest()
    assert data["inference"] == protocol["inference"]
    assert data["model_sha256"] == protocol["reference_model_sha256"]
    times = data["frame_times_sec"]
    assert len(times) == data["sampled_frames"] == 884
    assert all(a < b for a, b in zip(times, times[1:]))
    intervals = [b - a for a, b in zip(times, times[1:])]
    assert data["eligible_acquisition_sec"] == pytest.approx(sum(value for value in intervals if value <= 1))
    assert data["excluded_gap_sec"] == pytest.approx(sum(value for value in intervals if value > 1))
    assert data["source_span_sec"] == pytest.approx(times[-1] - times[0])
    decisions = data["decisions"]
    assert len(decisions) == data["window_count"] == 58
    assert all(row["source_time_sec"] in times for row in decisions)
    latencies = [row["arrival_to_decision_sec"] for row in decisions]
    assert all(value >= 0 for value in latencies)
    assert data["arrival_to_decision_sec"]["median"] == statistics.median(latencies)
    assert data["observation_label"] == "UNASSIGNED" and data["normal_exposure_added_hours"] == 0
    assert "accuracy" not in data and "false_alarm_rate" not in data
    assert "clean_video" not in data and "C:\\Users" not in path.read_text()
    for name, expected in protocol["implementation_sha256"].items():
        assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected
