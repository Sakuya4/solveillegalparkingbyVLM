from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from hardware_sim.python_golden.incident_review_workload import build_incident_review_workload, summarize_review_retention


def report():
    return {"annotation_free_inference": True, "sampled_frames": 12, "sampled_fps": 1, "window_starts": [0, 2, 4, 6, 8], "sequence_steps": 2, "threshold": 0.5, "window_probabilities": [0.8, 0.9, 0.1, 0.9, 0.1]}


def clip(name, accident=3):
    return SimpleNamespace(relative_path=Path(name), accident_time_sec=accident)


def test_episode_workload_resets_between_clips_and_labels_do_not_control_arrivals():
    data = report()
    flags, matching, count = build_incident_review_workload([(data, clip("a")), (data, clip("b", 100))], 1)
    assert np.flatnonzero(flags).tolist() == [2, 8, 14, 20]
    assert matching == {2: "a"} and count == 2
    conservative, matches, _ = build_incident_review_workload([(data, clip("a"))], 2)
    assert np.flatnonzero(conservative).tolist() == [4]
    assert matches == {4: "a"}


def test_retention_tracks_camera_identity_and_completed_service():
    raw = np.zeros((5, 2), dtype=bool)
    raw[2] = True
    admitted = raw.copy()
    admitted[2, 1] = False
    result = summarize_review_retention(raw, admitted, {2: "a"}, [{"source_index": 4, "status": "completed", "finish_ms": 3500}], clips=1, fps=1)
    assert result["detected_before_routing"] == 2
    assert result["admission_retention"] == 0.5
    assert result["completed_service_retention"] == 0.5
    assert result["completed_review_latency_sec"]["median"] == 1.5


def test_workload_rejects_duplicate_or_noncausal_inputs():
    data = report()
    with pytest.raises(ValueError, match="Duplicate"):
        build_incident_review_workload([(data, clip("a")), (data, clip("a"))], 1)
    data["window_starts"] = [0, 2, 2, 6, 8]
    with pytest.raises(ValueError, match="increasing"):
        build_incident_review_workload([(data, clip("a"))], 1)
