import json
import sys

import numpy as np
import pytest

from illegal_parking.incident_presentation_events import presentation_operating_summary
from scripts import run_presentation_event_reporting as adapter
from scripts.evaluate_presentation_events import select_models


def sample():
    report = {"path": "clip", "clock": "normalized_presentation_time", "annotation_free_inference": True,
              "window_starts": [0, 15, 30, 45, 60], "window_end_times_sec": [2, 4, 6, 8, 10],
              "window_probabilities": [.9, .9, .1, .9, .9], "rejected_window_starts": []}
    annotation = {"path": "clip", "event_time_sec": 7, "collision_type": "rear-end"}
    return report, annotation


def test_original_numpy_episode_counts_reproduce_failure_and_roundtrip_exactly():
    report, annotation = sample()
    summary = presentation_operating_summary([report], {"clip": annotation}, .5)
    assert summary["early_alert_episodes"] == 1
    assert isinstance(summary["early_alert_episodes"], np.integer)
    with pytest.raises(TypeError, match="not JSON serializable"):
        json.dumps(summary)
    result = json.loads(adapter.ScalarJSON.dumps(summary))
    assert result == summary
    assert type(result["early_alert_episodes"]) is int
    assert type(result["off_target_episodes_per_clip"]) is float


@pytest.mark.parametrize("value", [float("nan"), float("inf"), np.float32("nan"), np.float32("inf"), np.array([1])])
def test_nonfinite_or_array_values_are_not_silently_coerced(value):
    with pytest.raises((ValueError, TypeError)):
        adapter.ScalarJSON.dumps({"value": value})


def test_complete_selected_curve_roundtrips_without_numeric_or_selection_changes():
    report, annotation = sample()
    report["scores"] = {"model": report.pop("window_probabilities")}
    protocol = {"models": {"model": {"split": "iid", "window_threshold": .55}},
                "cohorts": {"iid": {"calibration_clip_paths": ["clip"]}}, "annotations": {"clip": annotation},
                "calibration": {"threshold_grid": [0, .5, 1], "episode_budget_per_clip": 1}}
    original = select_models(protocol, [report])
    decoded = json.loads(adapter.ScalarJSON.dumps(original))
    assert decoded == original and decoded["model"]["selected"]["threshold"] == .55


def test_adapter_is_reporter_local_and_restores_json_even_on_failure(monkeypatch):
    original = adapter.frozen.json
    monkeypatch.setattr(adapter, "validate_amendment", lambda: {})
    monkeypatch.setattr(sys, "argv", ["adapter", "calibrate"])

    def failing_main():
        assert adapter.frozen.json is adapter.ScalarJSON
        assert json.dumps(1) == "1"
        with pytest.raises(TypeError):
            json.dumps(np.int64(1))
        raise RuntimeError("fixture")

    monkeypatch.setattr(adapter.frozen, "main", failing_main)
    with pytest.raises(RuntimeError, match="fixture"):
        adapter.main()
    assert adapter.frozen.json is original
