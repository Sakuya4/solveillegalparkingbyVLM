import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from scripts import calibrate_continuous_events as study


def test_calibration_never_loads_test_reports_and_frozen_selection_is_not_rewritten(monkeypatch, tmp_path):
    monkeypatch.setattr(study, "ROOT", tmp_path)
    protocol = {"models": {"iid": {"original_window_threshold": 0.5}}, "threshold_grid": [0.1, 0.9], "evaluated_hop_steps": [5], "persistence_gates": [1], "off_target_budgets_per_clip": [0.25]}
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    report = {"annotation_free_inference": True, "sampled_frames": 26, "sampled_fps": 5, "sequence_steps": 5, "window_starts": [0, 5, 10, 15, 20], "window_probabilities": [0.8, 0.9, 0.1, 0.2, 0.95], "window_end_times_sec": [1, 2, 3, 4, 5]}
    clip = SimpleNamespace(relative_path=Path("fixture.mp4"), collision_type="head-on", accident_time_sec=2.5)
    calls = []
    provenance = {"batch_sha256": "fixture"}

    def load(directory, config, signature, split, cohort):
        calls.append(cohort)
        assert cohort == "calibration"
        return [(report, clip)], provenance, {}

    monkeypatch.setattr(study, "load_reports", load)
    monkeypatch.setattr(sys, "argv", ["calibrate", "calibrate", "--split-scheme", "iid", "--protocol", str(protocol_path)])
    study.main()
    output = tmp_path / "outputs/accident/continuous_event_iid_selection.json"
    first = hashlib.sha256(output.read_bytes()).hexdigest()
    study.main()
    assert calls == ["calibration", "calibration"]
    assert first == hashlib.sha256(output.read_bytes()).hexdigest()
    provenance["batch_sha256"] = "changed"
    with pytest.raises(ValueError, match="inputs changed"):
        study.main()
