import json
import sys

import pytest

from scripts import run_overnight_research as runner


def test_resolution_phase_preserves_reference_models_and_matched_event_selection():
    steps = runner.resolution_steps()
    batches = [arguments for _, arguments in steps if arguments[0].endswith("run_accident_event_batch.py")]
    assert len(steps) == 6 and len(batches) == 2
    for arguments in batches:
        assert arguments[arguments.index("--feature-width") + 1] == "320"
        assert arguments[arguments.index("--sampling-seed") + 1] == "42"
        assert arguments[arguments.index("--max-clips") + 1] == "100"
        assert arguments[arguments.index("--threads") + 1] == "2"
        assert "candidate500_tcn_only" in arguments[arguments.index("--model") + 1]


def test_representation_phase_keeps_full_cohort_and_frozen_calibration_protocol():
    steps = runner.representation_steps()
    assert len(steps) == 5
    assert len({name for name, _ in steps}) == 5
    for name, arguments in steps:
        if "--max-clips" in arguments:
            assert arguments[arguments.index("--max-clips") + 1] == "2027"
        if "frozen" in name:
            assert arguments[arguments.index("--target-train-fpr") + 1] == "0.2"
        assert not any("_500" in part for part in arguments)


def test_deadline_prevents_starting_new_work(monkeypatch, tmp_path):
    (tmp_path / "worker.py").write_text("print('work')", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "workflow_steps", lambda: [("first", ["worker.py"])])
    monkeypatch.setattr(sys, "argv", ["runner", "--deadline", "2000-01-01T00:00:00+00:00"])
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *args, **kwargs: pytest.fail("Work started after deadline"))
    runner.main()
    state = json.loads((tmp_path / "outputs/research_runs/overnight_status.json").read_text())
    assert state["status"] == "deadline_reached"
    assert not state["completed_steps"]


def test_failed_step_stops_downstream_work_and_preserves_status(monkeypatch, tmp_path):
    (tmp_path / "worker.py").write_text("raise RuntimeError('failed')", encoding="utf-8")
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "workflow_steps", lambda: [("first", ["worker.py"]), ("second", ["worker.py"])])
    monkeypatch.setattr(sys, "argv", ["runner", "--deadline", "2099-01-01T00:00:00+00:00"])
    calls = []

    class FailedProcess:
        pid = 123
        returncode = 1

        def poll(self):
            return 1

    def launch(*args, **kwargs):
        calls.append(args)
        return FailedProcess()

    monkeypatch.setattr(runner.subprocess, "Popen", launch)
    with pytest.raises(RuntimeError, match="first"):
        runner.main()
    state = json.loads((tmp_path / "outputs/research_runs/overnight_status.json").read_text())
    assert len(calls) == 1
    assert state["status"] == "failed"
    assert state["current_step"] == "first"
    assert state["child_pid"] is None
