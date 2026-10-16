import json
import sys
import os
import subprocess
import time

import pytest

from scripts import run_overnight_research as runner


def test_decoding_replication_queue_does_not_restart_training_or_overwrite_legacy():
    steps = runner.decoding_steps()
    assert len(steps) == 3
    assert all("forwarded_v2" in name for name, _ in steps)
    assert all(arguments[0] == "scripts/run_incident_vlm_batch.py" and arguments[1] == "review" for _, arguments in steps)
    assert "incident_vlm_paired_dense8" in steps[-1][1][-1]


def test_calibration_queue_freezes_thresholds_before_each_test_batch():
    steps = runner.calibration_steps()
    assert len(steps) == 8
    for offset, split in ((0, "iid"), (4, "geographic")):
        train, select, test, evaluate = [arguments for _, arguments in steps[offset:offset + 4]]
        assert train[train.index("--cohort") + 1] == "calibration"
        assert select[1] == "calibrate" and evaluate[1] == "evaluate"
        assert test[test.index("--cohort") + 1] == "test"
        assert "--threshold-selection" in test
        assert "--source-clock" in train and "--source-clock" in test
        assert train[train.index("--hop-steps") + 1] == "5"
        assert f"candidate_full_tcn_{split}" in train[train.index("--model") + 1]


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


def test_worker_shutdown_stops_descendants_including_windows_redirectors(tmp_path):
    import psutil

    pid_path = tmp_path / "descendant.txt"
    code = "import pathlib,subprocess,sys,time; child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    child = subprocess.Popen([sys.executable, "-c", code, str(pid_path)], creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
    try:
        deadline = time.monotonic() + 10
        while not pid_path.exists() and time.monotonic() < deadline:
            time.sleep(.05)
        assert pid_path.exists()
        descendant_pid = int(pid_path.read_text())
        assert psutil.pid_exists(descendant_pid)
        runner.stop_worker_tree(child.pid)
        child.wait(timeout=10)
        assert not psutil.pid_exists(descendant_pid)
    finally:
        if child.poll() is None:
            runner.stop_worker_tree(child.pid)
            child.wait(timeout=10)


def test_worker_shutdown_tolerates_an_already_exited_process(monkeypatch):
    import psutil

    def missing(pid):
        raise psutil.NoSuchProcess(pid)

    monkeypatch.setattr(psutil, "Process", missing)
    runner.stop_worker_tree(123)
