import json
import os
from pathlib import Path
import subprocess
import sys

import psutil
import pytest

from illegal_parking.gpu_job import active_research_child, exclusive_gpu_job


def test_gpu_lock_rejects_overlap_and_releases_after_exception(tmp_path):
    with pytest.raises(ValueError):
        with exclusive_gpu_job(tmp_path):
            with pytest.raises(RuntimeError, match="Another GPU"):
                with exclusive_gpu_job(tmp_path):
                    pytest.fail("Overlapping job entered")
            raise ValueError("worker failed")
    with exclusive_gpu_job(tmp_path):
        pass


def test_live_child_is_checked_even_with_stale_completed_status(tmp_path):
    directory = tmp_path / "outputs/research_runs"
    directory.mkdir(parents=True)
    command = psutil.Process().cmdline()
    path = directory / "calibration_status.json"
    path.write_text(json.dumps({"status": "complete", "child_pid": os.getpid(), "command": command}), encoding="utf-8")
    assert active_research_child(tmp_path) == os.getpid()
    with pytest.raises(RuntimeError, match="Research child"):
        with exclusive_gpu_job(tmp_path):
            pytest.fail("GPU job started over a live research child")
    path.write_text(json.dumps({"child_pid": os.getpid(), "command": ["python", "unrelated_script.py"]}), encoding="utf-8")
    assert active_research_child(tmp_path) is None


def test_terminated_gpu_worker_does_not_leave_a_stale_lock(tmp_path):
    source = Path(__file__).resolve().parents[1] / "Application/DebugVersion/src"
    code = "import sys,time; from pathlib import Path; from illegal_parking.gpu_job import exclusive_gpu_job;\nwith exclusive_gpu_job(Path(sys.argv[1])):\n print('locked',flush=True); time.sleep(60)"
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], env={**os.environ, "PYTHONPATH": str(source)}, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        assert child.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="Another GPU"):
            with exclusive_gpu_job(tmp_path):
                pytest.fail("Concurrent worker entered")
    finally:
        child.terminate()
        child.communicate(timeout=10)
    with exclusive_gpu_job(tmp_path):
        pass
