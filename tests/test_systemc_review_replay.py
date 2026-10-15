from pathlib import Path
import subprocess

import numpy as np
import pytest

from hardware_sim.python_golden.npu_queue_model import EdgeQueueProfile, simulate_edge_trace_queues


def test_systemc_finite_arrival_count_and_review_parity_when_built(tmp_path):
    root = Path(__file__).resolve().parents[1]
    executable = root / "outputs/toolchain/traffic-systemc-msvc-build/Release/npu_queue_sim.exe"
    if not executable.is_file():
        pytest.skip("Build the optional SystemC executable to exercise cross-language replay")
    flags = np.tile([True, False, False, False], 12330)
    path = tmp_path / "flags.txt"
    path.write_text("".join("1\n" if flag else "0\n" for flag in flags), encoding="ascii")
    profile = EdgeQueueProfile(4, 7.5, 1644, 3.9, 3, 8, 0.25, 8273.610775003908, 8)
    expected = simulate_edge_trace_queues(profile, flags).to_dict()
    command = [str(executable), "--camera-count", "4", "--camera-fps", "7.5", "--duration-sec", "1644", "--detector-latency-ms", "3.9", "--temporal-latency-ms", "3", "--review-latency-ms", str(profile.review_latency_ms), "--candidate-flags", str(path)]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=True)
    counters = dict(line.split("=", 1) for line in result.stdout.splitlines() if "=" in line)
    for key in ("generated_frames", "processed_frames", "pending_frames", "dropped_frames", "generated_candidates", "completed_reviews", "pending_reviews", "dropped_candidates"):
        assert int(counters[key]) == expected[key]
    assert int(counters["generated_frames"]) == len(flags)
