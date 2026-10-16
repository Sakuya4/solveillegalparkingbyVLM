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


def test_systemc_variable_review_service_matches_source_bound_python_replay(tmp_path):
    root = Path(__file__).resolve().parents[1]
    executable = root / "outputs/toolchain/traffic-systemc-msvc-build/Release/npu_queue_sim.exe"
    if not executable.is_file():
        pytest.skip("Build SystemC to exercise variable service replay")
    flags = [True, True, False, True, False, True, False, False, False, False]
    service = [100, 2000, 9000, 100, 9000, 300, 9000, 9000, 9000, 9000]
    flag_path, service_path = tmp_path / "flags.txt", tmp_path / "service.txt"
    flag_path.write_text("\n".join(str(int(value)) for value in flags), encoding="ascii")
    service_path.write_text("\n".join(str(value) for value in service), encoding="ascii")
    profile = EdgeQueueProfile(1, 2, 5, 1, 1, 8, 1, 1000, 1)
    expected = simulate_edge_trace_queues(profile, flags, review_service_ms=service).to_dict()
    command = [str(executable), "--camera-count", "1", "--camera-fps", "2", "--duration-sec", "5", "--detector-latency-ms", "1", "--temporal-latency-ms", "1", "--review-queue-capacity", "1", "--candidate-flags", str(flag_path), "--review-service-ms", str(service_path)]
    replay = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=True)
    counters = dict(line.split("=", 1) for line in replay.stdout.splitlines() if "=" in line)
    assert int(counters["review_service_trace_enabled"]) == 1
    for key in ("generated_frames", "processed_frames", "pending_frames", "dropped_frames", "generated_candidates", "completed_reviews", "pending_reviews", "dropped_candidates"):
        assert int(counters[key]) == expected[key]


@pytest.mark.parametrize("cameras,duration,detector_ms", [(2, 1.0, 100), (2, 1.05, 100), (1, 0.9, 10)])
def test_systemc_idle_consumer_fifo_ties_endpoint_and_half_up_frame_counts(tmp_path, cameras, duration, detector_ms):
    root = Path(__file__).resolve().parents[1]
    executable = root / "outputs/toolchain/traffic-systemc-msvc-build/Release/npu_queue_sim.exe"
    if not executable.is_file():
        pytest.skip("Build SystemC to exercise exact queue boundaries")
    profile = EdgeQueueProfile(cameras, 5, duration, detector_ms, 0, 1, 0, 100, 1)
    frames = int(np.floor(5 * duration + 0.5)) * cameras
    flags = [True] * frames
    flag_path = tmp_path / "flags.txt"
    flag_path.write_text("\n".join("1" for _ in flags), encoding="ascii")
    expected = simulate_edge_trace_queues(profile, flags).to_dict()
    command = [str(executable), "--camera-count", str(cameras), "--camera-fps", "5", "--duration-sec", str(duration), "--detector-latency-ms", str(detector_ms), "--temporal-latency-ms", "0", "--npu-queue-capacity", "1", "--review-queue-capacity", "1", "--review-latency-ms", "100", "--candidate-flags", str(flag_path)]
    replay = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30, check=True)
    counters = dict(line.split("=", 1) for line in replay.stdout.splitlines() if "=" in line)
    for key in ("generated_frames", "processed_frames", "pending_frames", "dropped_frames", "generated_candidates", "completed_reviews", "pending_reviews", "dropped_candidates"):
        assert int(counters[key]) == expected[key], key
