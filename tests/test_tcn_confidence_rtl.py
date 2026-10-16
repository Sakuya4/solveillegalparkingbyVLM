from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest

from hardware_sim.python_golden.tcn_confidence_model import ConfidenceGate, ConfidenceStep, context_decisions, quantize_probability

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("value,expected", [(0., 0), (.5, 32768), (1., 65535)])
def test_probability_encoding_is_endpoint_preserving_half_up(value, expected):
    assert quantize_probability(value) == expected


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -.1, 1.1])
def test_probability_encoding_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        quantize_probability(value)


def test_roundtrip_error_bound_and_near_threshold_decision_flip_are_explicit():
    values = np.random.default_rng(42).random(10000)
    assert max(abs(quantize_probability(float(value)) / 65535 - value) for value in values) <= .5 / 65535
    probability, threshold = .500001, .500002
    assert probability < threshold
    assert quantize_probability(probability) >= quantize_probability(threshold)


def test_idle_holds_state_reset_suppresses_valid_and_negative_rearms():
    gate = ConfidenceGate()
    assert gate.step(ConfidenceStep(True, False, 10, 10)) == (0, 0, 1)
    assert gate.step(ConfidenceStep(False, False, 0, 10)) == (0, 0, 1)
    assert gate.step(ConfidenceStep(True, False, 10, 10)) == (1, 1, 2)
    assert gate.step(ConfidenceStep(True, False, 10, 10)) == (1, 0, 2)
    assert gate.step(ConfidenceStep(True, True, 10, 10)) == (0, 0, 0)
    assert gate.step(ConfidenceStep(True, False, 10, 10)) == (0, 0, 1)
    assert gate.step(ConfidenceStep(True, False, 9, 10)) == (0, 0, 0)


def test_missing_source_hops_and_clip_boundaries_reset_without_backdating():
    tokens, decisions = context_decisions([0, 30, 45, 60], [.9] * 4, .5)
    assert [row["start"] for row in decisions if row["episode_pulse"]] == [45]
    assert sum(row[1] for row in tokens) == 2
    _, second_clip = context_decisions([0, 15], [.9] * 2, .5)
    assert [row["start"] for row in second_clip if row["episode_pulse"]] == [15]


def test_iverilog_matches_randomized_boundary_idle_and_reset_cycles(tmp_path):
    compiler = shutil.which("iverilog") or ("C:/iverilog/bin/iverilog.exe" if Path("C:/iverilog/bin/iverilog.exe").exists() else None)
    runtime = shutil.which("vvp") or ("C:/iverilog/bin/vvp.exe" if Path("C:/iverilog/bin/vvp.exe").exists() else None)
    if not compiler or not runtime:
        pytest.skip("Install Icarus Verilog to run the cross-language controller check")
    generator = np.random.default_rng(42)
    gate = ConfidenceGate()
    rows = []
    for _ in range(2000):
        probability = int(generator.integers(0, 65536))
        threshold = int(generator.choice([0, 1, 32768, 65535, probability]))
        step = ConfidenceStep(bool(generator.integers(0, 2)), bool(generator.integers(0, 8) == 0), probability, threshold)
        rows.append((int(step.decision_valid), int(step.context_reset), probability, threshold, *gate.step(step)))
    workspace = tmp_path / "workspace_\u6e2c\u8a66"
    workspace.mkdir()
    vectors = workspace / "vectors.txt"
    vectors.write_text("".join(" ".join(map(str, row)) + "\n" for row in rows), encoding="ascii")
    executable = workspace / "gate.vvp"
    subprocess.run([compiler, "-g2012", "-s", "tcn_confidence_gate_tb", "-o", str(executable), str(ROOT / "hardware_sim/rtl/tcn_confidence_gate.v"), str(ROOT / "hardware_sim/testbench/tcn_confidence_gate_tb.v")], check=True, capture_output=True, timeout=30)
    result = subprocess.run([runtime, executable.name, "+vectors=" + vectors.name], cwd=workspace, check=True, capture_output=True, text=True, timeout=30)
    assert "PASS tcn_confidence_gate_tb tokens=2000" in result.stdout


def test_published_real_vectors_and_float_quantization_audit_are_reproducible():
    import csv
    import hashlib
    import json

    metadata = json.loads((ROOT / "docs/results/tcn_confidence_rtl.json").read_text())
    vector_path = ROOT / "hardware_sim/test_vectors/tcn_confidence_real.txt"
    trace_path = ROOT / "docs/results/tcn_confidence_rtl_trace.csv"
    assert hashlib.sha256(vector_path.read_bytes()).hexdigest() == metadata["vectors_sha256"]
    assert hashlib.sha256(trace_path.read_bytes()).hexdigest() == metadata["trace_sha256"]
    vectors = [list(map(int, line.split())) for line in vector_path.read_text().splitlines()]
    gate = ConfidenceGate()
    for valid, reset, probability, threshold, *expected in vectors:
        assert list(gate.step(ConfidenceStep(bool(valid), bool(reset), probability, threshold))) == expected
    with trace_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == metadata["model_decisions"] == 11376
    assert len(vectors) == metadata["tokens_checked"] == 24252
    for row in rows:
        line = vectors[int(row["vector_line"]) - 1]
        assert line[0] == 1 and line[1] == 0
        assert line[2] == int(row["probability_unorm16"]) == quantize_probability(float(row["probability"]))
        assert line[3] == int(row["threshold_unorm16"]) == quantize_probability(float(row["threshold"]))
        assert line[4] == int(row["integer_review"]) == int(row["float_review"])
        assert line[5] == int(row["integer_episode"]) == int(row["float_episode"])
    assert metadata["rtl_golden_mismatches"] == 0
    assert all(row["event_detection_changes"] == row["trigger_time_changes"] == 0 for row in metadata["summaries"])
    assert max(row["maximum_probability_error"] for row in metadata["summaries"]) <= .5 / 65535
