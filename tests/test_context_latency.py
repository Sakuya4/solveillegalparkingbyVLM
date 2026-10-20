import importlib.util
from pathlib import Path

import pytest


def benchmark():
    path = Path(__file__).resolve().parents[1] / "scripts/benchmark_context_models.py"
    spec = importlib.util.spec_from_file_location("context_latency", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temporal_cpu_timing_summary_retains_calls_and_millisecond_units():
    result = benchmark().summarize_timings([.001, .002, .003])
    assert result["calls"] == 3 and result["median_ms"] == 2
    assert result["mean_ms"] == 2 and result["p95_ms"] == pytest.approx(2.9)


@pytest.mark.parametrize("values", [[], [-1], [float("nan")], [float("inf")]])
def test_latency_summary_rejects_invalid_samples(values):
    with pytest.raises(ValueError):
        benchmark().summarize_timings(values)
