from __future__ import annotations

import json

import pytest

from hardware_sim.python_golden.candidate_trace_workload import (
    build_candidate_flags,
    load_candidate_trace,
    repeat_candidate_flags,
)


def test_load_candidate_trace_and_threshold_flags(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    rows = [
        {"candidate_available": 1.0, "candidate_score": 0.95},
        {"candidate_available": 0.0, "candidate_score": 0.99},
        {"candidate_available": 1.0, "candidate_score": 0.949},
    ]
    trace_path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    records = load_candidate_trace(trace_path)

    assert build_candidate_flags(records, score_threshold=0.95).tolist() == [True, False, False]


def test_load_candidate_trace_rejects_invalid_score(tmp_path) -> None:
    trace_path = tmp_path / "trace.jsonl"
    trace_path.write_text(
        json.dumps({"candidate_available": 1.0, "candidate_score": 1.2}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="line 1"):
        load_candidate_trace(trace_path)


def test_repeat_candidate_flags_matches_frame_arrival_order() -> None:
    flags = repeat_candidate_flags([True, False, True], camera_count=2)

    assert flags.tolist() == [True, True, False, False, True, True]


def test_candidate_workload_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="score_threshold"):
        build_candidate_flags([], score_threshold=-0.1)
    with pytest.raises(ValueError, match="camera_count"):
        repeat_candidate_flags([True], camera_count=0)
