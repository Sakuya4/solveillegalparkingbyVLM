from __future__ import annotations

import pytest

from hardware_sim.python_golden.npu_queue_model import (
    EdgeQueueProfile,
    simulate_camera_capacity_sweep,
    simulate_edge_queues,
    simulate_edge_trace_queues,
)


def test_edge_queue_processes_all_frames_when_npu_has_capacity() -> None:
    profile = EdgeQueueProfile(
        camera_count=2,
        camera_fps=5.0,
        duration_sec=10.0,
        detector_latency_ms=20.0,
        temporal_latency_ms=5.0,
        npu_queue_capacity=8,
        candidate_probability=0.1,
        review_latency_ms=100.0,
        review_queue_capacity=4,
    )

    result = simulate_edge_queues(profile)

    assert result.generated_frames == 100
    assert result.processed_frames == 100
    assert result.dropped_frames == 0
    assert result.npu_utilization == pytest.approx(0.25)
    assert result.generated_candidates == 10
    assert result.completed_reviews == 10


def test_edge_queue_drops_frames_when_shared_npu_is_overloaded() -> None:
    profile = EdgeQueueProfile(
        camera_count=4,
        camera_fps=20.0,
        duration_sec=5.0,
        detector_latency_ms=30.0,
        temporal_latency_ms=5.0,
        npu_queue_capacity=2,
        candidate_probability=0.0,
        review_latency_ms=100.0,
        review_queue_capacity=2,
    )

    result = simulate_edge_queues(profile)

    assert result.generated_frames == 400
    assert result.dropped_frames > 0
    assert result.processed_frames < result.generated_frames
    assert result.pending_frames > 0
    assert result.processed_frames + result.pending_frames + result.dropped_frames == result.generated_frames
    assert result.npu_utilization == pytest.approx(1.0, abs=0.02)
    assert result.p95_frame_latency_ms > profile.detector_latency_ms


def test_review_queue_can_drop_candidates_independently_of_npu_frames() -> None:
    profile = EdgeQueueProfile(
        camera_count=1,
        camera_fps=10.0,
        duration_sec=5.0,
        detector_latency_ms=5.0,
        temporal_latency_ms=1.0,
        npu_queue_capacity=4,
        candidate_probability=1.0,
        review_latency_ms=1000.0,
        review_queue_capacity=1,
    )

    result = simulate_edge_queues(profile)

    assert result.dropped_frames == 0
    assert result.generated_candidates == 50
    assert result.dropped_candidates > 0
    assert result.completed_reviews < result.generated_candidates


def test_capacity_sweep_finds_maximum_zero_drop_camera_count() -> None:
    profile = EdgeQueueProfile(
        camera_count=1,
        camera_fps=15.0,
        duration_sec=10.0,
        detector_latency_ms=29.0,
        temporal_latency_ms=3.0,
        npu_queue_capacity=8,
        candidate_probability=0.0,
        review_latency_ms=0.0,
        review_queue_capacity=0,
    )

    sweep = simulate_camera_capacity_sweep(profile, max_camera_count=4)

    assert sweep["max_zero_drop_cameras"] == 2
    assert [row["camera_count"] for row in sweep["profiles"]] == [1, 2, 3, 4]
    assert sweep["profiles"][1]["frame_drop_rate"] == pytest.approx(0.0)
    assert sweep["profiles"][2]["frame_drop_rate"] > 0.0


def test_trace_queue_uses_explicit_candidate_flags() -> None:
    profile = EdgeQueueProfile(
        camera_count=1,
        camera_fps=5.0,
        duration_sec=1.0,
        detector_latency_ms=5.0,
        temporal_latency_ms=1.0,
        npu_queue_capacity=4,
        candidate_probability=0.0,
        review_latency_ms=10.0,
        review_queue_capacity=4,
    )

    result = simulate_edge_trace_queues(profile, [False, True, False, True, False])

    assert result.generated_frames == 5
    assert result.generated_candidates == 2


def test_trace_review_identities_preserve_completion_pending_and_overflow():
    profile = EdgeQueueProfile(2, 2, 2, 1, 1, 8, 1, 1000, 1)
    events = []
    result = simulate_edge_trace_queues(profile, [True] * 8, review_events=events)
    assert len(events) == result.generated_candidates == 8
    assert [event["source_index"] for event in events] == list(range(8))
    for status, expected in (("completed", result.completed_reviews), ("pending", result.pending_reviews), ("dropped", result.dropped_candidates)):
        assert sum(event["status"] == status for event in events) == expected
        assert expected > 0
    assert events[0]["finish_ms"] == pytest.approx(1002)


def test_variable_service_is_bound_to_source_frame_and_actual_service_cost():
    profile = EdgeQueueProfile(1, 2, 5, 1, 1, 8, 1, 1000, 1)
    flags = [True, True, False, True, False, True, False, False, False, False]
    service = [100, 2000, 9000, 100, 9000, 300, 9000, 9000, 9000, 9000]
    events = []
    result = simulate_edge_trace_queues(profile, flags, review_events=events, review_service_ms=service)
    assert result.completed_reviews == 4
    assert result.review_utilization == pytest.approx(0.5)
    assert [event["source_index"] for event in events] == [0, 1, 3, 5]
    assert [event["finish_ms"] for event in events] == pytest.approx([102, 2502, 2602, 2902])
    for invalid in ([1] * 9, [float("nan")] * 10, [-1] * 10):
        with pytest.raises(ValueError, match="service trace"):
            simulate_edge_trace_queues(profile, flags, review_service_ms=invalid)


def test_trace_queue_rejects_flag_count_mismatch() -> None:
    profile = EdgeQueueProfile(
        camera_count=1,
        camera_fps=5.0,
        duration_sec=1.0,
        detector_latency_ms=5.0,
        temporal_latency_ms=1.0,
        npu_queue_capacity=4,
        candidate_probability=0.0,
        review_latency_ms=10.0,
        review_queue_capacity=4,
    )

    with pytest.raises(ValueError, match="candidate flags"):
        simulate_edge_trace_queues(profile, [True, False])
