from __future__ import annotations

import pytest

from hardware_sim.python_golden.npu_queue_model import EdgeQueueProfile, simulate_edge_queues


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
