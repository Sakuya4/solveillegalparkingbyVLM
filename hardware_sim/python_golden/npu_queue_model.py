from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class EdgeQueueProfile:
    camera_count: int
    camera_fps: float
    duration_sec: float
    detector_latency_ms: float
    temporal_latency_ms: float
    npu_queue_capacity: int
    candidate_probability: float
    review_latency_ms: float
    review_queue_capacity: int

    def validate(self) -> None:
        if self.camera_count <= 0 or self.camera_fps <= 0 or self.duration_sec <= 0:
            raise ValueError("camera_count, camera_fps, and duration_sec must be positive")
        if self.detector_latency_ms < 0 or self.temporal_latency_ms < 0 or self.review_latency_ms < 0:
            raise ValueError("latencies must be non-negative")
        if self.npu_queue_capacity < 0 or self.review_queue_capacity < 0:
            raise ValueError("queue capacities must be non-negative")
        if not 0.0 <= self.candidate_probability <= 1.0:
            raise ValueError("candidate_probability must be between zero and one")


@dataclass(frozen=True)
class EdgeQueueResult:
    generated_frames: int
    processed_frames: int
    pending_frames: int
    dropped_frames: int
    frame_drop_rate: float
    npu_utilization: float
    mean_frame_latency_ms: float
    p95_frame_latency_ms: float
    generated_candidates: int
    completed_reviews: int
    pending_reviews: int
    dropped_candidates: int
    review_drop_rate: float
    review_utilization: float

    def to_dict(self) -> dict:
        return asdict(self)


def simulate_edge_queues(profile: EdgeQueueProfile) -> EdgeQueueResult:
    return _simulate_edge_queues(profile, candidate_flags=None)


def simulate_edge_trace_queues(
    profile: EdgeQueueProfile,
    candidate_flags: list[bool] | np.ndarray,
) -> EdgeQueueResult:
    flags = np.asarray(candidate_flags, dtype=np.bool_).reshape(-1)
    return _simulate_edge_queues(profile, candidate_flags=flags)


def _simulate_edge_queues(
    profile: EdgeQueueProfile,
    candidate_flags: np.ndarray | None,
) -> EdgeQueueResult:
    profile.validate()
    duration_ms = profile.duration_sec * 1000.0
    arrivals = _frame_arrivals(profile.camera_count, profile.camera_fps, profile.duration_sec)
    if candidate_flags is not None and len(candidate_flags) != len(arrivals):
        raise ValueError(
            f"candidate flags must match generated frames: {len(candidate_flags)} != {len(arrivals)}"
        )
    service_ms = profile.detector_latency_ms + profile.temporal_latency_ms

    npu_finishes: deque[float] = deque()
    accepted_latencies: list[float] = []
    accepted_finishes: list[float] = []
    accepted_source_indices: list[int] = []
    dropped_frames = 0
    for source_index, arrival in enumerate(arrivals):
        _discard_completed(npu_finishes, arrival)
        if len(npu_finishes) >= profile.npu_queue_capacity + 1:
            dropped_frames += 1
            continue
        start = max(arrival, npu_finishes[-1] if npu_finishes else arrival)
        finish = start + service_ms
        npu_finishes.append(finish)
        accepted_finishes.append(finish)
        accepted_latencies.append(finish - arrival)
        accepted_source_indices.append(source_index)

    completed_indices = [index for index, finish in enumerate(accepted_finishes) if finish <= duration_ms]
    completed_finishes = [accepted_finishes[index] for index in completed_indices]
    frame_latencies = [accepted_latencies[index] for index in completed_indices]
    if candidate_flags is None:
        candidate_times = _candidate_times(completed_finishes, profile.candidate_probability)
    else:
        candidate_times = [
            finish
            for index, finish in zip(completed_indices, completed_finishes)
            if candidate_flags[accepted_source_indices[index]]
        ]
    review_finishes: deque[float] = deque()
    accepted_reviews: list[float] = []
    dropped_candidates = 0
    for candidate_time in candidate_times:
        _discard_completed(review_finishes, candidate_time)
        if len(review_finishes) >= profile.review_queue_capacity + 1:
            dropped_candidates += 1
            continue
        start = max(candidate_time, review_finishes[-1] if review_finishes else candidate_time)
        finish = start + profile.review_latency_ms
        review_finishes.append(finish)
        accepted_reviews.append(finish)

    completed_reviews = sum(finish <= duration_ms for finish in accepted_reviews)
    pending_reviews = len(accepted_reviews) - completed_reviews
    processed_frames = len(completed_finishes)
    pending_frames = len(accepted_finishes) - processed_frames
    generated_candidates = len(candidate_times)
    return EdgeQueueResult(
        generated_frames=len(arrivals),
        processed_frames=processed_frames,
        pending_frames=pending_frames,
        dropped_frames=dropped_frames,
        frame_drop_rate=_divide(dropped_frames, len(arrivals)),
        npu_utilization=min(1.0, _divide(len(accepted_finishes) * service_ms, duration_ms)),
        mean_frame_latency_ms=float(np.mean(frame_latencies)) if frame_latencies else 0.0,
        p95_frame_latency_ms=float(np.percentile(frame_latencies, 95)) if frame_latencies else 0.0,
        generated_candidates=generated_candidates,
        completed_reviews=completed_reviews,
        pending_reviews=pending_reviews,
        dropped_candidates=dropped_candidates,
        review_drop_rate=_divide(dropped_candidates, generated_candidates),
        review_utilization=min(1.0, _divide(len(accepted_reviews) * profile.review_latency_ms, duration_ms)),
    )


def simulate_camera_capacity_sweep(
    base_profile: EdgeQueueProfile,
    max_camera_count: int,
) -> dict:
    if max_camera_count <= 0:
        raise ValueError("max_camera_count must be positive")
    profiles: list[dict] = []
    for camera_count in range(1, max_camera_count + 1):
        result = simulate_edge_queues(replace(base_profile, camera_count=camera_count))
        profiles.append({"camera_count": camera_count, **result.to_dict()})
    zero_drop_counts = [row["camera_count"] for row in profiles if row["dropped_frames"] == 0]
    return {
        "max_zero_drop_cameras": max(zero_drop_counts, default=0),
        "profiles": profiles,
    }


def _frame_arrivals(camera_count: int, camera_fps: float, duration_sec: float) -> list[float]:
    frame_count_per_camera = int(round(camera_fps * duration_sec))
    return sorted(
        frame_index * 1000.0 / camera_fps
        for _camera_index in range(camera_count)
        for frame_index in range(frame_count_per_camera)
    )


def _candidate_times(frame_finishes: list[float], probability: float) -> list[float]:
    if probability <= 0:
        return []
    accumulator = 0.0
    candidates: list[float] = []
    for finish in frame_finishes:
        accumulator += probability
        if accumulator + 1e-12 >= 1.0:
            candidates.append(finish)
            accumulator -= 1.0
    return candidates


def _discard_completed(finishes: deque[float], timestamp: float) -> None:
    while finishes and finishes[0] <= timestamp:
        finishes.popleft()


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
