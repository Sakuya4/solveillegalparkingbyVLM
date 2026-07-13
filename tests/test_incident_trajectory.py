from __future__ import annotations

import numpy as np
import pytest

from illegal_parking.incident_trajectory import (
    TrackBox,
    aggregate_trajectory_features,
    build_track_boxes,
    compute_trajectory_sequence,
)


def _box(frame: int, track_id: int, center_x: float, center_y: float = 0.5) -> TrackBox:
    return TrackBox(
        frame_index=frame,
        track_id=track_id,
        x1=center_x - 0.05,
        y1=center_y - 0.05,
        x2=center_x + 0.05,
        y2=center_y + 0.05,
        confidence=0.9,
    )


def test_approaching_tracks_produce_finite_ttc() -> None:
    observations = [
        _box(0, 1, 0.2),
        _box(0, 2, 0.8),
        _box(1, 1, 0.3),
        _box(1, 2, 0.7),
        _box(2, 1, 0.4),
        _box(2, 2, 0.6),
    ]

    sequence = compute_trajectory_sequence(observations, fps=10.0, max_ttc_sec=10.0)

    assert sequence[-1]["max_closing_speed"] == pytest.approx(2.0)
    assert sequence[-1]["min_ttc_sec"] == pytest.approx(0.1)
    assert sequence[-1]["min_center_distance"] == pytest.approx(0.2)


def test_parallel_tracks_do_not_create_false_collision_ttc() -> None:
    observations = [
        _box(0, 1, 0.2),
        _box(0, 2, 0.6),
        _box(1, 1, 0.3),
        _box(1, 2, 0.7),
    ]

    sequence = compute_trajectory_sequence(observations, fps=10.0, max_ttc_sec=8.0)

    assert sequence[-1]["max_closing_speed"] == pytest.approx(0.0)
    assert sequence[-1]["min_ttc_sec"] == pytest.approx(8.0)


def test_single_track_returns_safe_pair_defaults() -> None:
    sequence = compute_trajectory_sequence(
        [_box(0, 1, 0.2), _box(1, 1, 0.3)],
        fps=10.0,
        max_ttc_sec=7.0,
    )

    assert sequence[-1]["track_count"] == 1.0
    assert sequence[-1]["pair_count"] == 0.0
    assert sequence[-1]["min_center_distance"] == 1.0
    assert sequence[-1]["min_ttc_sec"] == 7.0
    assert sequence[-1]["max_track_speed"] == pytest.approx(1.0)


def test_empty_detection_frames_remain_in_sequence() -> None:
    sequence = compute_trajectory_sequence(
        [],
        fps=10.0,
        max_ttc_sec=7.0,
        frame_indices=[4, 6],
    )

    assert len(sequence) == 2
    assert all(frame["track_count"] == 0.0 for frame in sequence)
    assert all(frame["min_ttc_sec"] == 7.0 for frame in sequence)


def test_trajectory_aggregation_is_fixed_and_finite() -> None:
    sequence = compute_trajectory_sequence(
        [_box(0, 1, 0.2), _box(1, 1, 0.3)],
        fps=10.0,
    )

    features = aggregate_trajectory_features(sequence)

    assert "min_ttc_sec_min" in features
    assert "max_track_speed_max" in features
    assert "risk_peak_position" in features
    assert all(np.isfinite(value) for value in features.values())


@pytest.mark.parametrize("fps", [0.0, -1.0])
def test_trajectory_sequence_rejects_invalid_fps(fps: float) -> None:
    with pytest.raises(ValueError, match="fps"):
        compute_trajectory_sequence([_box(0, 1, 0.2)], fps=fps)


def test_track_box_rejects_non_normalized_coordinates() -> None:
    with pytest.raises(ValueError, match="normalized"):
        TrackBox(frame_index=0, track_id=1, x1=-0.1, y1=0.0, x2=0.2, y2=0.2)


def test_build_track_boxes_normalizes_and_clips_detector_output() -> None:
    boxes = build_track_boxes(
        frame_index=3,
        boxes_xyxy=np.asarray([[-2.0, 10.0, 50.0, 110.0], [30.0, 20.0, 30.0, 40.0]]),
        track_ids=np.asarray([7, 8]),
        confidences=np.asarray([0.8, 0.9]),
        frame_width=100,
        frame_height=100,
    )

    assert len(boxes) == 1
    assert boxes[0].track_id == 7
    assert boxes[0].x1 == 0.0
    assert boxes[0].y2 == 1.0
