from __future__ import annotations

import numpy as np
import pytest
import cv2

from illegal_parking.incident_features import (
    aggregate_motion_features,
    compute_frame_motion,
    compute_motion_sequence,
    extract_video_window_features,
    motion_sequence_matrix,
    read_video_window,
)


def test_compute_frame_motion_is_zero_for_identical_frames() -> None:
    frame = np.zeros((48, 64, 3), dtype=np.uint8)

    features = compute_frame_motion(frame, frame.copy(), (0.25, 0.25, 0.75, 0.75))

    assert features["global_diff_mean"] == pytest.approx(0.0)
    assert features["roi_diff_mean"] == pytest.approx(0.0)
    assert features["global_flow_mean"] == pytest.approx(0.0, abs=1e-6)
    assert features["roi_flow_mean"] == pytest.approx(0.0, abs=1e-6)


def test_compute_frame_motion_emphasizes_change_inside_accident_roi() -> None:
    previous = np.zeros((60, 80, 3), dtype=np.uint8)
    current = previous.copy()
    current[20:40, 30:50] = 255

    features = compute_frame_motion(previous, current, (0.35, 0.3, 0.65, 0.7))

    assert features["roi_diff_mean"] > features["global_diff_mean"]
    assert features["roi_change_ratio"] > features["global_change_ratio"]
    assert features["roi_diff_mean"] > 0


def test_compute_motion_sequence_returns_one_record_per_frame_pair() -> None:
    frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in (0, 0, 64, 128)]

    sequence = compute_motion_sequence(frames, (0.0, 0.0, 1.0, 1.0))

    assert len(sequence) == 3
    assert sequence[0]["global_diff_mean"] == pytest.approx(0.0)
    assert sequence[1]["global_diff_mean"] > 0


def test_motion_sequence_matrix_has_stable_feature_order() -> None:
    frames = [np.full((24, 32, 3), value, dtype=np.uint8) for value in (0, 32, 64, 96)]

    matrix = motion_sequence_matrix(
        compute_motion_sequence(frames, (0.0, 0.0, 1.0, 1.0))
    )

    assert matrix.shape == (3, 8)
    assert matrix.dtype == np.float32
    assert np.all(np.isfinite(matrix))


def test_aggregate_motion_features_reports_mean_max_std_and_peak_position() -> None:
    sequence = [
        {"global_diff_mean": 0.0, "roi_diff_mean": 1.0},
        {"global_diff_mean": 4.0, "roi_diff_mean": 8.0},
        {"global_diff_mean": 2.0, "roi_diff_mean": 3.0},
    ]

    result = aggregate_motion_features(sequence)

    assert result["global_diff_mean_mean"] == pytest.approx(2.0)
    assert result["global_diff_mean_max"] == pytest.approx(4.0)
    assert result["global_diff_mean_std"] == pytest.approx(np.std([0.0, 4.0, 2.0]))
    assert result["motion_peak_position"] == pytest.approx(0.5)


def test_aggregate_motion_features_rejects_empty_sequence() -> None:
    with pytest.raises(ValueError, match="empty motion sequence"):
        aggregate_motion_features([])


def test_video_window_feature_extraction_reads_selected_frames(tmp_path) -> None:
    video_path = tmp_path / "motion.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for offset in range(8):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        frame[16:28, 8 + offset * 3 : 20 + offset * 3] = 255
        writer.write(frame)
    writer.release()

    frames = read_video_window(video_path, 1, 8, frame_stride=2, target_width=32)
    features = extract_video_window_features(
        video_path,
        start_frame=0,
        end_frame=8,
        roi_normalized=(0.1, 0.2, 0.8, 0.8),
        frame_stride=2,
        target_width=32,
    )

    assert len(frames) == 4
    assert frames[0].shape[1] == 32
    assert features["roi_flow_mean_max"] > 0.0
    assert all(np.isfinite(value) for value in features.values())
