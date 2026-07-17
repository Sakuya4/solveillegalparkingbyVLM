from __future__ import annotations

import numpy as np
import pytest

from illegal_parking.incident_tcn_inference import (
    build_sequence_windows,
    persistent_positive_flags,
    summarize_normal_video_predictions,
)


def test_build_sequence_windows_uses_non_overlapping_hops() -> None:
    features = np.arange(12, dtype=np.float32).reshape(6, 2)

    windows, starts = build_sequence_windows(features, sequence_steps=3, hop_steps=2)

    assert windows.shape == (2, 3, 2)
    assert starts.tolist() == [0, 2]
    assert windows[1].tolist() == features[2:5].tolist()


def test_build_sequence_windows_rejects_short_input() -> None:
    with pytest.raises(ValueError, match="not contain a complete window"):
        build_sequence_windows(np.zeros((2, 3)), sequence_steps=3, hop_steps=1)


def test_normal_video_summary_counts_alert_episodes_not_positive_windows() -> None:
    summary = summarize_normal_video_predictions(
        probabilities=np.asarray([0.7, 0.8, 0.2, 0.9]),
        threshold=0.6,
        observed_duration_sec=120.0,
    )

    assert summary["positive_windows"] == 3
    assert summary["false_alert_episodes"] == 2
    assert summary["false_alerts_per_camera_hour"] == pytest.approx(60.0)


def test_normal_video_summary_requires_positive_duration() -> None:
    with pytest.raises(ValueError, match="observed_duration_sec"):
        summarize_normal_video_predictions(np.asarray([0.1]), 0.5, 0.0)


def test_normal_video_summary_applies_consecutive_window_gate() -> None:
    summary = summarize_normal_video_predictions(
        probabilities=np.asarray([0.7, 0.2, 0.8, 0.9, 0.95]),
        threshold=0.6,
        observed_duration_sec=60.0,
        min_consecutive_windows=2,
    )

    assert summary["raw_positive_windows"] == 4
    assert summary["false_alert_episodes"] == 1
    assert summary["false_alerts_per_camera_hour"] == pytest.approx(60.0)


def test_persistent_positive_flags_marks_only_confirmed_part_of_run() -> None:
    flags = persistent_positive_flags(
        probabilities=np.asarray([0.7, 0.8, 0.9, 0.1, 0.95]),
        threshold=0.6,
        min_consecutive_windows=2,
    )

    assert flags.tolist() == [False, True, True, False, False]


def test_persistent_positive_flags_rejects_invalid_scores() -> None:
    with pytest.raises(ValueError, match="probabilities"):
        persistent_positive_flags(np.asarray([np.nan]), 0.5, 1)
    with pytest.raises(ValueError, match="threshold"):
        persistent_positive_flags(np.asarray([0.5]), 1.1, 1)
