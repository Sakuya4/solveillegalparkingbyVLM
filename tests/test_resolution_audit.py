import pytest

from scripts.publish_resolution_audit import align_window_grid


def test_common_grid_recomputes_persistence_instead_of_reusing_fine_grid_flags():
    report = {"window_starts": [0, 5, 10, 15], "window_probabilities": [0.9, 0.1, 0.2, 0.9], "window_review_flags": [False] * 4, "threshold": 0.5, "min_positive_windows": 2}
    aligned = align_window_grid(report, [0, 15], 15)
    assert aligned["window_probabilities"] == [0.9, 0.9]
    assert aligned["window_review_flags"] == [False, True]
    assert aligned["hop_steps"] == 15
    assert report["window_starts"] == [0, 5, 10, 15]
    with pytest.raises(ValueError, match="requested"):
        align_window_grid(report, [0, 30], 15)
