import numpy as np

from illegal_parking.incident_video_pipeline import expand_window_flags, expand_window_probabilities


def test_overlay_waits_until_complete_window_and_holds_latest_decision():
    probabilities = expand_window_probabilities(10, np.array([0, 3]), np.array([0.9, 0.1]), 3)
    assert np.all(np.isnan(probabilities[:3]))
    np.testing.assert_array_equal(probabilities[3:6], [0.9, 0.9, 0.9])
    np.testing.assert_array_equal(probabilities[6:], [0.1] * 4)
    flags = expand_window_flags(10, np.array([0, 3]), np.array([True, False]), 3)
    assert flags.tolist() == [False, False, False, True, True, True, False, False, False, False]


def test_overlapping_windows_never_display_future_scores_or_take_future_max():
    probabilities = expand_window_probabilities(8, np.array([0, 1, 2]), np.array([0.2, 0.9, 0.1]), 3)
    assert np.isnan(probabilities[2])
    assert probabilities[3] == 0.2
    assert probabilities[4] == 0.9
    assert probabilities[5] == 0.1
