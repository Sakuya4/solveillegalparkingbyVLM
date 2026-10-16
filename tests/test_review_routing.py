import numpy as np
import pytest

from hardware_sim.python_golden.review_routing import coalesce_candidate_flags, meter_review_flags


def records(count):
    return [{"path": "camera.mp4", "frame_index": index * 2, "candidate_bbox": [0, 0, 0.5, 0.5]} for index in range(count)]


def test_overlap_merging_is_causal_and_independent_of_labels():
    rows = records(6)
    flags = np.array([True, True, False, True, False, True])
    expected = [True, False, False, False, False, False]
    assert coalesce_candidate_flags(rows, flags, max_gap_steps=2).tolist() == expected
    for row in rows:
        row.update(label="changed", target=99, accident_time_sec=0)
    assert coalesce_candidate_flags(rows, flags, max_gap_steps=2).tolist() == expected
    prefix = coalesce_candidate_flags(rows[:3], flags[:3], max_gap_steps=2)
    assert prefix.tolist() == expected[:3]


def test_merge_resets_at_source_discontinuity_and_nonoverlapping_box():
    rows = records(5)
    rows[1]["path"] = "other.mp4"
    rows[3]["frame_index"] = 999
    rows[4]["candidate_bbox"] = [0.6, 0.6, 1, 1]
    assert coalesce_candidate_flags(rows, np.ones(5), max_gap_steps=15).all()


def test_no_box_proposals_are_not_silently_coalesced():
    rows = records(3)
    rows[1]["candidate_bbox"] = None
    assert coalesce_candidate_flags(rows, np.ones(3), max_gap_steps=3).all()


def test_token_bucket_respects_burst_and_rotates_camera_priority():
    admitted = meter_review_flags(np.ones((20, 4)), camera_fps=1, rate_per_sec=1, burst=1)
    assert admitted.sum() == 20
    assert admitted.sum(axis=0).tolist() == [5, 5, 5, 5]
    assert (admitted.sum(axis=1) == 1).all()


def test_token_bucket_is_causal_and_does_not_invent_proposals():
    flags = np.random.default_rng(42).random((100, 3)) > 0.3
    full = meter_review_flags(flags, camera_fps=7.5, rate_per_sec=0.12, burst=4)
    prefix = meter_review_flags(flags[:50], camera_fps=7.5, rate_per_sec=0.12, burst=4)
    assert np.array_equal(full[:50], prefix)
    assert not np.any(full & ~flags)
    assert full.sum() <= 4 + 99 / 7.5 * 0.12


@pytest.mark.parametrize("rate", [0, -1, float("nan"), float("inf")])
def test_token_bucket_rejects_invalid_rates(rate):
    with pytest.raises(ValueError):
        meter_review_flags(np.ones((10, 2)), camera_fps=1, rate_per_sec=rate)


def test_merge_rejects_invalid_boxes():
    rows = records(1)
    rows[0]["candidate_bbox"] = [0, 0, float("nan"), 1]
    with pytest.raises(ValueError):
        coalesce_candidate_flags(rows, np.ones(1), max_gap_steps=1)
