import pytest

from scripts.publish_presentation_event_trace import outcome_rows


def fixture():
    protocol = {"models": {"fixed_candidate_iid": {"split": "iid", "arm": "fixed", "feature": "candidate"}},
                "cohorts": {"iid": {"calibration_clip_paths": ["a"], "test_clip_paths": ["b"]}},
                "annotations": {name: {"path": name, "event_time_sec": 3, "collision_type": "rear-end"} for name in ("a", "b")}}
    report = {"path": "a", "source_sha256": "media", "clock": "normalized_presentation_time", "annotation_free_inference": True,
              "window_starts": [0, 15], "window_end_times_sec": [2, 4], "rejected_window_starts": [],
              "scores": {"fixed_candidate_iid": [.9, .9]}}
    selection = {"policies": {"fixed_candidate_iid": {"selected": {"threshold": .5}}}}
    return protocol, selection, [report]


def test_public_trace_retains_clock_controls_source_identity_and_true_misses():
    protocol, selection, reports = fixture()
    rows = outcome_rows(protocol, selection, reports, "calibration")
    assert len(rows) == 3
    assert rows[0]["detected"] == rows[1]["detected"] == 1
    assert rows[0]["signed_delay_sec"] == 1
    assert rows[2]["detected"] == 0 and rows[2]["signed_delay_sec"] is None
    assert all(row["source_sha256"] == "media" and row["path"] == "a" for row in rows)
    assert all(row["cohort"] == "calibration" for row in rows)


def test_trace_does_not_silently_replace_missing_test_clips_or_infeasible_policies():
    protocol, selection, reports = fixture()
    with pytest.raises(KeyError):
        outcome_rows(protocol, selection, reports, "test")
    selection["policies"]["fixed_candidate_iid"]["selected"] = None
    rows = outcome_rows(protocol, selection, reports, "calibration")
    assert rows[0]["threshold"] is None and rows[0]["detected"] is None
    assert rows[1]["detected"] == 1
