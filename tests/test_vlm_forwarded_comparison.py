import pytest

from scripts.publish_vlm_forwarded_comparison import paired_correctness


def test_paired_vlm_correctness_keeps_cases_within_source_clip():
    rows = [{"case_id": f"{clip}_{event}", "clip": clip, "expected_collision": bool(event), "schema_valid": True, "likely_violation": bool(event)} for clip in ("a", "b") for event in (0, 1)]
    other = [{**row, "schema_valid": False} for row in rows]
    interval = paired_correctness(rows, other)
    assert interval["clusters"] == 2
    assert interval["difference"] == interval["lower_95"] == interval["upper_95"] == 1


def test_paired_vlm_comparison_rejects_changed_labels_or_cohorts():
    rows = [{"case_id": "a", "clip": "a", "expected_collision": True, "schema_valid": True, "likely_violation": True}]
    with pytest.raises(ValueError, match="identical"):
        paired_correctness(rows, [{**rows[0], "expected_collision": False}])
    with pytest.raises(ValueError, match="identical"):
        paired_correctness(rows, [{**rows[0], "case_id": "b"}])
