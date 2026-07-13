from __future__ import annotations

import pytest

from illegal_parking.incident_feature_fusion import merge_feature_rows


def _row(path: str, label: str, feature_name: str, value: str) -> dict[str, str]:
    return {
        "path": path,
        "label": label,
        "target": str(int(label == "incident")),
        "collision_type": "t-bone",
        "region": "test",
        "quality": "Good",
        "day_time": "day",
        "iid_split": "train",
        "geographic_split": "test",
        "start_frame": "10",
        "end_frame": "42",
        feature_name: value,
    }


def test_merge_feature_rows_prefixes_both_modalities() -> None:
    motion = [_row("clip.mp4", "incident", "roi_flow_mean", "1.5")]
    trajectory = [_row("clip.mp4", "incident", "min_ttc_sec_min", "0.4")]

    merged = merge_feature_rows(motion, trajectory)

    assert len(merged) == 1
    assert merged[0]["motion_roi_flow_mean"] == "1.5"
    assert merged[0]["trajectory_min_ttc_sec_min"] == "0.4"
    assert merged[0]["target"] == "1"


def test_merge_feature_rows_rejects_window_mismatch() -> None:
    motion = [_row("first.mp4", "incident", "roi_flow_mean", "1.5")]
    trajectory = [_row("second.mp4", "incident", "min_ttc_sec_min", "0.4")]

    with pytest.raises(ValueError, match="Window keys do not match"):
        merge_feature_rows(motion, trajectory)


def test_merge_feature_rows_rejects_split_disagreement() -> None:
    motion = [_row("clip.mp4", "incident", "roi_flow_mean", "1.5")]
    trajectory = [_row("clip.mp4", "incident", "min_ttc_sec_min", "0.4")]
    trajectory[0]["iid_split"] = "test"

    with pytest.raises(ValueError, match="iid_split"):
        merge_feature_rows(motion, trajectory)
