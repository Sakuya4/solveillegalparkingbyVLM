from __future__ import annotations

import math

import pytest

from illegal_parking.cctv_evaluation import aggregate_normal_cctv_reports


def _report(duration: float, windows: int, positives: int, events: int) -> dict:
    return {
        "model": "model.pt",
        "threshold": 0.85,
        "min_positive_windows": 2,
        "normal_only_evaluation": {
            "observed_camera_hours": duration / 3600.0,
            "evaluated_windows": windows,
            "raw_positive_windows": positives,
            "false_alert_episodes": events,
        },
    }


def test_aggregate_normal_cctv_reports_sums_independent_sessions() -> None:
    result = aggregate_normal_cctv_reports([
        _report(60.0, 20, 3, 1),
        _report(120.0, 40, 2, 0),
    ])

    assert result["session_count"] == 2
    assert result["observed_camera_hours"] == pytest.approx(0.05)
    assert result["evaluated_windows"] == 60
    assert result["raw_positive_windows"] == 5
    assert result["false_alert_episodes"] == 1
    assert result["false_alerts_per_camera_hour"] == pytest.approx(20.0)
    assert result["zero_event_95pct_upper_per_camera_hour"] is None


def test_zero_event_aggregate_reports_exposure_upper_bound() -> None:
    result = aggregate_normal_cctv_reports([_report(3600.0, 100, 4, 0)])

    assert result["false_alerts_per_camera_hour"] == 0.0
    assert result["zero_event_95pct_upper_per_camera_hour"] == pytest.approx(-math.log(0.05))


def test_aggregate_normal_cctv_reports_rejects_mixed_operating_points() -> None:
    first = _report(60.0, 20, 0, 0)
    second = _report(60.0, 20, 0, 0)
    second["threshold"] = 0.9

    with pytest.raises(ValueError, match="operating point"):
        aggregate_normal_cctv_reports([first, second])


def test_aggregate_normal_cctv_reports_requires_normal_evaluation() -> None:
    with pytest.raises(ValueError, match="normal_only_evaluation"):
        aggregate_normal_cctv_reports([{"threshold": 0.5}])


def test_aggregate_normal_cctv_reports_reads_legacy_nested_gate() -> None:
    report = _report(60.0, 20, 0, 0)
    del report["min_positive_windows"]
    report["normal_only_evaluation"]["min_consecutive_windows"] = 2

    result = aggregate_normal_cctv_reports([report])

    assert result["min_positive_windows"] == 2
