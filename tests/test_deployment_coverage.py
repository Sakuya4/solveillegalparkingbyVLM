from pathlib import Path

import pytest

from illegal_parking.deployment_coverage import (
    haversine_m,
    load_camera_sites,
    summarize_risk_camera_coverage,
)


def test_haversine_distance_is_symmetric() -> None:
    first = haversine_m(25.0, 121.5, 25.001, 121.5)
    second = haversine_m(25.001, 121.5, 25.0, 121.5)

    assert first == pytest.approx(second)
    assert 110 < first < 112


def test_camera_coverage_reports_count_and_risk_weighted_rates(tmp_path: Path) -> None:
    camera_csv = tmp_path / "cameras.csv"
    camera_csv.write_text(
        "seqno,location,item,latitude,longitude\n"
        "1,near,parking,25.0,121.5\n"
        "bad,row,ignored,nope,nope\n",
        encoding="utf-8",
    )
    hotspots = [
        {
            "rank": 1,
            "grid_id": "near",
            "representative_location": "near",
            "risk_score": 9,
            "latitude": 25.001,
            "longitude": 121.5,
        },
        {
            "rank": 2,
            "grid_id": "far",
            "representative_location": "far",
            "risk_score": 1,
            "latitude": 25.02,
            "longitude": 121.5,
        },
    ]

    report = summarize_risk_camera_coverage(
        hotspots, load_camera_sites(camera_csv), radii_m=(500.0,)
    )

    coverage = report["coverage_by_radius_m"]["500"]
    assert report["camera_site_count"] == 1
    assert coverage["hotspot_coverage_rate"] == 0.5
    assert coverage["risk_weighted_coverage_rate"] == 0.9
