from __future__ import annotations

import math
from typing import Any


def aggregate_normal_cctv_reports(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("At least one CCTV report is required")
    evaluations = []
    for report in reports:
        evaluation = report.get("normal_only_evaluation")
        if not isinstance(evaluation, dict):
            raise ValueError("Each report must contain normal_only_evaluation")
        evaluations.append(evaluation)
    operating_points = {
        (
            str(report.get("model", "")),
            float(report.get("threshold", -1.0)),
            int(report.get(
                "min_positive_windows",
                evaluation.get("min_consecutive_windows", 0),
            )),
        )
        for report, evaluation in zip(reports, evaluations)
    }
    if len(operating_points) != 1:
        raise ValueError("CCTV reports must use the same model operating point")
    camera_hours = sum(float(item["observed_camera_hours"]) for item in evaluations)
    if camera_hours <= 0:
        raise ValueError("Total observed camera hours must be positive")
    evaluated_windows = sum(int(item["evaluated_windows"]) for item in evaluations)
    raw_positive_windows = sum(int(item["raw_positive_windows"]) for item in evaluations)
    alert_episodes = sum(int(item["false_alert_episodes"]) for item in evaluations)
    model, threshold, minimum_windows = next(iter(operating_points))
    return {
        "session_count": len(reports),
        "model": model,
        "threshold": threshold,
        "min_positive_windows": minimum_windows,
        "observed_camera_hours": camera_hours,
        "evaluated_windows": evaluated_windows,
        "raw_positive_windows": raw_positive_windows,
        "raw_positive_window_rate": (
            raw_positive_windows / evaluated_windows if evaluated_windows else 0.0
        ),
        "false_alert_episodes": alert_episodes,
        "false_alerts_per_camera_hour": alert_episodes / camera_hours,
        "zero_event_95pct_upper_per_camera_hour": (
            -math.log(0.05) / camera_hours if alert_episodes == 0 else None
        ),
    }
