from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from illegal_parking.accident_dataset import AccidentClip
from illegal_parking.incident_evaluation import (
    evaluate_incident_report,
    summarize_incident_evaluations,
)


def _clip(accident_time_sec: float = 2.5) -> AccidentClip:
    return AccidentClip(
        relative_path=Path("real_videos/a.mp4"),
        collision_type="t-bone",
        accident_time_sec=accident_time_sec,
        accident_frame=25,
        bbox_normalized=(0.2, 0.2, 0.4, 0.4),
        region="Taiwan",
        scene_layout="intersection",
        weather="normal",
        day_time="day",
        quality="Good",
        frame_count=100,
        duration_sec=10.0,
        height=720,
        width=1280,
        iid_split="test",
        geographic_split="test",
    )


def _report(flags: list[bool]) -> dict:
    return {
        "input": "dataset/real_videos/a.mp4",
        "annotation_free_inference": True,
        "sampled_fps": 10.0,
        "sequence_steps": 10,
        "window_starts": [0, 10, 20, 30, 40],
        "window_probabilities": [0.1, 0.9, 0.8, 0.2, 0.95],
        "window_review_flags": flags,
    }


def test_evaluate_incident_report_uses_window_end_as_decision_time() -> None:
    result = evaluate_incident_report(
        _report([False, True, True, False, True]),
        _clip(accident_time_sec=2.5),
        early_tolerance_sec=1.0,
        late_tolerance_sec=1.0,
    )

    assert result.detected is True
    assert result.trigger_time_sec == pytest.approx(2.0)
    assert result.trigger_delay_sec == pytest.approx(-0.5)
    assert result.early_alert_episodes == 0
    assert result.late_alert_episodes == 1


def test_evaluate_incident_report_marks_early_only_alert_as_missed() -> None:
    result = evaluate_incident_report(
        _report([True, True, False, False, False]),
        _clip(accident_time_sec=5.0),
        early_tolerance_sec=1.0,
        late_tolerance_sec=2.0,
    )

    assert result.detected is False
    assert result.trigger_time_sec is None
    assert result.trigger_delay_sec is None
    assert result.early_alert_episodes == 1
    assert result.late_alert_episodes == 0


def test_evaluate_incident_report_rejects_oracle_inference() -> None:
    report = _report([False, True, False, False, False])
    report["annotation_free_inference"] = False

    with pytest.raises(ValueError, match="annotation-free"):
        evaluate_incident_report(report, _clip())


def test_summarize_incident_evaluations_reports_recall_delay_and_class_breakdown() -> None:
    detected = evaluate_incident_report(
        _report([False, True, False, False, False]),
        _clip(accident_time_sec=2.5),
        early_tolerance_sec=1.0,
        late_tolerance_sec=1.0,
    )
    missed = evaluate_incident_report(
        _report([False, False, False, False, False]),
        _clip(accident_time_sec=2.5),
        early_tolerance_sec=1.0,
        late_tolerance_sec=1.0,
    )

    summary = summarize_incident_evaluations([detected, missed])

    assert summary["clip_count"] == 2
    assert summary["detected_clips"] == 1
    assert summary["missed_clips"] == 1
    assert summary["event_recall"] == pytest.approx(0.5)
    assert summary["trigger_delay_sec"]["median"] == pytest.approx(-0.5)
    assert summary["by_collision_type"]["t-bone"]["event_recall"] == pytest.approx(0.5)


def test_evaluate_incident_reports_cli_matches_report_to_accident_metadata(tmp_path) -> None:
    metadata = tmp_path / "metadata-real.csv"
    metadata.write_text(
        "path,type,rollover,accident_time,accident_frame,center_x,center_y,"
        "x1,y1,x2,y2,region,scene_layout,weather,day_time,quality,no_frames,"
        "duration,height,width,split_in_distribution,split_geo_aware\n"
        "real_videos/a.mp4,t-bone,0,2.5,25,0.3,0.3,0.2,0.2,0.4,0.4,"
        "Taiwan,intersection,normal,day,Good,100,10.0,720,1280,test,test\n",
        encoding="utf-8",
    )
    report_path = tmp_path / "a.json"
    report_path.write_text(json.dumps(_report([False, True, False, False, False])), encoding="utf-8")
    output_path = tmp_path / "evaluation.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_incident_reports.py",
            "--metadata",
            str(metadata),
            "--report",
            str(report_path),
            "--early-tolerance-sec",
            "1.0",
            "--late-tolerance-sec",
            "1.0",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["incident_evaluation"]["event_recall"] == pytest.approx(1.0)
    assert payload["clips"][0]["clip_path"] == "real_videos/a.mp4"
    assert payload["methodology"]["trigger_time"] == "window_end"


def test_evaluate_incident_reports_cli_can_override_persistence_gate(tmp_path) -> None:
    metadata = tmp_path / "metadata-real.csv"
    metadata.write_text(
        "path,type,rollover,accident_time,accident_frame,center_x,center_y,"
        "x1,y1,x2,y2,region,scene_layout,weather,day_time,quality,no_frames,"
        "duration,height,width,split_in_distribution,split_geo_aware\n"
        "real_videos/a.mp4,t-bone,0,2.5,25,0.3,0.3,0.2,0.2,0.4,0.4,"
        "Taiwan,intersection,normal,day,Good,100,10.0,720,1280,test,test\n",
        encoding="utf-8",
    )
    report = _report([False, False, False, False, False])
    report["threshold"] = 0.8
    report_path = tmp_path / "a.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")
    output_path = tmp_path / "evaluation.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_incident_reports.py",
            "--metadata",
            str(metadata),
            "--report",
            str(report_path),
            "--min-positive-windows",
            "1",
            "--early-tolerance-sec",
            "1.0",
            "--late-tolerance-sec",
            "1.0",
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert payload["incident_evaluation"]["event_recall"] == pytest.approx(1.0)
    assert payload["methodology"]["min_positive_windows"] == 1
