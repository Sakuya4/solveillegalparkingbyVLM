import csv
import json
import subprocess
import sys

from illegal_parking.event_validation import (
    EventValidationLabel,
    evaluate_vlm_review_manifest,
    load_event_validation_manifest,
)


def test_load_event_validation_manifest_reads_real_and_simulated_rows(tmp_path):
    manifest_path = tmp_path / "events_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "event_id",
                "image_path",
                "label",
                "source_type",
                "is_simulated",
                "has_red_line",
                "red_line_visible",
                "plate_redacted",
                "timestamp_redacted",
                "redistribution_ok",
                "notes",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "event_id": "tw_redline_001",
                "image_path": "images/tw_redline_001.jpg",
                "label": "violation",
                "source_type": "self_collected",
                "is_simulated": "false",
                "has_red_line": "true",
                "red_line_visible": "clear",
                "plate_redacted": "true",
                "timestamp_redacted": "true",
                "redistribution_ok": "true",
                "notes": "user photo",
            }
        )
        writer.writerow(
            {
                "event_id": "sim_no_redline_001",
                "image_path": "",
                "label": "no_violation",
                "source_type": "simulated",
                "is_simulated": "yes",
                "has_red_line": "no",
                "red_line_visible": "none",
                "plate_redacted": "true",
                "timestamp_redacted": "true",
                "redistribution_ok": "false",
                "notes": "synthetic negative",
            }
        )

    rows = load_event_validation_manifest(manifest_path)

    assert rows[0].event_id == "tw_redline_001"
    assert rows[0].label == EventValidationLabel.VIOLATION
    assert rows[0].has_red_line is True
    assert rows[1].is_simulated is True
    assert rows[1].has_red_line is False


def test_evaluate_vlm_review_manifest_skips_uncertain_and_reports_provider_metrics(tmp_path):
    manifest_path = tmp_path / "events_manifest.csv"
    manifest_path.write_text(
        "\n".join(
            [
                "event_id,image_path,label,source_type,is_simulated,has_red_line,red_line_visible,plate_redacted,timestamp_redacted,redistribution_ok,notes",
                "ev1,,violation,simulated,true,true,clear,true,true,false,positive",
                "ev2,,no_violation,simulated,true,false,none,true,true,false,negative",
                "ev3,,uncertain,simulated,true,true,occluded,true,true,false,skip",
            ]
        ),
        encoding="utf-8",
    )
    results_root = tmp_path / "review_results"
    _write_result(results_root, "ev1", "offline_evidence_reviewer", True, 0.8, False)
    _write_result(results_root, "ev2", "offline_evidence_reviewer", True, 0.6, True)
    _write_result(results_root, "ev1", "qwen2_5_vl_3b", True, 0.95, False)
    _write_result(results_root, "ev2", "qwen2_5_vl_3b", False, 0.72, False)
    _write_result(
        results_root,
        "ev1",
        "small_vlm",
        False,
        0.0,
        True,
        missing_evidence=["VLM response could not be normalized: invalid json"],
    )

    report = evaluate_vlm_review_manifest(
        manifest_path=manifest_path,
        results_root=results_root,
        providers=["offline_evidence_reviewer", "qwen2_5_vl_3b", "small_vlm"],
    )

    assert report["evaluated_events"] == 2
    assert report["skipped_uncertain_events"] == 1
    offline = report["providers"]["offline_evidence_reviewer"]
    assert offline["metrics"]["tp"] == 1
    assert offline["metrics"]["fp"] == 1
    assert offline["human_review_rate"] == 0.5
    qwen = report["providers"]["qwen2_5_vl_3b"]
    assert qwen["metrics"]["precision"] == 1.0
    assert qwen["schema_success_rate"] == 1.0
    small = report["providers"]["small_vlm"]
    assert small["missing_result_count"] == 1
    assert small["schema_success_rate"] == 0.0


def test_simulated_validation_cli_outputs_manifest_and_evaluation_report(tmp_path):
    sim_dir = tmp_path / "sim_validation"
    report_path = tmp_path / "report.json"

    subprocess.run(
        [
            sys.executable,
            "scripts/create_simulated_validation_set.py",
            "--output-dir",
            str(sim_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            "scripts/evaluate_vlm_validation.py",
            "--manifest",
            str(sim_dir / "events_manifest.csv"),
            "--results-root",
            str(sim_dir / "review_results"),
            "--provider",
            "offline_evidence_reviewer",
            "--provider",
            "qwen2_5_vl_3b",
            "--provider",
            "small_vlm",
            "--output",
            str(report_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert report["total_events"] == 5
    assert report["evaluated_events"] == 4
    assert report["providers"]["qwen2_5_vl_3b"]["metrics"]["precision"] == 1.0
    assert report["providers"]["small_vlm"]["schema_success_rate"] == 0.0


def _write_result(
    results_root,
    event_id,
    provider,
    likely_violation,
    confidence,
    human_review_needed,
    missing_evidence=None,
):
    event_dir = results_root / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    (event_dir / f"{provider}.json").write_text(
        json.dumps(
            {
                "likely_violation": likely_violation,
                "confidence": confidence,
                "visual_reasons": [],
                "missing_evidence": missing_evidence or [],
                "human_review_needed": human_review_needed,
                "provider": provider,
            }
        ),
        encoding="utf-8",
    )
