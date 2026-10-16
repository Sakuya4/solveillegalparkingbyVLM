from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from illegal_parking.incident_vlm_evidence import select_review_context, write_clean_review_package, review_disposition
from scripts.prepare_incident_vlm_review import sampled_source_indices


def test_select_review_context_uses_first_confirmed_window() -> None:
    selection = select_review_context({
        "window_starts": [0, 15, 30, 45],
        "window_probabilities": [0.2, 0.9, 0.95, 0.1],
        "window_review_flags": [False, False, True, False],
        "sequence_steps": 15,
        "sampled_frames": 70,
    })

    assert selection.window_index == 2
    assert selection.frame_indices == (23, 38, 53)
    assert selection.model_probability == pytest.approx(0.95)
    assert selection.consecutive_positive_windows == 2


def test_select_review_context_rejects_report_without_review_event() -> None:
    with pytest.raises(ValueError, match="confirmed review window"):
        select_review_context({
            "window_starts": [0, 15],
            "window_probabilities": [0.2, 0.9],
            "window_review_flags": [False, False],
            "sequence_steps": 15,
            "sampled_frames": 31,
        })


def test_prepare_incident_vlm_review_cli_writes_privacy_evidence(tmp_path) -> None:
    video_path = tmp_path / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        7.5,
        (64, 48),
    )
    assert writer.isOpened()
    for frame_index in range(70):
        frame = np.full((48, 64, 3), frame_index, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    report_path = tmp_path / "report.json"
    report_path.write_text(
        json.dumps({
            "annotation_free_inference": True,
            "window_starts": [0, 15, 30, 45],
            "window_probabilities": [0.2, 0.9, 0.95, 0.1],
            "window_review_flags": [False, False, True, False],
            "sequence_steps": 15,
            "sampled_frames": 70,
            "threshold": 0.85,
            "min_positive_windows": 2,
            "privacy": "Plate regions are blurred.",
        }),
        encoding="utf-8",
    )
    output_dir = tmp_path / "review"

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_incident_vlm_review.py",
            "--inference-report",
            str(report_path),
            "--annotated-video",
            str(video_path),
            "--output-dir",
            str(output_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    evidence = json.loads((output_dir / "evidence.json").read_text(encoding="utf-8"))
    request = json.loads((output_dir / "vlm_review_request.json").read_text(encoding="utf-8"))

    assert evidence["privacy_redacted"] is True
    assert evidence["frame_indices"] == [23, 38, 53]
    assert request["task"] == "traffic_incident_review"
    assert all(
        (output_dir / name).is_file()
        for name in ("before.jpg", "trigger.jpg", "after.jpg")
    )


def test_sampled_source_indices_match_video_sampling_contract() -> None:
    indices = sampled_source_indices(16, source_fps=8.0, target_fps=4.0)

    assert indices == list(range(0, 16, 2))


def test_clean_review_package_preserves_order_and_hides_model_scores(tmp_path):
    frames = tuple(np.full((24, 32, 3), value, dtype=np.uint8) for value in (10, 100, 200))
    paths = write_clean_review_package(frames, (1.0, 3.0, 5.0), tmp_path, privacy_method="Vehicle lower-region blur")
    request = json.loads((tmp_path / "vlm_review_request.json").read_text(encoding="utf-8"))
    assert "model_probability" not in request["evidence"]
    assert request["evidence"]["evidence_available_time_sec"] == 5.0
    assert [path.rsplit("/", 1)[-1].rsplit("\\", 1)[-1] for path in request["image_paths"]] == ["before.jpg", "trigger.jpg", "after.jpg"]
    assert cv2.imread(paths["contact_sheet"]).shape == (24, 96, 3)
    with pytest.raises(ValueError, match="ordered"):
        write_clean_review_package(frames, (5.0, 3.0, 1.0), tmp_path, privacy_method="blur")


def test_visual_disagreement_retains_temporal_event_for_review():
    review = {"schema_valid": True, "likely_violation": False, "confidence": 0.9, "human_review_needed": False}
    assert review_disposition(True, review) == "temporal_visual_disagreement"
    assert review_disposition(False, review) == "consistent_background"
    assert review_disposition(True, {**review, "schema_valid": False}) == "response_contract_review"
    assert review_disposition(True, {**review, "likely_violation": True, "confidence": 0.3}) == "low_confidence_review"


def test_visual_agreement_does_not_override_missing_evidence_or_requested_review():
    review = {"schema_valid": True, "likely_violation": True, "confidence": 0.95, "visual_reasons": ["Visible contact"], "missing_evidence": [], "human_review_needed": False}
    assert review_disposition(True, review) == "consistent_visual_support"
    assert review_disposition(True, {**review, "missing_evidence": ["Collision moment absent"]}) == "incomplete_visual_evidence"
    assert review_disposition(True, {**review, "human_review_needed": True}) == "operator_review_requested"
    assert review_disposition(True, {**review, "visual_reasons": []}) == "missing_visual_rationale"
    assert review_disposition(True, {**review, "confidence": float("nan")}) == "response_contract_review"
    assert review_disposition(True, {**review, "confidence": 1.1}) == "response_contract_review"
    assert review_disposition(True, {**review, "confidence": "invalid"}) == "response_contract_review"
