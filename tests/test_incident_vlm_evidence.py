from __future__ import annotations

import json
import subprocess
import sys

import cv2
import numpy as np
import pytest

from illegal_parking.incident_vlm_evidence import select_review_context


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
    assert all((output_dir / name).is_file() for name in ("before.jpg", "trigger.jpg", "after.jpg"))
