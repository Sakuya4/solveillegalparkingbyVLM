import json
import subprocess
import sys
from pathlib import Path

from illegal_parking.vlm_review import (
    VlmReviewRequest,
    compare_vlm_review_results,
    build_redline_parking_review_request,
    parse_vlm_review_result_text,
    review_redline_parking_offline,
)


def test_build_redline_parking_review_request_includes_privacy_instruction(tmp_path):
    evidence = {
        "mask_source": "sam_prompt",
        "footprint_overlap_pixels": 5882,
        "footprint_overlap_ratio": 0.1628,
        "restricted_line_margin_px": 24,
        "artifacts": {
            "original": "original.jpg",
            "bbox_overlay": "bbox_overlay.jpg",
            "overlap_overlay": "overlap_overlay.jpg",
        },
    }

    request = build_redline_parking_review_request(evidence, tmp_path)

    assert request.task == "redline_parking_review"
    assert "Do not infer or recover the license plate" in request.prompt
    assert "0.1628" in request.prompt
    assert str(tmp_path / "overlap_overlay.jpg") in request.image_paths


def test_prepare_vlm_review_cli_writes_request(tmp_path):
    evidence_path = tmp_path / "evidence.json"
    output_path = tmp_path / "vlm_review_request.json"
    evidence_path.write_text(
        json.dumps(
            {
                "mask_source": "bbox",
                "footprint_overlap_pixels": 12,
                "footprint_overlap_ratio": 0.25,
                "artifacts": {"original": "original.jpg", "overlap_overlay": "overlap_overlay.jpg"},
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/prepare_vlm_review.py",
            "--evidence-json",
            str(evidence_path),
            "--image-dir",
            str(tmp_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    request = json.loads(output_path.read_text(encoding="utf-8"))

    assert request["task"] == "redline_parking_review"
    assert request["evidence"]["footprint_overlap_ratio"] == 0.25
    assert "human_review_needed" in request["prompt"]


def test_offline_redline_review_confirms_strong_sam_contact():
    request = VlmReviewRequest(
        task="redline_parking_review",
        prompt="review",
        image_paths=["original.jpg", "overlap_overlay.jpg"],
        evidence={
            "mask_source": "sam_prompt",
            "footprint_overlap_pixels": 5882,
            "footprint_overlap_ratio": 0.1628,
            "restricted_line": [205, 585, 138, 952, 24],
            "restricted_line_margin_px": 24,
            "artifacts": {"original": "original.jpg", "overlap_overlay": "overlap_overlay.jpg"},
        },
    )

    result = review_redline_parking_offline(request)

    assert result.likely_violation is True
    assert result.confidence >= 0.8
    assert result.human_review_needed is False
    assert result.provider == "offline_evidence_reviewer"


def test_offline_redline_review_flags_bbox_only_as_human_review():
    request = VlmReviewRequest(
        task="redline_parking_review",
        prompt="review",
        image_paths=["original.jpg"],
        evidence={
            "mask_source": "bbox",
            "footprint_overlap_pixels": 8,
            "footprint_overlap_ratio": 0.04,
            "artifacts": {"original": "original.jpg"},
        },
    )

    result = review_redline_parking_offline(request)

    assert result.likely_violation is False
    assert result.human_review_needed is True
    assert any("bbox-only" in item for item in result.missing_evidence)


def test_run_vlm_review_cli_writes_result(tmp_path):
    request_path = tmp_path / "vlm_review_request.json"
    output_path = tmp_path / "vlm_review_result.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "redline_parking_review",
                "prompt": "review",
                "image_paths": ["original.jpg", "overlap_overlay.jpg"],
                "evidence": {
                    "mask_source": "sam_prompt",
                    "footprint_overlap_pixels": 30,
                    "footprint_overlap_ratio": 0.12,
                    "restricted_line": [1, 2, 3, 4, 5],
                    "artifacts": {"original": "original.jpg", "overlap_overlay": "overlap_overlay.jpg"},
                },
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_vlm_review.py",
            "--request-json",
            str(request_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["likely_violation"] is True
    assert result["provider"] == "offline_evidence_reviewer"
    assert "visual_reasons" in result


def test_parse_vlm_review_result_text_normalizes_json_fence():
    result = parse_vlm_review_result_text(
        """```json
        {
          "likely_violation": true,
          "confidence": 0.77,
          "visual_reasons": ["front wheel is beside the red line"],
          "missing_evidence": [],
          "human_review_needed": false
        }
        ```""",
        provider="sample_vlm",
    )

    assert result.likely_violation is True
    assert result.confidence == 0.77
    assert result.provider == "sample_vlm"


def test_parse_vlm_review_result_text_handles_string_booleans():
    result = parse_vlm_review_result_text(
        '{"likely_violation": "false", "confidence": "0.33", "visual_reasons": [], "missing_evidence": [], "human_review_needed": "true"}',
        provider="sample_vlm",
    )

    assert result.likely_violation is False
    assert result.confidence == 0.33
    assert result.human_review_needed is True


def test_run_vlm_review_cli_accepts_vlm_json_response(tmp_path):
    request_path = tmp_path / "vlm_review_request.json"
    response_path = tmp_path / "vlm_response.txt"
    output_path = tmp_path / "vlm_review_result.json"
    request_path.write_text(
        json.dumps(
            {
                "task": "redline_parking_review",
                "prompt": "review",
                "image_paths": ["original.jpg"],
                "evidence": {"artifacts": {"original": "original.jpg"}},
            }
        ),
        encoding="utf-8",
    )
    response_path.write_text(
        json.dumps(
            {
                "likely_violation": False,
                "confidence": 0.42,
                "visual_reasons": ["red line contact is unclear"],
                "missing_evidence": ["overlap overlay"],
                "human_review_needed": True,
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/run_vlm_review.py",
            "--request-json",
            str(request_path),
            "--output",
            str(output_path),
            "--provider",
            "vlm-json",
            "--vlm-response-text",
            str(response_path),
            "--vlm-provider-name",
            "sample_vlm",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    result = json.loads(output_path.read_text(encoding="utf-8"))

    assert result["likely_violation"] is False
    assert result["provider"] == "sample_vlm"


def test_compare_vlm_review_results_reports_agreement():
    baseline = parse_vlm_review_result_text(
        '{"likely_violation": true, "confidence": 0.85, "visual_reasons": [], "missing_evidence": [], "human_review_needed": false}',
        provider="offline_evidence_reviewer",
    )
    model = parse_vlm_review_result_text(
        '{"likely_violation": true, "confidence": 0.73, "visual_reasons": [], "missing_evidence": [], "human_review_needed": true}',
        provider="qwen_vl",
    )

    rows = compare_vlm_review_results([baseline, model])

    assert rows[0]["provider"] == "offline_evidence_reviewer"
    assert rows[0]["agrees_with_baseline"] is True
    assert rows[1]["provider"] == "qwen_vl"
    assert rows[1]["confidence_delta_from_baseline"] == -0.12


def test_compare_vlm_reviews_cli_writes_report(tmp_path):
    offline_path = tmp_path / "offline.json"
    model_path = tmp_path / "model.json"
    output_path = tmp_path / "comparison.json"
    offline_path.write_text(
        json.dumps(
            {
                "likely_violation": True,
                "confidence": 0.85,
                "visual_reasons": [],
                "missing_evidence": [],
                "human_review_needed": False,
                "provider": "offline_evidence_reviewer",
            }
        ),
        encoding="utf-8",
    )
    model_path.write_text(
        json.dumps(
            {
                "likely_violation": False,
                "confidence": 0.51,
                "visual_reasons": [],
                "missing_evidence": ["unclear curb"],
                "human_review_needed": True,
                "provider": "blip2_local",
            }
        ),
        encoding="utf-8",
    )

    subprocess.run(
        [
            sys.executable,
            "scripts/compare_vlm_reviews.py",
            "--result-json",
            str(offline_path),
            "--result-json",
            str(model_path),
            "--output",
            str(output_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(output_path.read_text(encoding="utf-8"))

    assert report["baseline_provider"] == "offline_evidence_reviewer"
    assert report["rows"][1]["provider"] == "blip2_local"
    assert report["rows"][1]["agrees_with_baseline"] is False
