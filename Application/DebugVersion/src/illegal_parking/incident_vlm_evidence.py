from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any


def review_disposition(temporal_positive: bool, review: dict[str, Any]) -> str:
    if not review.get("schema_valid", False):
        return "response_contract_review"
    if bool(review["likely_violation"]) != temporal_positive:
        return "temporal_visual_disagreement"
    if float(review["confidence"]) < 0.7:
        return "low_confidence_review"
    return "corroborated_incident" if temporal_positive else "consistent_background"


def write_clean_review_package(
    frames: tuple[Any, Any, Any], timestamps_sec: tuple[float, float, float],
    output_dir: Path, *, privacy_method: str,
) -> dict[str, str]:
    """Export already-redacted frames without model-score overlays."""
    import cv2
    import numpy as np
    from .vlm_review import build_traffic_incident_review_request

    if len(frames) != 3 or len(timestamps_sec) != 3:
        raise ValueError("Review evidence requires three ordered frames and timestamps")
    if any(not math.isfinite(time) or time < 0 for time in timestamps_sec) or list(timestamps_sec) != sorted(timestamps_sec):
        raise ValueError("Evidence timestamps must be finite, nonnegative, and ordered")
    if not privacy_method.strip():
        raise ValueError("A privacy method must be recorded")
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts = {name: f"{name}.jpg" for name in ("before", "trigger", "after")}
    for frame, name in zip(frames, artifacts.values()):
        if not cv2.imwrite(str(output_dir / name), frame):
            raise OSError(f"Could not write evidence frame: {name}")
    contact_sheet = output_dir / "contact_sheet.jpg"
    if not cv2.imwrite(str(contact_sheet), np.concatenate(frames, axis=1)):
        raise OSError("Could not write evidence contact sheet")
    evidence = {
        "event_type": "traffic_incident", "timestamps_sec": list(timestamps_sec),
        "evidence_available_time_sec": max(timestamps_sec),
        "privacy_redacted": True, "privacy_method": privacy_method,
        "artifacts": artifacts,
    }
    request = build_traffic_incident_review_request(evidence, output_dir.resolve(), include_model_context=False)
    request_path = output_dir / "vlm_review_request.json"
    request_path.write_text(json.dumps(request.to_dict(), indent=2) + "\n", encoding="utf-8")
    (output_dir / "evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return {"request_json": str(request_path.resolve()), "contact_sheet": str(contact_sheet.resolve()), "evidence_json": str((output_dir / "evidence.json").resolve())}


@dataclass(frozen=True)
class IncidentReviewContext:
    window_index: int
    frame_indices: tuple[int, int, int]
    model_probability: float
    consecutive_positive_windows: int


def select_review_context(report: dict[str, Any]) -> IncidentReviewContext:
    starts = [int(value) for value in report.get("window_starts", [])]
    probabilities = [float(value) for value in report.get("window_probabilities", [])]
    review_flags = [bool(value) for value in report.get("window_review_flags", [])]
    if not starts or len(starts) != len(probabilities) or len(starts) != len(review_flags):
        raise ValueError("Inference report window arrays must be non-empty and have equal lengths.")

    try:
        window_index = review_flags.index(True)
    except ValueError as exc:
        raise ValueError("Inference report does not contain a confirmed review window.") from exc

    sequence_steps = int(report.get("sequence_steps", 0))
    sampled_frames = int(report.get("sampled_frames", 0))
    if sequence_steps <= 0 or sampled_frames <= 0:
        raise ValueError("sequence_steps and sampled_frames must be positive.")

    threshold = float(report.get("threshold", 0.5))
    consecutive_positive_windows = 0
    for probability in reversed(probabilities[: window_index + 1]):
        if probability < threshold:
            break
        consecutive_positive_windows += 1

    context_indices = (
        max(0, window_index - 1),
        window_index,
        min(len(starts) - 1, window_index + 1),
    )
    frame_indices = tuple(
        min(sampled_frames - 1, starts[index] + 1 + sequence_steps // 2)
        for index in context_indices
    )
    return IncidentReviewContext(
        window_index=window_index,
        frame_indices=frame_indices,
        model_probability=probabilities[window_index],
        consecutive_positive_windows=consecutive_positive_windows,
    )
