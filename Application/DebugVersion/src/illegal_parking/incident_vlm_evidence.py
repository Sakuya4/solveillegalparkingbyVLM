from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
