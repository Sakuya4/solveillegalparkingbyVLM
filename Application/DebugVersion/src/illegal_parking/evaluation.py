from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .metrics import BinaryEventMetrics, evaluate_binary_events


@dataclass(frozen=True)
class FrameAnnotation:
    frame_index: int
    expected_violation: bool


@dataclass(frozen=True)
class FrameEvaluationResult:
    metrics: BinaryEventMetrics
    evaluated_frames: int
    predicted_event_frames: list[int]

    def to_dict(self) -> dict:
        return {
            "evaluated_frames": self.evaluated_frames,
            "predicted_event_frames": self.predicted_event_frames,
            "metrics": self.metrics.to_dict(),
        }


def load_frame_annotations(path: str | Path) -> list[FrameAnnotation]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw.get("frames", raw) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError("Annotation file must be a list or an object with a 'frames' list.")

    annotations: list[FrameAnnotation] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Annotation row {index} must be an object.")
        if "frame_index" not in row or "expected_violation" not in row:
            raise ValueError(f"Annotation row {index} requires frame_index and expected_violation.")
        annotations.append(
            FrameAnnotation(
                frame_index=int(row["frame_index"]),
                expected_violation=bool(row["expected_violation"]),
            )
        )
    return annotations


def load_predicted_event_frames(path: str | Path) -> set[int]:
    events_path = Path(path)
    if not events_path.exists():
        return set()

    frames: set[int] = set()
    for line_number, line in enumerate(events_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        if _is_candidate_event(record):
            try:
                frames.add(int(record["frame_index"]))
            except KeyError as exc:
                raise ValueError(f"Event line {line_number} is missing frame_index.") from exc
    return frames


def evaluate_frame_events(
    annotations: Iterable[FrameAnnotation],
    predicted_event_frames: set[int],
) -> FrameEvaluationResult:
    annotation_list = list(annotations)
    y_true = [row.expected_violation for row in annotation_list]
    y_pred = [row.frame_index in predicted_event_frames for row in annotation_list]

    return FrameEvaluationResult(
        metrics=evaluate_binary_events(y_true, y_pred),
        evaluated_frames=len(annotation_list),
        predicted_event_frames=sorted(predicted_event_frames),
    )


def _is_candidate_event(record: dict) -> bool:
    if "is_candidate" in record:
        return bool(record["is_candidate"])
    return record.get("status") == "CANDIDATE"
