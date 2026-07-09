from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from .metrics import evaluate_binary_events
from .vlm_review import VlmReviewResult


class EventValidationLabel(StrEnum):
    VIOLATION = "violation"
    NO_VIOLATION = "no_violation"
    UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class EventValidationRecord:
    event_id: str
    image_path: str
    label: EventValidationLabel
    source_type: str
    is_simulated: bool
    has_red_line: bool
    red_line_visible: str
    plate_redacted: bool
    timestamp_redacted: bool
    redistribution_ok: bool
    notes: str = ""

    @property
    def is_evaluable(self) -> bool:
        return self.label is not EventValidationLabel.UNCERTAIN

    @property
    def expected_violation(self) -> bool:
        if self.label is EventValidationLabel.UNCERTAIN:
            raise ValueError("Uncertain events do not have a binary expected violation value.")
        return self.label is EventValidationLabel.VIOLATION


def load_event_validation_manifest(path: str | Path) -> list[EventValidationRecord]:
    rows: list[EventValidationRecord] = []
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for index, row in enumerate(reader, start=2):
            rows.append(_row_to_record(row, line_number=index))
    return rows


def evaluate_vlm_review_manifest(
    manifest_path: str | Path,
    results_root: str | Path,
    providers: Iterable[str],
) -> dict:
    records = load_event_validation_manifest(manifest_path)
    evaluable_records = [record for record in records if record.is_evaluable]
    results_root_path = Path(results_root)
    provider_reports = {
        provider: _evaluate_provider(evaluable_records, results_root_path, provider)
        for provider in providers
    }
    return {
        "manifest_path": str(manifest_path),
        "results_root": str(results_root),
        "total_events": len(records),
        "evaluated_events": len(evaluable_records),
        "skipped_uncertain_events": len(records) - len(evaluable_records),
        "providers": provider_reports,
    }


def _evaluate_provider(records: list[EventValidationRecord], results_root: Path, provider: str) -> dict:
    y_true: list[bool] = []
    y_pred: list[bool] = []
    result_count = 0
    missing_result_count = 0
    schema_success_count = 0
    human_review_count = 0

    for record in records:
        expected = record.expected_violation
        result_path = results_root / record.event_id / f"{provider}.json"
        if not result_path.exists():
            missing_result_count += 1
            y_true.append(expected)
            y_pred.append(False)
            human_review_count += 1
            continue

        result = VlmReviewResult.from_dict(json.loads(result_path.read_text(encoding="utf-8")))
        result_count += 1
        y_true.append(expected)
        y_pred.append(result.likely_violation)
        if result.human_review_needed:
            human_review_count += 1
        if _schema_succeeded(result):
            schema_success_count += 1

    metrics = evaluate_binary_events(y_true, y_pred)
    denominator = max(1, len(records))
    schema_denominator = max(1, result_count)
    return {
        "provider": provider,
        "evaluated_events": len(records),
        "result_count": result_count,
        "missing_result_count": missing_result_count,
        "metrics": metrics.to_dict(),
        "human_review_rate": human_review_count / denominator,
        "schema_success_rate": schema_success_count / schema_denominator,
    }


def _schema_succeeded(result: VlmReviewResult) -> bool:
    return not any(item.startswith("VLM response could not be normalized") for item in result.missing_evidence)


def _row_to_record(row: dict[str, str], line_number: int) -> EventValidationRecord:
    try:
        label = EventValidationLabel(row["label"].strip())
        return EventValidationRecord(
            event_id=row["event_id"].strip(),
            image_path=row.get("image_path", "").strip(),
            label=label,
            source_type=row.get("source_type", "").strip(),
            is_simulated=_parse_bool(row.get("is_simulated", "")),
            has_red_line=_parse_bool(row.get("has_red_line", "")),
            red_line_visible=row.get("red_line_visible", "").strip(),
            plate_redacted=_parse_bool(row.get("plate_redacted", "")),
            timestamp_redacted=_parse_bool(row.get("timestamp_redacted", "")),
            redistribution_ok=_parse_bool(row.get("redistribution_ok", "")),
            notes=row.get("notes", "").strip(),
        )
    except KeyError as exc:
        raise ValueError(f"Manifest line {line_number} is missing required column {exc}.") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid manifest line {line_number}: {exc}") from exc


def _parse_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y"}:
        return True
    if normalized in {"false", "0", "no", "n", ""}:
        return False
    raise ValueError(f"Unsupported boolean value: {value}")
