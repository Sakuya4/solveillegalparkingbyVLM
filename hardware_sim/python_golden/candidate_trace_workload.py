from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np


def load_candidate_trace(path: str | Path) -> list[dict]:
    trace_path = Path(path)
    records: list[dict] = []
    with trace_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                available = float(record["candidate_available"])
                score = float(record["candidate_score"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid candidate trace at line {line_number}: {exc}") from exc
            if available not in (0.0, 1.0) or not math.isfinite(score) or not 0.0 <= score <= 1.0:
                raise ValueError(f"Invalid candidate trace values at line {line_number}")
            records.append(record)
    if not records:
        raise ValueError("Candidate trace must contain at least one record")
    return records


def build_candidate_flags(records: list[dict], score_threshold: float) -> np.ndarray:
    if not 0.0 <= score_threshold <= 1.0:
        raise ValueError("score_threshold must be between zero and one")
    return np.asarray(
        [
            bool(float(record["candidate_available"]))
            and float(record["candidate_score"]) >= score_threshold
            for record in records
        ],
        dtype=np.bool_,
    )


def repeat_candidate_flags(candidate_flags, camera_count: int) -> np.ndarray:
    if camera_count <= 0:
        raise ValueError("camera_count must be positive")
    flags = np.asarray(candidate_flags, dtype=np.bool_).reshape(-1)
    return np.repeat(flags, camera_count)
