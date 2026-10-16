from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_presentation_events as frozen
from scripts.prepare_presentation_protocol import sha256


class ScalarJSON:
    loads = staticmethod(json.loads)

    @staticmethod
    def dumps(value, *args, **kwargs):
        def scalar(item):
            if isinstance(item, np.generic):
                return item.item()
            raise TypeError(f"Unsupported JSON value: {type(item).__name__}")

        return json.dumps(value, *args, default=scalar, allow_nan=False, **kwargs)


def amendment_path():
    return ROOT / "docs/results/presentation_event_serialization.json"


def validate_amendment():
    record = json.loads(amendment_path().read_text(encoding="utf-8"))
    checks = {"adapter_sha256": Path(__file__),
              "frozen_reporter_sha256": ROOT / "scripts/evaluate_presentation_events.py",
              "protocol_sha256": ROOT / "docs/results/presentation_event_protocol.json",
              "calibration_batch_sha256": ROOT / "outputs/accident/presentation_events_calibration/batch.json"}
    if any(sha256(path) != record[key] for key, path in checks.items()):
        raise ValueError("The serialization amendment no longer matches its frozen inputs")
    return record


def freeze_amendment():
    path = amendment_path()
    if path.exists():
        raise FileExistsError("Serialization amendments are immutable")
    protocol_path = ROOT / "docs/results/presentation_event_protocol.json"
    protocol = frozen.load_event_protocol(protocol_path)
    directory = ROOT / "outputs/accident/presentation_events_calibration"
    frozen.validate_batch(directory, protocol_path, protocol, "calibration")
    if (ROOT / "outputs/accident/presentation_event_selection.json").exists() or (ROOT / "outputs/accident/presentation_events_test/batch.json").exists():
        raise ValueError("The format-only amendment must precede selection and test replay")
    record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "kind": "numpy_scalar_serialization_only",
              "adapter_sha256": sha256(Path(__file__)), "frozen_reporter_sha256": sha256(ROOT / "scripts/evaluate_presentation_events.py"),
              "protocol_sha256": sha256(protocol_path), "calibration_batch_sha256": sha256(directory / "batch.json"),
              "original_failure_log_sha256": sha256(ROOT / "outputs/research_runs/pts_event_threshold_selection_initial.log"),
              "unchanged": ["models", "source clips", "annotation clocks", "features", "probabilities", "threshold grid", "selection rule", "endpoints"],
              "behavior": "Convert NumPy scalar values to their identical Python scalar representation during JSON encoding; reject nonfinite values; no arrays or scientific settings are changed"}
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(record, indent=2))


def main():
    if sys.argv[1:] == ["--freeze"]:
        freeze_amendment()
        return
    validate_amendment()
    original = frozen.json
    try:
        # Replace only this reporter's JSON interface, not the shared json module.
        frozen.json = ScalarJSON
        frozen.main()
    finally:
        frozen.json = original


if __name__ == "__main__":
    main()
