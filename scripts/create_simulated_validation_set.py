from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PROVIDERS = ("offline_evidence_reviewer", "qwen2_5_vl_3b", "small_vlm")


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a small simulated event validation set.")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    results_root = output_dir / "review_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "events_manifest.csv"
    rows = _rows()

    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        if row["label"] == "uncertain":
            continue
        _write_provider_results(results_root, row["event_id"], row["label"])

    inventory = {
        "manifest": str(manifest_path),
        "results_root": str(results_root),
        "providers": list(PROVIDERS),
        "note": "Simulated validation data is for workflow and stress testing, not final accuracy claims.",
    }
    (output_dir / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(inventory, ensure_ascii=False, indent=2))
    return 0


def _rows() -> list[dict[str, str]]:
    return [
        _row("sim_redline_violation_001", "violation", True, "clear", "synthetic red-line positive"),
        _row("sim_redline_violation_002", "violation", True, "faded", "synthetic faded-red-line positive"),
        _row("sim_redline_negative_001", "no_violation", True, "clear", "vehicle near but outside red-line band"),
        _row("sim_no_redline_negative_001", "no_violation", False, "none", "vehicle without red-line context"),
        _row("sim_occluded_uncertain_001", "uncertain", True, "occluded", "occluded red-line case for skip behavior"),
    ]


def _row(event_id: str, label: str, has_red_line: bool, red_line_visible: str, notes: str) -> dict[str, str]:
    return {
        "event_id": event_id,
        "image_path": "",
        "label": label,
        "source_type": "simulated",
        "is_simulated": "true",
        "has_red_line": str(has_red_line).lower(),
        "red_line_visible": red_line_visible,
        "plate_redacted": "true",
        "timestamp_redacted": "true",
        "redistribution_ok": "false",
        "notes": notes,
    }


def _write_provider_results(results_root: Path, event_id: str, label: str) -> None:
    expected_violation = label == "violation"
    event_dir = results_root / event_id
    event_dir.mkdir(parents=True, exist_ok=True)
    _write_result(event_dir, "offline_evidence_reviewer", expected_violation, 0.82, False)
    _write_result(event_dir, "qwen2_5_vl_3b", expected_violation, 0.9, False)
    _write_result(
        event_dir,
        "small_vlm",
        False,
        0.0,
        True,
        missing_evidence=["VLM response could not be normalized: simulated schema failure"],
    )


def _write_result(
    event_dir: Path,
    provider: str,
    likely_violation: bool,
    confidence: float,
    human_review_needed: bool,
    missing_evidence: list[str] | None = None,
) -> None:
    (event_dir / f"{provider}.json").write_text(
        json.dumps(
            {
                "likely_violation": likely_violation,
                "confidence": confidence,
                "visual_reasons": [],
                "missing_evidence": missing_evidence or [],
                "human_review_needed": human_review_needed,
                "provider": provider,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
