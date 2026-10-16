from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from scripts.replay_presentation_events import load_event_protocol, validate_batch
from scripts.evaluate_presentation_events import model_reports
from scripts.audit_presentation_event_predictions import clock_control
from illegal_parking.incident_presentation_events import evaluate_presentation_event


def outcome_rows(protocol, selection, reports, cohort):
    rows = []
    methods = [(key, definition["split"], definition["arm"], definition["feature"],
                selection["policies"][key]["selected"]["threshold"] if selection["policies"][key]["selected"] else None,
                model_reports(protocol, reports, key, cohort)) for key, definition in protocol["models"].items()]
    for split in protocol["cohorts"]:
        key = next(key for key, definition in protocol["models"].items() if definition["split"] == split)
        selected_reports = model_reports(protocol, reports, key, cohort)
        for positive, name in ((True, "clock_only_positive"), (False, "always_negative")):
            methods.append((name, split, "control", "none", .5, [clock_control(report, positive) for report in selected_reports]))
    for key, split, arm, feature, threshold, selected_reports in methods:
        for report in selected_reports:
            annotation = protocol["annotations"][report["path"]]
            outcome = evaluate_presentation_event(report, annotation, threshold) if threshold is not None else None
            rows.append({"cohort": cohort, "split": split, "method": key, "training_arm": arm, "features": feature,
                         "path": report["path"], "source_sha256": report["source_sha256"],
                         "annotation_pts_sec": annotation["event_time_sec"], "threshold": threshold,
                         "valid_contexts": len(report["window_starts"]), "rejected_contexts": len(report["rejected_window_starts"]),
                         "detected": int(outcome.detected) if outcome else None,
                         "trigger_pts_sec": outcome.trigger_time_sec if outcome else None,
                         "signed_delay_sec": outcome.trigger_delay_sec if outcome else None,
                         "early_episodes": int(outcome.early_alert_episodes) if outcome else None,
                         "late_episodes": int(outcome.late_alert_episodes) if outcome else None,
                         "all_episodes": outcome.alert_episode_count if outcome else None})
    return rows


def main():
    protocol_path = ROOT / "docs/results/presentation_event_protocol.json"
    protocol = load_event_protocol(protocol_path)
    selection_path = ROOT / "outputs/accident/presentation_event_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    audit_path = ROOT / "docs/results/presentation_event_controls.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("The completed prediction/control audit must match this protocol")
    output = ROOT / "docs/results/presentation_event_outcomes.csv"
    manifest = output.with_suffix(".json")
    if output.exists() or manifest.exists():
        raise FileExistsError("Published event traces are immutable")
    rows = []
    for cohort in ("calibration", "test"):
        directory = ROOT / f"outputs/accident/presentation_events_{cohort}"
        _, reports = validate_batch(directory, protocol_path, protocol, cohort, sha256(selection_path) if cohort == "test" else None)
        if audit[cohort + "_batch_sha256"] != sha256(directory / "batch.json"):
            raise ValueError("The probability audit does not bind this batch")
        rows.extend(outcome_rows(protocol, selection, reports, cohort))
    if len({(row["cohort"], row["split"], row["method"], row["path"]) for row in rows}) != len(rows):
        raise ValueError("Published outcomes must have unique cohort/split/method/clip identities")
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    record = {"protocol_sha256": sha256(protocol_path), "selection_sha256": sha256(selection_path),
              "prediction_audit_sha256": sha256(audit_path), "outcomes_sha256": sha256(output), "rows": len(rows),
              "publisher_sha256": sha256(Path(__file__)),
              "model_sha256": {key: definition["sha256"] for key, definition in protocol["models"].items()},
              "clock": "Normalized video presentation timestamps; supplied annotation frame, no legacy clamp",
              "missing_values": "Empty delay/trigger fields indicate no matched detection; empty outcome fields with empty threshold indicate an infeasible policy, not a negative prediction",
              "scope": "Calibration and test accident clips, source identities retained; not reviewed-normal CCTV exposure"}
    manifest.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"rows": len(rows), "outcomes_sha256": record["outcomes_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
