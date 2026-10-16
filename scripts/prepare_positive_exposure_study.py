from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from scripts.extract_presentation_features import load_manifest
from scripts.replay_presentation_events import load_event_protocol, validate_batch
from scripts.evaluate_presentation_events import model_reports
from scripts.run_presentation_event_reporting import ScalarJSON
from illegal_parking.incident_presentation_events import presentation_operating_summary
from illegal_parking.incident_positive_exposure import pre_event_positive_contexts, select_positive_exposure_policy


def replay_inventory(root, media):
    names = {row["path"] for row in media}
    hashes = {row["sha256"]: row["path"] for row in media}
    basenames = {Path(name).name: name for name in names}
    inventory, excluded = [], set()
    for directory in ("outputs/accident", "outputs/desktop", "outputs/streaming"):
        for path in sorted((root / directory).rglob("*.json")):
            report = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(report, dict) or report.get("annotation_free_inference") is not True or not ("window_probabilities" in report or "scores" in report):
                continue
            source = str(report.get("path", report.get("input", ""))).replace("\\", "/")
            name = basenames.get(source.rsplit("/", 1)[-1]) or hashes.get(report.get("source_sha256", report.get("input_sha256")))
            if name:
                excluded.add(name)
                inventory.append({"report": path.relative_to(root).as_posix(), "sha256": sha256(path), "clip": name})
    return sorted(excluded), inventory


def choose_new_event_cohort(rows, scheme, excluded, count=100):
    eligible = {row["path"] for row in rows if row[scheme + "_partition"] == "test" and row["path"] not in excluded}
    ordered = sorted(eligible, key=lambda name: hashlib.sha256(f"positive-exposure-v1:42:{scheme}:{name}".encode()).hexdigest())
    if len(ordered) < count:
        raise ValueError("Too few unreplayed sources; do not reduce the frozen cohort size silently")
    return sorted(ordered[:count]), len(ordered)


def main():
    output = ROOT / "docs/results/positive_exposure_protocol.json"
    selection_path = ROOT / "docs/results/positive_exposure_selection.json"
    if output.exists() or selection_path.exists():
        raise FileExistsError("Positive-exposure studies are immutable")
    parent_path = ROOT / "docs/results/presentation_protocol.json"
    parent, rows = load_manifest(parent_path)
    event_path = ROOT / "docs/results/presentation_event_protocol.json"
    event = load_event_protocol(event_path)
    original_selection_path = ROOT / "docs/results/presentation_event_selection.json"
    original_selection = json.loads(original_selection_path.read_text(encoding="utf-8"))
    if original_selection["protocol_sha256"] != sha256(event_path):
        raise ValueError("Original onset-only policies no longer match the parent study")
    directory = ROOT / "outputs/accident/presentation_events_calibration"
    _, reports = validate_batch(directory, event_path, event, "calibration")
    history, inventory = replay_inventory(ROOT, parent["input_media"])
    excluded = set(history)
    for definition in event["cohorts"].values():
        excluded.update(definition["calibration_clip_paths"])
        excluded.update(definition["test_clip_paths"])
    cohorts, eligible_counts = {}, {}
    for scheme in ("iid", "geographic"):
        chosen, count = choose_new_event_cohort(rows, scheme, excluded)
        cohorts[scheme] = {"calibration_clip_paths": event["cohorts"][scheme]["calibration_clip_paths"], "test_clip_paths": chosen}
        eligible_counts[scheme] = count
    settings = {"episode_budget_per_clip": .5, "macro_pre_event_positive_cap": .2, "minimum_supported_calibration_clips": 20,
                "pre_event_definition": "Valid context end PTS strictly less than supplied event PTS minus 1 second",
                "aggregation": "Equal weight per clip with at least one eligible context; unsupported clips are disclosed, not assigned zero exposure",
                "selection_rule": "Maximize training-holdout event recall subject to both constraints; ties minimize off-target episodes then maximize threshold; insufficient support or infeasibility remains unselected"}
    policies = {}
    for key, definition in event["models"].items():
        chosen_reports = model_reports(event, reports, key, "calibration")
        thresholds = sorted(set([*event["calibration"]["threshold_grid"], definition["window_threshold"]]))
        curve = [{"threshold": threshold,
                  "summary": presentation_operating_summary(chosen_reports, event["annotations"], threshold),
                  "pre_event": pre_event_positive_contexts(chosen_reports, event["annotations"], threshold)} for threshold in thresholds]
        selected = select_positive_exposure_policy(curve)
        policies[key] = {"selected": selected,
                         "onset_only_threshold": original_selection["policies"][key]["selected"]["threshold"] if original_selection["policies"][key]["selected"] else None,
                         "curve": curve}
    needed = {name for definition in cohorts.values() for names in definition.values() for name in names}
    annotations = {row["path"]: {key: row[key] for key in ("path", "supplied_event_frame", "event_time_sec", "collision_type")} for row in rows if row["path"] in needed}
    implementations = dict(event["implementation_sha256"])
    implementations.update({name: sha256(ROOT / name) for name in (
        "scripts/prepare_positive_exposure_study.py", "Application/DebugVersion/src/illegal_parking/incident_positive_exposure.py")})
    protocol = {"status": "frozen_before_new_continuous_event_replay", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_protocol_sha256": sha256(parent_path), "parent_event_protocol_sha256": sha256(event_path),
                "models": event["models"], "cohorts": cohorts, "annotations": annotations,
                "input_media": [row for row in parent["input_media"] if row["path"] in needed],
                "sampling": event["sampling"], "inference": event["inference"], "timing": event["timing"],
                "constraints": settings, "threshold_grid": event["calibration"]["threshold_grid"],
                "calibration_batch_sha256": sha256(directory / "batch.json"),
                "original_selection_sha256": sha256(original_selection_path),
                "excluded_previously_replayed_clips": sorted(excluded), "prior_replay_inventory": inventory,
                "eligible_new_test_clips": eligible_counts, "implementation_sha256": implementations,
                "detector_sha256": event["detector_sha256"], "reference_model_sha256": event["reference_model_sha256"],
                "ffprobe_sha256": event["ffprobe_sha256"], "extraction_versions": event["extraction_versions"],
                "comparison": "All eight positive-context constrained policies versus their original onset-only training-selected policies and same-grid constant controls on each new 100-clip cohort",
                "uncertainty": "Pointwise paired source-clip bootstrap,2000 resamples,seed42,fixed models/calibration; report all policies and test constraint exceedance",
                "scope": "Calibration-motivated follow-up after the previous event study; new continuous-event replays exclude previously replayed sources, but paired-window test scores on these sources were already observed. Pre-event contexts are annotation-derived accident-clip prefixes, not independently reviewed normal CCTV.",
                "default_policy": event["default_policy"]}
    selection = {"policies": policies, "calibration_batch_sha256": protocol["calibration_batch_sha256"],
                 "selected_at_utc": datetime.now(timezone.utc).isoformat(), "selection_inputs": "Existing official-training holdouts only; no new event test reports exist or are read"}
    output.write_text(ScalarJSON.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n")
    selection["protocol_sha256"] = sha256(output)
    selection_path.write_text(ScalarJSON.dumps(selection, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(ScalarJSON.dumps({"protocol_sha256": sha256(output), "eligible_counts": eligible_counts,
                           "excluded_clips": len(excluded), "models": {key: row["selected"]["threshold"] if row["selected"] else None for key, row in policies.items()}}, indent=2))


if __name__ == "__main__":
    main()
