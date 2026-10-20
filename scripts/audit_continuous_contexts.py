from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.audit_positive_contexts import assert_same, clip_outcome, digest, summarize
from illegal_parking.research_statistics import paired_cluster_coverage_interval
import numpy as np


def read_csv(root, record):
    path = root / record["path"]
    if digest(path) != record["sha256"]:
        raise ValueError("Public CSV provenance changed")
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != record["rows"]:
        raise ValueError("Public CSV row count changed")
    return rows


def audit(root=ROOT):
    base = root / "docs/results"
    manifest = json.loads((base / "continuous_context_inventory.json").read_text())
    for field, name in (("protocol", "continuous_training_protocol.json"),
                        ("results", "continuous_context_results.json"), ("selection", "continuous_context_selection.json")):
        if digest(base / name) != manifest[field + "_sha256"]:
            raise ValueError("Public study provenance changed")
    study = json.loads((base / "continuous_context_results.json").read_text())
    protocol = json.loads((base / "continuous_training_protocol.json").read_text())
    selection = json.loads((base / "continuous_context_selection.json").read_text())
    inventory = read_csv(root, manifest["inventory"])
    contexts = read_csv(root, study["files"]["contexts"])
    outcomes = read_csv(root, study["files"]["outcomes"])
    curves = read_csv(root, study["files"]["calibration"])
    identity = lambda row: (row["cohort"], row["model"], row["path"])
    clips = {identity(row): row for row in inventory}
    expected = {(cohort, row["model"], name) for row in study["models"]
                for cohort in ("calibration", "test")
                for name in protocol["memberships"][row["model"].rsplit("_", 1)[-1]][cohort + "_clip_paths"]}
    if len(clips) != len(inventory) or set(clips) != expected:
        raise ValueError("Public source membership is incomplete or duplicated")
    grouped = defaultdict(list)
    media = {row["path"]: row["sha256"] for row in protocol.get("input_media", [])}
    for row in contexts:
        key = identity(row)
        if key not in clips:
            raise ValueError("Orphan public context")
        clip = clips[key]
        if row["source_sha256"] != clip["source_sha256"] or row["split"] != clip["split"]:
            raise ValueError("Context/source provenance differs")
        assert_same(float(row["event_pts_sec"]), float(clip["annotation_pts_sec"]))
        grouped[key].append({**row, "start_slot": row["start"]})
    for key, clip in clips.items():
        annotation = protocol["annotations"][clip["path"]]
        if media and clip["source_sha256"] != media[clip["path"]]:
            raise ValueError("Inventory differs from frozen input media")
        if clip["collision_type"] != annotation["collision_type"]:
            raise ValueError("Collision annotation changed")
        assert_same(float(clip["annotation_pts_sec"]), annotation["event_time_sec"])
        if len(grouped[key]) != int(clip["valid_contexts"]):
            raise ValueError("Missing or duplicated public contexts")
    indexed_outcomes = {identity(row): row for row in outcomes}
    if len(indexed_outcomes) != len(outcomes):
        raise ValueError("Duplicate public outcome")
    actual_outcome_keys, detections = set(), {}
    curve_index = {(row["model"], float(row["threshold"])): row for row in curves}
    if len(curve_index) != 8 * len(protocol["calibration"]["threshold_grid"]):
        raise ValueError("Calibration curve is incomplete")

    def evaluate(cohort, model, threshold, constant=None):
        split = model.rsplit("_", 1)[-1]
        selected = [clips[(cohort, model, name)] for name in protocol["memberships"][split][cohort + "_clip_paths"]]
        values = [clip_outcome(clip, grouped[identity(clip)], threshold, constant) for clip in selected]
        summary, prefix = summarize(selected, values)
        return {"threshold": threshold, "summary": summary, "pre_event": prefix}, selected, values

    def verify_outcomes(model, selected, values, threshold):
        for clip, value in zip(selected, values):
            key = ("test", model, clip["path"])
            actual_outcome_keys.add(key)
            row = indexed_outcomes[key]
            if row["source_sha256"] != clip["source_sha256"]:
                raise ValueError("Outcome source hash changed")
            assert_same(float(row["threshold"]) if row["threshold"] else None, threshold)
            for field in ("detected", "trigger_pts_sec", "signed_delay_sec", "early_episodes", "late_episodes"):
                actual = float(row[field]) if row[field] else None
                assert_same(actual, value[field] if value is not None else None)
            for field in ("valid_contexts", "rejected_contexts"):
                assert_same(int(row[field]), int(clip[field]))

    for model in study["models"]:
        key = model["model"]
        reconstructed = []
        for threshold in protocol["calibration"]["threshold_grid"]:
            point, _, _ = evaluate("calibration", key, threshold)
            reconstructed.append(point)
            row = curve_index[(key, threshold)]
            for field, value in (("event_recall", point["summary"]["event_recall"]),
                                 ("off_target_per_clip", point["summary"]["off_target_episodes_per_clip"]),
                                 ("prefix_support", point["pre_event"]["clips_with_support"]),
                                 ("prefix_macro", point["pre_event"]["macro_positive_fraction"])):
                assert_same(float(row[field]) if row[field] else None, value)
        assert_same(reconstructed, selection["policies"][key]["curve"])
        settings = protocol["calibration"]
        eligible = [point for point in reconstructed
                    if point["summary"]["off_target_episodes_per_clip"] <= settings["episode_budget_per_clip"]
                    and point["pre_event"]["clips_with_support"] >= settings["minimum_supported_clips"]
                    and point["pre_event"]["macro_positive_fraction"] is not None
                    and point["pre_event"]["macro_positive_fraction"] <= settings["macro_pre_event_positive_cap"]]
        chosen = max(eligible, key=lambda point: (point["summary"]["event_recall"],
                     -point["summary"]["off_target_episodes_per_clip"], -point["pre_event"]["macro_positive_fraction"], point["threshold"])) if eligible else None
        assert_same(chosen, model["calibration"])
        assert_same(chosen, selection["policies"][key]["selected"])
        assert_same(chosen["threshold"] if chosen else None, model["selected_threshold"])
        if chosen:
            point, selected, values = evaluate("test", key, chosen["threshold"])
            assert_same(point, model["test"])
            verify_outcomes(key, selected, values, chosen["threshold"])
            detections[key] = [value["detected"] for value in values]
            assert_same(point["summary"]["off_target_episodes_per_clip"] > .5, model["test_episode_cap_exceeded"])
            prefix = point["pre_event"]["macro_positive_fraction"]
            assert_same(prefix > .2 if prefix is not None else None, model["test_prefix_cap_exceeded"])
        else:
            split = key.rsplit("_", 1)[-1]
            selected = [clips[("test", key, name)] for name in protocol["memberships"][split]["test_clip_paths"]]
            verify_outcomes(key, selected, [None] * len(selected), None)
            assert_same(model["test"], None)
    for split in protocol["splits"]:
        reference = next(row["model"] for row in study["models"] if row["model"].endswith("_" + split))
        for name, constant in (("clock_only_positive", 1), ("always_negative", 0)):
            point, selected, values = evaluate("test", reference, .5, constant)
            assert_same(point, study["controls"][split][name])
            verify_outcomes(name + "_" + split, selected, values, .5)
    if actual_outcome_keys != set(indexed_outcomes):
        raise ValueError("Extra or missing public outcomes")
    for row in study["contrasts"]:
        continuous = f'continuous_{row["feature"]}_{row["split"]}'
        anchored = f'anchored_{row["feature"]}_{row["split"]}'
        interval = paired_cluster_coverage_interval(np.ones(len(detections[continuous])), detections[continuous], detections[anchored]) if continuous in detections and anchored in detections else None
        assert_same(interval, row["paired_interval"])
    return {"calibration_points": len(curves), "clip_inventory_rows": len(inventory), "context_scores": len(contexts),
            "outcomes": len(outcomes), "paired_intervals": len(study["contrasts"]),
            "audit_sha256": digest(Path(__file__)), "independent_evaluator_sha256": digest(ROOT / "scripts/audit_positive_contexts.py"),
            "bootstrap_sha256": digest(ROOT / "Application/DebugVersion/src/illegal_parking/research_statistics.py"),
            "results_sha256": manifest["results_sha256"], "inventory_sha256": manifest["inventory"]["sha256"],
            "scope": "Public CSV and protocol reconstruction, without videos, fitted checkpoints, production event evaluators or GPU. Bootstrap helper is shared and hash-recorded."}


if __name__ == "__main__":
    result = audit()
    output = ROOT / "docs/results/continuous_context_audit.json"
    if output.exists():
        raise FileExistsError("Do not overwrite a published audit")
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(result, indent=2))
