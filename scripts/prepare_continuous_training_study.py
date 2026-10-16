from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/results"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def new_test_sources(rows, split, excluded, count=100):
    eligible = {row["path"] for row in rows if row[split + "_partition"] == "test" and row["path"] not in excluded}
    ordered = sorted(eligible, key=lambda name: hashlib.sha256(f"continuous-training-v1:42:{split}:{name}".encode()).hexdigest())
    if len(ordered) < count:
        raise ValueError("Insufficient sources for the declared event cohort")
    return sorted(ordered[:count]), len(ordered)


def main():
    output = BASE / "continuous_training_protocol.json"
    windows_file = output.with_suffix(".jsonl")
    if output.exists() or windows_file.exists():
        raise FileExistsError("Continuous training protocols are immutable")
    parent_path = BASE / "presentation_protocol.json"
    parent = json.loads(parent_path.read_text())
    rows_path = BASE / parent["windows_file"]
    if sha(rows_path) != parent["windows_sha256"]:
        raise ValueError("Parent source partitions changed")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    support_path = BASE / "continuous_training_support.json"
    support = json.loads(support_path.read_text())
    audit_path = BASE / "continuous_training_support_protocol.json"
    audit = json.loads(audit_path.read_text())
    if support["protocol_sha256"] != sha(audit_path) or audit["parent_protocol_sha256"] != sha(parent_path):
        raise ValueError("Fit-only audit lineage changed")
    prior_path = BASE / "positive_exposure_protocol.json"
    prior = json.loads(prior_path.read_text())
    excluded = set(prior["excluded_previously_replayed_clips"])
    excluded.update(name for definition in prior["cohorts"].values() for names in definition.values() for name in names)
    memberships, eligible_counts = {}, {}
    for split in ("iid", "geographic"):
        chosen, eligible_counts[split] = new_test_sources(rows, split, excluded)
        memberships[split] = {"fit_clip_paths": sorted({row["path"] for row in rows if row[split + "_partition"] == "fit"}),
                              "calibration_clip_paths": sorted({row["path"] for row in rows if row[split + "_partition"] == "calibration"}),
                              "test_clip_paths": chosen}
    fit_rows = []
    for row in rows:
        splits = [split for split in memberships if row["path"] in memberships[split]["fit_clip_paths"]]
        if splits:
            fit_rows.append({"path": row["path"], "arm": "anchored", "fit_splits": splits,
                             "target": row["target"], "indices": row["fixed_indices"],
                             "times_sec": row["fixed_times_sec"], "event_time_sec": row["event_time_sec"]})
    definition = support["files"]["contexts"]
    contexts_path = ROOT / definition["path"]
    if sha(contexts_path) != definition["sha256"]:
        raise ValueError("Fit-context audit bytes changed")
    anchors = {row["path"]: row["event_time_sec"] for row in rows}
    with contexts_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["label"] == "post":
                continue
            fit_rows.append({"path": row["path"], "arm": "continuous", "fit_splits": row["fit_splits"].split(";"),
                             "target": int(row["label"] == "target"), "indices": json.loads(row["indices"]),
                             "end_time_sec": float(row["end_time_sec"]), "start": int(row["start"]), "event_time_sec": anchors[row["path"]]})
    windows_file.write_text("".join(json.dumps(row) + "\n" for row in fit_rows), encoding="utf-8", newline="\n")
    needed = {name for definition in memberships.values() for names in definition.values() for name in names}
    implementations = dict(parent["implementation_sha256"])
    implementations[Path(__file__).relative_to(ROOT).as_posix()] = sha(Path(__file__))
    protocol = {"status": "frozen_before_continuous_training_extraction_and_fitting", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_protocol_sha256": sha(parent_path), "support_protocol_sha256": sha(audit_path), "support_results_sha256": sha(support_path),
                "prior_event_protocol_sha256": sha(prior_path), "fit_windows_file": windows_file.name, "fit_windows_sha256": sha(windows_file),
                "memberships": memberships, "eligible_new_test_clips": eligible_counts, "excluded_prior_event_sources": sorted(excluded),
                "input_media": [row for row in parent["input_media"] if row["path"] in needed],
                "annotations": {row["path"]: {key: row[key] for key in ("path", "supplied_event_frame", "event_time_sec", "collision_type")} for row in rows if row["path"] in needed},
                "arms": ["anchored", "continuous"], "features": ["candidate_", "global_"], "splits": ["iid", "geographic"],
                "training": {"model_count": 8, "architecture": "Existing TemporalConvClassifier", "channels": 32, "dropout": .1,
                             "epochs": 40, "batch_size": 64, "learning_rate": .001, "weight_decay": .0001, "optimizer": "AdamW", "seed": 42,
                             "checkpoint": "Final epoch; no early stopping or calibration-based epoch selection",
                             "standardization": "Fit-only per-channel mean/scale using the corresponding arm/split contexts",
                             "loss": "BCEWithLogitsLoss; pos_weight=fit negative count / fit positive count; equal context weights, shuffled seeded batches",
                             "scope": "All corresponding fit sources retained in inventory; only prefix/target contexts enter continuous fitting; no post contexts or held-out media in fitting"},
                "sampling": parent["sampling"],
                "inference": {"target_fps": 7.5, "sample_count": 16, "sequence_steps": 15, "hop_steps": 15, "gate": 2,
                              "width": 320, "image_size": 320, "confidence": .15, "tracker": "bytetrack.yaml", "tracker_scope": "Reset per context",
                              "missing_contexts": "No interpolation, no predictions; missing hops reset persistence; empty clips retained as misses"},
                "calibration": {"threshold_grid": [value / 100 for value in range(101)], "episode_budget_per_clip": .5,
                                "macro_pre_event_positive_cap": .2, "minimum_supported_clips": 20,
                                "prefix": "end_pts < event_pts - 1; macro mean over supported clips only",
                                "target": "Inclusive [event_pts-1,event_pts+3] episode endpoint",
                                "selection": "Maximize calibration event recall subject to both caps; ties prefer fewer off-target episodes, then lower macro prefix fraction, then higher threshold; infeasible remains unselected"},
                "execution_order": ["Freeze and test execution implementations before real jobs", "Verify every selected source and decoded context against PTS before accepting features",
                                    "Fit all eight models without test scoring", "Replay corresponding TRAIN holdouts and freeze all operating policies",
                                    "Only then replay both new fixed test cohorts and publish every model/control"],
                "evaluation": {"primary": "Four paired continuous-minus-anchored event recall contrasts by feature/split",
                               "secondary": ["Off-target episodes per accident clip", "Macro prefix positive fraction and support", "Detected-case median and p95 signed trigger delay", "Test cap exceedances", "Training/inference cost"],
                               "controls": ["Always-negative", "Same-grid always-positive clock-only; same gate and event bands"],
                               "bootstrap": "2000 paired source-clip percentile resamples, seed42, pointwise95%; conditional on one fitted seed and calibration",
                               "failure_policy": "Source/decoder/provenance mismatch stops publication; no replacement or outcome-based exclusion"},
                "implementation_sha256": implementations, "detector_sha256": parent["detector_sha256"], "reference_model_sha256": parent["reference_model_sha256"], "ffprobe_sha256": parent["ffprobe_sha256"],
                "scope": "Calibration-motivated joint context-phase/label intervention, not a pure architecture effect. Reused TRAIN holdouts are development data. New continuous-event cohorts exclude prior event replays, but their earlier window-test observations are known. Prefixes are accident-clip annotations, not independently reviewed normal cameras.",
                "representation_gate": "VideoMAE or another representation requires a separate frozen arm under these same source/context/label contracts; no unmatched architecture claims",
                "default_policy": "Retain all original 500-clip binaries/defaults and prior study artifacts; no test-based promotion"}
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"protocol_sha256": sha(output), "fit_rows": len(fit_rows), "membership_counts": {key: {part: len(names) for part, names in value.items()} for key, value in memberships.items()}, "eligible_counts": eligible_counts}, indent=2))


if __name__ == "__main__":
    main()
