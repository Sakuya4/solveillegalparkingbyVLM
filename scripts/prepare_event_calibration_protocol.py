from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.incident_batch import select_evaluation_clips
from illegal_parking.incident_event_calibration import reconstruct_tcn_partitions


def main():
    parser = argparse.ArgumentParser(description="Freeze continuous-event calibration cohorts and operating-point rules before inference.")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/continuous_event_calibration_protocol.json")
    args = parser.parse_args()
    archive = ROOT / "data/processed/accident/candidate_roi_sequences_full.npz"
    published = json.loads((ROOT / "docs/results/full_cohort_temporal.json").read_text(encoding="utf-8"))
    if hashlib.sha256(archive.read_bytes()).hexdigest() != published["sequences_sha256"]:
        raise ValueError("Training sequence archive differs from the published full-cohort study")
    metadata = ROOT / "data/raw/accident/full/extracted/metadata-real.csv"
    if hashlib.sha256(metadata.read_bytes()).hexdigest() != published["metadata_sha256"]:
        raise ValueError("Dataset metadata changed since the verified full-cohort study")
    payload = np.load(archive, allow_pickle=False)
    clips = load_accident_manifest(metadata)
    models = {}
    for split in ("iid", "geographic"):
        stem = f"candidate_full_tcn_{split}_fpr20"
        checkpoint = ROOT / f"models/{stem}.pt"
        report_path = ROOT / f"outputs/accident/{stem}_report.json"
        verified = next(row for row in published["models"] if row["feature_set"] == "candidate" and row["split"] == split)
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != verified["model_sha256"] or hashlib.sha256(report_path.read_bytes()).hexdigest() != verified["report_sha256"]:
            raise ValueError("Model or training report differs from its published hash")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        groups = reconstruct_tcn_partitions(payload, report)
        test = [clip.relative_path.as_posix() for clip in select_evaluation_clips(clips, split, 100, 42)]
        if not set(test) <= set(groups["test"]):
            raise ValueError("Fixed test selection includes a clip outside the eligible test cohort")
        models[split] = {"checkpoint": checkpoint.relative_to(ROOT).as_posix(), "model_sha256": verified["model_sha256"],
                         "training_report_sha256": verified["report_sha256"], "original_window_threshold": report["decision_threshold"]["effective"],
                         "fit_clip_paths": groups["fit"], "calibration_clip_paths": groups["calibration"], "test_clip_paths": test}
    result = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "frozen_before_continuous_event_inference",
              "sequences_sha256": published["sequences_sha256"], "metadata_sha256": published["metadata_sha256"],
              "inference": {"target_fps": 7.5, "target_width": 640, "feature_width": 320, "image_size": 320, "confidence": 0.15, "sequence_steps": 15, "hop_steps": 5, "source_clock": True},
              "evaluated_hop_steps": [5, 15], "persistence_gates": [1, 2], "off_target_budgets_per_clip": [0.25, 0.5, 1.0],
              "threshold_grid": [index / 100 for index in range(101)],
              "selection_rule": "On held-out training clips only: maximize event recall subject to off-target episode budget; ties favor fewer off-target episodes, then higher threshold; infeasible budgets remain unselected",
              "timing": {"early_tolerance_sec": 1.0, "late_tolerance_sec": 3.0, "decision_clock": "actual sampled source-frame time"},
              "scope": "Off-target episodes in accident clips are not a normal-camera false-alarm rate. This study changes operating thresholds, not model weights or reference deployment defaults.",
              "models": models,
              "sources": ["https://doi.org/10.1016/j.patrec.2005.10.010", "https://scikit-learn.org/stable/modules/classification_threshold.html"]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        existing = json.loads(args.output.read_text(encoding="utf-8"))
        if {key: value for key, value in existing.items() if key != "created_at_utc"} != {key: value for key, value in result.items() if key != "created_at_utc"}:
            raise ValueError("A different frozen protocol already exists; choose a new output path")
    else:
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({split: {key: len(value) for key, value in model.items() if key.endswith("clip_paths")} for split, model in models.items()}, indent=2))


if __name__ == "__main__":
    main()
