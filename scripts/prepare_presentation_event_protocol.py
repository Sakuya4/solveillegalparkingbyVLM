from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.extract_presentation_features import load_manifest
from scripts.prepare_presentation_protocol import sha256


def main():
    parent_path = ROOT / "docs/results/presentation_protocol.json"
    parent, rows = load_manifest(parent_path)
    output = ROOT / "docs/results/presentation_event_protocol.json"
    if output.exists():
        raise FileExistsError("Event protocols are immutable")
    comparison_path = ROOT / "docs/results/presentation_comparison.json"
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison["protocol_sha256"] != sha256(parent_path) or len(comparison["models"]) != 8:
        raise ValueError("The complete verified eight-model study is required")
    models = {}
    for row in comparison["models"]:
        key = f"{row['arm']}_{row['feature']}_{row['split']}"
        path = f"models/presentation_{key}.pt"
        if sha256(ROOT / path) != row["model_sha256"]:
            raise ValueError("Fitted model bytes changed")
        models[key] = {"path": path, "sha256": row["model_sha256"], "arm": row["arm"],
                       "feature": row["feature"], "split": row["split"],
                       "window_threshold": row["decision_threshold"]["effective"]}
    names = sorted({name for scheme in parent["continuous_cohorts"].values() for cohort in scheme.values() for name in cohort})
    annotations = {}
    for row in rows:
        if row["path"] in names:
            record = {key: row[key] for key in ("path", "supplied_event_frame", "event_time_sec", "collision_type")}
            if row["path"] in annotations and annotations[row["path"]] != record:
                raise ValueError("Conflicting annotation anchors")
            annotations[row["path"]] = record
    if set(annotations) != set(names):
        raise ValueError("Frozen event cohorts must retain all annotation anchors")
    implementations = dict(parent["implementation_sha256"])
    for name in ("scripts/prepare_presentation_event_protocol.py", "scripts/replay_presentation_events.py",
                 "scripts/evaluate_presentation_events.py", "scripts/prepare_presentation_protocol.py",
                 "Application/DebugVersion/src/illegal_parking/incident_presentation_events.py",
                 "Application/DebugVersion/src/illegal_parking/incident_tcn_inference.py",
                 "Application/DebugVersion/src/illegal_parking/incident_features.py",
                 "Application/DebugVersion/src/illegal_parking/incident_trajectory.py",
                 "Application/DebugVersion/src/illegal_parking/incident_evaluation.py",
                 "Application/DebugVersion/src/illegal_parking/incident_event_calibration.py"):
        implementations[name] = sha256(ROOT / name)
    if any(sha256(ROOT / name) != value for name, value in implementations.items()):
        raise ValueError("Frozen implementation bytes changed")
    protocol = {"status": "frozen_before_continuous_pts_replay", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "parent_protocol_sha256": sha256(parent_path), "comparison_sha256": sha256(comparison_path),
                "cohorts": parent["continuous_cohorts"], "models": models, "annotations": annotations,
                "input_media": [row for row in parent["input_media"] if row["path"] in names],
                "sampling": parent["sampling"], "detector_sha256": parent["detector_sha256"],
                "extraction_versions": json.loads((ROOT / "data/processed/accident/presentation_fixed/extraction.json").read_text(encoding="utf-8"))["versions"],
                "inference": {"width": 320, "image_size": 320, "confidence": .15, "hop_steps": 15,
                              "tracker": "ByteTrack reset within each 16-image context, matching training",
                              "common_input": "Both fitted arms use the same annotation-free 7.5-Hz presentation grid",
                              "invalid_window": "No prediction; gap resets persistence; never impute or replace a clip",
                              "clock": "Actual last sampled frame PTS, normalized to the first source frame"},
                "calibration": parent["evaluation"]["continuous"], "timing": {"early_tolerance_sec": 1, "late_tolerance_sec": 3},
                "implementation_sha256": implementations, "ffprobe_sha256": parent["ffprobe_sha256"],
                "reference_model_sha256": parent["reference_model_sha256"],
                "reporting": "All eight fixed models; four pointwise paired source-clip recall contrasts; 2000 resamples, seed 42",
                "scope": "Off-target episodes per accident clip, not independently reviewed normal-camera false-alarm rate",
                "default_policy": parent["default_policy"]}
    output.write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"models": len(models), "union_clips": len(names), "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
