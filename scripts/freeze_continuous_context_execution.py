from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_continuous_context_study import PROTOCOL, EXECUTION, sha, dump, versions, now


def main():
    if EXECUTION.exists():
        raise FileExistsError("Frozen execution appendices are immutable")
    protocol = json.loads(PROTOCOL.read_text())
    rows_path = PROTOCOL.with_name(protocol["fit_windows_file"])
    if sha(rows_path) != protocol["fit_windows_sha256"]:
        raise ValueError("Frozen fitting identities changed")
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    common = set.intersection(*(set(value["fit_clip_paths"]) for value in protocol["memberships"].values()))
    support = {}
    for row in rows:
        support.setdefault((row["path"], row["arm"]), set()).add(row["target"])
    smoke_sources = [name for name in sorted(common) if all(support.get((name, arm)) == {0, 1} for arm in protocol["arms"])][:2]
    if len(smoke_sources) != 2:
        raise ValueError("Two common-fit smoke sources with both labels per arm are required")
    implementations = {name: sha(ROOT / name) for name in (
        "scripts/run_continuous_context_study.py", "scripts/publish_continuous_context_study.py", "scripts/freeze_continuous_context_execution.py",
        "scripts/replay_presentation_events.py", "scripts/prepare_presentation_protocol.py",
        *("Application/DebugVersion/src/illegal_parking/" + name for name in (
            "incident_context_study.py", "incident_tcn_inference.py", "incident_presentation_events.py", "incident_evaluation.py",
            "incident_positive_exposure.py", "incident_event_calibration.py", "research_statistics.py", "incident_trajectory.py", "incident_features.py", "gpu_job.py")))}
    prior = ROOT / "data/processed/accident/presentation_fixed"
    old = json.loads((prior / "extraction.json").read_text())
    if old["protocol_sha256"] != protocol["parent_protocol_sha256"] or old["sequences_sha256"] != sha(prior / "sequences.npz") or old["versions"] != versions():
        raise ValueError("Prior anchored features no longer match the frozen runtime and parent")
    calibration = ROOT / "outputs/accident/presentation_events_calibration"
    batch = json.loads((calibration / "batch.json").read_text())
    if batch["versions"] != versions() or set(batch["clips"]) != {name for value in protocol["memberships"].values() for name in value["calibration_clip_paths"]}:
        raise ValueError("Prior continuous calibration features must match the full TRAIN holdout union")
    import torch
    dump(EXECUTION, {"created_at_utc": now(), "protocol_sha256": sha(PROTOCOL), "implementation_sha256": implementations,
                     "versions": versions(), "cuda_runtime": torch.version.cuda, "gpu_name": torch.cuda.get_device_name(0),
                     "detector_device": "0", "threads": 2, "smoke_sources": smoke_sources,
                     "smoke_policy": "CPU-only real-detector extraction on two common-fit sources; no partial-cohort fitting; separate cache/artifacts",
                     "anchored_cache": {"directory": prior.relative_to(ROOT).as_posix(),
                                        "files": {name: sha(prior / name) for name in ("extraction.json", "sequences.npz")}},
                     "calibration_cache": {"directory": calibration.relative_to(ROOT).as_posix(), "batch_sha256": sha(calibration / "batch.json")},
                     "cache_policy": "Reuse only checksum-verified parent fixed-time anchored tensors and continuous TRAIN calibration tensors with matching source/PTS identities, runtime and feature contract. Ignore all old model scores. Newly generated continuous-fit and test features use separate per-context/per-source caches.",
                     "decode_gate": "Every expanded fitting frame/annotation is decoded and checked before any complete-cohort extraction/fitting. Retain all failures and stop; no replacement or label changes.",
                     "order": ["CPU fixtures and execution freeze", "Two-clip CPU smoke", "Complete fit decoder audit", "Complete fit extraction", "All eight final-epoch fits", "TRAIN replay and all-policy freeze", "New test replay", "Checkpoint-score verification and all-policy/control publication"],
                     "fitting_checkpoint_threshold": "0.5 is an explicitly unselected placeholder, not a deployment policy; only the separate TRAIN selection supplies operating thresholds",
                     "feature_cost": "Report newly extracted continuous tensors and reused anchored cache separately; do not claim an anchored extraction-speed improvement from cache reuse"})
    print(json.dumps({"execution_sha256": sha(EXECUTION), "smoke_sources": smoke_sources}, indent=2))


if __name__ == "__main__":
    main()
