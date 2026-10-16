import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from illegal_parking.incident_baseline import grouped_calibration_indices
from illegal_parking.incident_event_calibration import reconstruct_tcn_partitions, protocol_clips, regrid_event_report, select_event_threshold


def test_partition_reconstruction_matches_training_holdout_and_rejects_overlap():
    payload = {"path": np.array(["a", "a", "b", "c", "d", "e"]), "iid_split": np.array(["train"] * 4 + ["test"] * 2)}
    fit, calibration = grouped_calibration_indices(payload["path"][:4], fraction=0.34, seed=42)
    report = {"split_scheme": "iid", "decision_threshold": {"source": "train_holdout_fpr_calibration", "calibration_fraction": 0.34, "seed": 42}, "fit": {"sample_count": len(fit)}, "calibration": {"sample_count": len(calibration)}, "test": {"sample_count": 2}}
    groups = reconstruct_tcn_partitions(payload, report)
    assert set(groups["fit"]) | set(groups["calibration"]) == {"a", "b", "c"}
    assert not set(groups["fit"]) & set(groups["calibration"])
    payload["path"][-1] = "a"
    with pytest.raises(ValueError, match="overlap"):
        reconstruct_tcn_partitions(payload, report)


def test_protocol_requires_correct_model_disjoint_clips_and_official_training_membership(tmp_path):
    model = tmp_path / "model.pt"
    model.write_bytes(b"model fixture")
    definition = {"model_sha256": hashlib.sha256(model.read_bytes()).hexdigest(), "fit_clip_paths": ["fit"], "calibration_clip_paths": ["cal"], "test_clip_paths": ["test"]}
    protocol = {"models": {"iid": definition}}
    clips = [SimpleNamespace(relative_path=Path(name), iid_split=split) for name, split in (("fit", "train"), ("cal", "train"), ("test", "test"))]
    assert protocol_clips(protocol, clips, "iid", "calibration", model)[0].relative_path == Path("cal")
    definition["calibration_clip_paths"] = ["test"]
    with pytest.raises(ValueError, match="disjoint"):
        protocol_clips(protocol, clips, "iid", "calibration", model)
    definition["test_clip_paths"] = []
    with pytest.raises(ValueError, match="official"):
        protocol_clips(protocol, clips, "iid", "calibration", model)


def test_regrid_retains_actual_source_times_and_recomputes_persistence():
    report = {"sampled_frames": 32, "sequence_steps": 15, "window_starts": [0, 5, 10, 15], "window_probabilities": [0.9, 0.1, 0.2, 0.9], "window_end_times_sec": [2.1, 2.8, 3.4, 4.2]}
    result = regrid_event_report(report, hop_steps=15, threshold=0.5, gate=2)
    assert result["window_end_times_sec"] == [2.1, 4.2]
    assert result["window_review_flags"] == [False, True]
    assert result["window_starts"] == [0, 15]
    assert report["window_starts"] == [0, 5, 10, 15]


def test_threshold_selection_enforces_budget_and_declared_tiebreaks():
    def row(threshold, recall, episodes):
        return {"threshold": threshold, "summary": {"event_recall": recall, "off_target_episodes_per_clip": episodes}}

    curve = [row(0.2, 1, 2), row(0.4, 0.8, 0.5), row(0.5, 0.8, 0.25), row(0.6, 0.8, 0.25)]
    assert select_event_threshold(curve, 0.5)["threshold"] == 0.6
    assert select_event_threshold(curve, 0.1) is None
    with pytest.raises(ValueError, match="nonnegative"):
        select_event_threshold(curve, -1)
