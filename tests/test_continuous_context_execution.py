import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from illegal_parking.incident_context_study import fit_indices, validate_features, model_reports, select_policy, operating_point
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES


ROOT = Path(__file__).resolve().parents[1]


def worker():
    spec = importlib.util.spec_from_file_location("context_worker", ROOT / "scripts/run_continuous_context_study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def policy(threshold, recall=.5, episodes=.4, prefix=.1, support=20):
    return {"threshold": threshold, "summary": {"event_recall": recall, "off_target_episodes_per_clip": episodes},
            "pre_event": {"macro_positive_fraction": prefix, "clips_with_support": support}}


SETTINGS = {"episode_budget_per_clip": .5, "macro_pre_event_positive_cap": .2, "minimum_supported_clips": 20}


def test_policy_selection_follows_all_four_frozen_tie_breaks():
    rows = [policy(.1, recall=.1), policy(.2, episodes=.5), policy(.3, prefix=.2), policy(.4), policy(.5)]
    assert select_policy(rows, SETTINGS)["threshold"] == .5
    assert select_policy([policy(.9, recall=.8, prefix=.8), *rows], SETTINGS)["threshold"] == .5


@pytest.mark.parametrize("row", [policy(.1, support=19), policy(.1, prefix=None), policy(.1, prefix=.201), policy(.1, episodes=.501)])
def test_unsupported_or_infeasible_policies_remain_unselected(row):
    assert select_policy([row], SETTINGS) is None


def test_fit_membership_never_uses_calibration_or_other_split_labels():
    protocol = {"memberships": {"iid": {"fit_clip_paths": ["fit"]}}}
    rows = [{"path": "fit", "arm": "continuous", "fit_splits": ["iid"], "target": target} for target in (0, 1)]
    rows.append({"path": "heldout", "arm": "continuous", "fit_splits": ["geographic"], "target": 1})
    assert fit_indices(rows, protocol, "continuous", "iid").tolist() == [0, 1]
    rows[-1]["fit_splits"] = ["iid"]
    with pytest.raises(ValueError, match="partition"):
        fit_indices(rows, protocol, "continuous", "iid")


@pytest.mark.parametrize("fault", ["shape", "dtype", "nonfinite"])
def test_feature_contract_rejects_malformed_tensors(fault):
    matrix = np.zeros((2, 15, len(CANDIDATE_MOTION_FEATURE_NAMES)), dtype=np.float32)
    if fault == "shape":
        matrix = matrix[:, :14]
    elif fault == "dtype":
        matrix = matrix.astype(np.float64)
    else:
        matrix[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="Feature tensors"):
        validate_features(matrix, 2)


def test_empty_context_feature_contract_preserves_miss_denominator():
    matrix = np.empty((0, 15, len(CANDIDATE_MOTION_FEATURE_NAMES)), dtype=np.float32)
    assert validate_features(matrix, 0).shape[0] == 0


def test_cohort_replay_requires_each_frozen_identity_once():
    protocol = {"memberships": {"iid": {"test_clip_paths": ["a", "empty"]}}}
    reports = [{"path": "a", "scores": {"continuous_candidate_iid": [.8]}}, {"path": "empty", "scores": {"continuous_candidate_iid": []}}]
    chosen = model_reports(protocol, reports, "continuous_candidate_iid", "test")
    assert len(chosen) == 2 and chosen[-1]["window_probabilities"] == []
    with pytest.raises(ValueError, match="exactly one"):
        model_reports(protocol, reports[:1], "continuous_candidate_iid", "test")


def test_empty_clip_and_missing_hops_remain_in_operating_denominators():
    reports = [{"path": "a", "clock": "normalized_presentation_time", "annotation_free_inference": True,
                "window_starts": [0, 30], "window_end_times_sec": [2, 6], "window_probabilities": [1, 1], "rejected_window_starts": [15]},
               {"path": "empty", "clock": "normalized_presentation_time", "annotation_free_inference": True,
                "window_starts": [], "window_end_times_sec": [], "window_probabilities": [], "rejected_window_starts": []}]
    annotations = {name: {"path": name, "event_time_sec": 5, "collision_type": "fixture"} for name in ("a", "empty")}
    result = operating_point(reports, annotations, .5)
    assert result["summary"]["clip_count"] == 2 and result["summary"]["event_recall"] == 0
    assert result["pre_event"]["clips_with_support"] == 1


def test_json_scalar_adapter_rejects_nonfinite_and_preserves_numpy_int(tmp_path):
    module = worker()
    file = tmp_path / "record.json"
    module.dump(file, {"value": np.int64(2)})
    assert json.loads(file.read_text()) == {"value": 2}
    with pytest.raises(ValueError):
        module.dump(file, {"value": float("nan")})


def test_smoke_features_are_explicitly_barred_from_fitting(tmp_path, monkeypatch):
    module = worker()
    monkeypatch.setattr(module, "read_bound", lambda _: {"smoke": True})
    with pytest.raises(ValueError, match="Smoke extraction"):
        module.load_fit([], tmp_path)


def test_final_epoch_toy_fit_is_reproducible_without_test_inputs():
    import torch
    torch.set_num_threads(2)
    module = worker()
    settings = dict(seed=42, channels=4, dropout=.1, epochs=2, learning_rate=.001, weight_decay=.0001, batch_size=4)
    features = np.random.default_rng(42).normal(size=(8, 15, 2)).astype(np.float32)
    labels = np.asarray([0, 1] * 4)
    a, scaler, losses = module.fit_one(features, labels, settings, "cpu")
    b, _, other = module.fit_one(features, labels, settings, "cpu")
    assert losses == other and len(losses) == 2
    assert all(torch.equal(a.state_dict()[name], value) for name, value in b.state_dict().items())
    np.testing.assert_allclose(scaler.feature_mean, features.mean(axis=(0, 1)))
