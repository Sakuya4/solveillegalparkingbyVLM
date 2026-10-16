import numpy as np
import pytest

from illegal_parking.incident_positive_exposure import pre_event_positive_contexts, select_positive_exposure_policy
from scripts.prepare_positive_exposure_study import choose_new_event_cohort, replay_inventory


def test_prefix_fraction_uses_equal_clip_weights_not_duration_or_missing_zeroes():
    reports = [{"path": "a", "window_end_times_sec": [2, 4, 6], "window_probabilities": [1, 0, 0]},
               {"path": "b", "window_end_times_sec": [2], "window_probabilities": [1]},
               {"path": "c", "window_end_times_sec": [2], "window_probabilities": [0]}]
    annotations = {"a": {"event_time_sec": 8}, "b": {"event_time_sec": 4}, "c": {"event_time_sec": 3}}
    result = pre_event_positive_contexts(reports, annotations, .5)
    assert result["macro_positive_fraction"] == pytest.approx(2 / 3)
    assert result["pooled_positive_fraction"] == .5
    assert result["clips_with_support"] == 2 and result["clips_without_support"] == 1
    assert result["pre_event_contexts"] == 4


def test_zero_support_is_undefined_and_invalid_values_are_rejected():
    report = {"path": "a", "window_end_times_sec": [], "window_probabilities": []}
    assert pre_event_positive_contexts([report], {"a": {"event_time_sec": 0}}, .5)["macro_positive_fraction"] is None
    report["window_probabilities"] = [np.nan]
    with pytest.raises(ValueError):
        pre_event_positive_contexts([report], {"a": {"event_time_sec": 0}}, .5)


def test_additional_cap_rejects_degenerate_episode_feasible_always_positive_policy():
    def point(threshold, recall, exposure, support=30):
        return {"threshold": threshold, "summary": {"event_recall": recall, "off_target_episodes_per_clip": .4},
                "pre_event": {"clips_with_support": support, "macro_positive_fraction": exposure}}
    curve = [point(0, .8, 1), point(.7, .4, .2), point(.9, .4, .1), point(1, .9, 0, 19)]
    selected = select_positive_exposure_policy(curve)
    assert selected["threshold"] == .9
    assert select_positive_exposure_policy(curve[:1]) is None


def test_new_cohort_selection_is_deterministic_disjoint_and_not_score_based():
    rows = [{"path": str(index), "iid_partition": "test" if index > 2 else "fit"} for index in range(20)]
    chosen, eligible = choose_new_event_cohort(rows, "iid", {"4", "5"}, count=10)
    reversed_chosen, _ = choose_new_event_cohort(list(reversed(rows)), "iid", {"4", "5"}, count=10)
    assert chosen == reversed_chosen and len(chosen) == 10 and eligible == 15
    assert not set(chosen) & {"0", "1", "2", "4", "5"}
    with pytest.raises(ValueError, match="Too few"):
        choose_new_event_cohort(rows, "iid", set(), count=100)


def test_prior_replay_inventory_uses_source_identity_and_does_not_include_window_summaries(tmp_path):
    import json
    directory = tmp_path / "outputs/desktop"
    directory.mkdir(parents=True)
    (directory / "alias.json").write_text(json.dumps({"annotation_free_inference": True, "input": "alias.mp4",
                                                     "input_sha256": "hash", "window_probabilities": []}))
    (directory / "window.json").write_text(json.dumps({"path": "real_videos/new.mp4", "test": {"f1": .99}}))
    media = [{"path": "real_videos/source.mp4", "sha256": "hash"}, {"path": "real_videos/new.mp4", "sha256": "other"}]
    excluded, records = replay_inventory(tmp_path, media)
    assert excluded == ["real_videos/source.mp4"]
    assert len(records) == 1 and records[0]["report"].endswith("alias.json")


def test_frozen_followup_preserves_models_and_excludes_all_prior_replay_sources():
    import json
    from pathlib import Path
    from scripts.prepare_presentation_protocol import sha256
    from scripts.run_overnight_research import positive_exposure_steps

    results = Path(__file__).resolve().parents[1] / "docs/results"
    protocol = json.loads((results / "positive_exposure_protocol.json").read_text())
    selection = json.loads((results / "positive_exposure_selection.json").read_text())
    assert selection["protocol_sha256"] == sha256(results / "positive_exposure_protocol.json")
    assert len(protocol["models"]) == len(selection["policies"]) == 8
    excluded = set(protocol["excluded_previously_replayed_clips"])
    for definition in protocol["cohorts"].values():
        assert len(definition["test_clip_paths"]) == 100
        assert not set(definition["test_clip_paths"]) & excluded
    for policy in selection["policies"].values():
        chosen = policy["selected"]
        assert chosen is not None and chosen["threshold"] > 0
        assert chosen["pre_event"]["clips_with_support"] >= 20
        assert chosen["pre_event"]["macro_positive_fraction"] <= .2
        assert chosen["summary"]["off_target_episodes_per_clip"] <= .5
    assert len(positive_exposure_steps()) == 2
    assert "calibrat" not in str(positive_exposure_steps())


def test_publication_curve_reads_only_assigned_calibration_paths():
    from scripts.publish_positive_exposure import calibration_curve
    protocol = {"models": {"key": {"split": "iid", "window_threshold": .7}},
                "cohorts": {"iid": {"calibration_clip_paths": ["a"], "test_clip_paths": ["not_read"]}},
                "threshold_grid": [0, .5, 1], "annotations": {"a": {"path": "a", "event_time_sec": 7, "collision_type": "rear-end"}}}
    reports = [{"path": "a", "clock": "normalized_presentation_time", "annotation_free_inference": True,
                "window_starts": [0, 15, 30], "window_end_times_sec": [2, 4, 6], "rejected_window_starts": [],
                "scores": {"key": [.2, .8, .9]}}]
    curve = calibration_curve(protocol, reports, "key")
    assert len(curve) == 4
    point = next(row for row in curve if row["threshold"] == .5)
    assert point["summary"]["event_recall"] == 1
    assert point["pre_event"]["macro_positive_fraction"] == .5
