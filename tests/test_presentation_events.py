from copy import deepcopy

import numpy as np
import pytest

from illegal_parking.incident_presentation_events import (
    continuous_presentation_windows, evaluate_presentation_event,
    presentation_episode_indices, presentation_operating_summary,
)
from scripts.evaluate_presentation_events import model_reports, select_models
from scripts.replay_presentation_events import cohort_paths
from scripts.run_overnight_research import presentation_event_steps


def report(starts=(0, 15), scores=(.9, .9), times=(2., 4.)):
    return {"path": "a.mp4", "clock": "normalized_presentation_time", "annotation_free_inference": True,
            "window_starts": list(starts), "window_probabilities": list(scores),
            "window_end_times_sec": list(times), "rejected_window_starts": []}


def annotation(event=3.5):
    return {"path": "a.mp4", "event_time_sec": event, "supplied_event_frame": 99, "collision_type": "rear-end"}


def test_contexts_are_causal_global_grid_and_annotation_free():
    times = 8 + np.arange(121) / 15
    windows, rejected = continuous_presentation_windows(times)
    assert rejected == []
    assert [row["start"] for row in windows] == [0, 15, 30, 45]
    assert [row["end_time_sec"] for row in windows] == [2, 4, 6, 8]
    assert all(len(row["indices"]) == len(set(row["indices"])) == 16 for row in windows)
    assert all(row["times_sec"][-1] == row["end_time_sec"] for row in windows)


def test_duplicate_grid_frames_and_long_gaps_are_rejected_not_interpolated():
    windows, rejected = continuous_presentation_windows(np.arange(13) / 2)
    assert windows == [] and rejected == [0, 15, 30]
    times = np.concatenate([np.arange(30) / 15, 4 + np.arange(61) / 15])
    windows, rejected = continuous_presentation_windows(times)
    assert 0 in rejected and 15 in rejected
    assert [window["start"] for window in windows] == [30, 45]


def test_gate_emits_at_second_actual_decision_without_backdating():
    result = evaluate_presentation_event(report(times=(2, 5.3)), annotation(4.5), .5)
    assert result.detected and result.trigger_time_sec == 5.3
    assert result.trigger_delay_sec == pytest.approx(.8)
    assert result.accident_time_sec == 4.5


def test_invalid_context_resets_run_even_if_both_neighbors_are_positive():
    assert presentation_episode_indices([0, 30], [.9, .9], .5) == []
    assert presentation_episode_indices([0, 30, 45], [.9] * 3, .5) == [2]
    assert presentation_episode_indices([0, 15, 30, 60, 75], [.9] * 5, .5) == [1, 4]


def test_early_and_late_episode_bounds_are_closed_for_matching():
    for event in (1, 5):
        assert evaluate_presentation_event(report(), annotation(event), .5).detected
    assert evaluate_presentation_event(report(), annotation(5.01), .5).early_alert_episodes == 1
    assert evaluate_presentation_event(report(), annotation(.99), .5).late_alert_episodes == 1


def test_no_valid_context_is_missed_clip_not_removed_from_denominator():
    empty = report((), (), ())
    empty["rejected_window_starts"] = [0, 15]
    summary = presentation_operating_summary([empty], {"a.mp4": annotation()}, .5)
    assert summary["clip_count"] == summary["missed_clips"] == summary["clips_without_valid_windows"] == 1
    assert summary["rejected_windows"] == 2 and summary["event_recall"] == 0
    assert summary["trigger_delay_sec"]["median"] is None


@pytest.mark.parametrize("starts,scores", [([0, 15], [.9]), ([0, 0], [.9, .9]), ([0, 16], [.9, .9]),
                                           ([0, 15], [.9, float('nan')]), ([0, 15], [.9, 1.1]), ([0., 15.], [.9, .9])])
def test_invalid_probability_grid_contract_is_rejected(starts, scores):
    with pytest.raises(ValueError):
        presentation_episode_indices(starts, scores, .5)


@pytest.mark.parametrize("field,value", [("window_end_times_sec", [2, 2]), ("window_end_times_sec", [2, float('nan')]),
                                          ("clock", "frame_index_over_average_fps"), ("annotation_free_inference", False)])
def test_invalid_clocks_or_annotated_predictions_are_rejected(field, value):
    data = report()
    data[field] = value
    with pytest.raises(ValueError):
        evaluate_presentation_event(data, annotation(), .5)


def test_model_selection_uses_only_corresponding_calibration_membership():
    protocol = {"models": {"candidate": {"split": "iid", "window_threshold": .55}},
                "cohorts": {"iid": {"calibration_clip_paths": ["a.mp4"], "test_clip_paths": ["unread.mp4"]}},
                "annotations": {"a.mp4": annotation()},
                "calibration": {"threshold_grid": [0, .5, 1], "episode_budget_per_clip": .5}}
    data = report()
    data["scores"] = {"candidate": data.pop("window_probabilities")}
    result = select_models(protocol, [data])
    assert result["candidate"]["calibration_clips"] == 1
    assert result["candidate"]["selected"]["threshold"] == .55
    assert len(result["candidate"]["curve"]) == 4
    with pytest.raises(KeyError):
        model_reports(protocol, [data], "candidate", "test")


def test_cohort_union_deduplicates_across_splits_but_not_within_splits():
    protocol = {"cohorts": {"iid": {"calibration_clip_paths": ["a"], "test_clip_paths": ["b"]},
                            "geographic": {"calibration_clip_paths": ["a", "c"], "test_clip_paths": ["b", "d"]}}}
    assert cohort_paths(protocol, "calibration") == ["a", "c"]
    assert cohort_paths(protocol, "test") == ["b", "d"]
    invalid = deepcopy(protocol)
    invalid["cohorts"]["iid"]["test_clip_paths"] += ["a"]
    with pytest.raises(ValueError, match="disjoint"):
        cohort_paths(invalid, "calibration")


def test_event_queue_freezes_train_selection_before_test_and_has_no_training():
    steps = presentation_event_steps()
    assert len(steps) == 4
    assert steps[0][1][-1] == "calibration"
    assert steps[1][1][-1] == "calibrate"
    assert steps[2][1][-1] == "test"
    assert steps[3][1][-1] == "evaluate"
    assert not any("train_" in token for _, arguments in steps for token in arguments)


def test_real_decoder_and_motion_feature_shapes_on_cpu_fixture(tmp_path, monkeypatch):
    import cv2
    from scripts import replay_presentation_events as replay

    video = tmp_path / "fixture.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    for offset in range(61):
        image = np.zeros((48, 64, 3), dtype=np.uint8)
        image[10:20, offset % 40:offset % 40 + 10] = 255
        writer.write(image)
    writer.release()
    monkeypatch.setattr(replay, "track_vehicle_frames", lambda detector, images, indices, **kwargs: {index: [] for index in indices})
    matrix, windows, rejected = replay.clip_features(video, np.arange(61) / 15, object(), "cpu")
    assert matrix.shape == (2, 15, 14) and np.all(np.isfinite(matrix))
    assert [window["end_time_sec"] for window in windows] == [2, 4] and rejected == []


def test_batch_hashes_reject_changed_clip_outputs_and_missing_cohort(tmp_path):
    import json
    from scripts.prepare_presentation_protocol import sha256
    from scripts.replay_presentation_events import validate_batch

    protocol = {"cohorts": {"iid": {"calibration_clip_paths": ["a.mp4"], "test_clip_paths": ["b.mp4"]}},
                "input_media": [{"path": "a.mp4", "sha256": "source"}], "models": {"model": {}}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    features = tmp_path / "features.npz"
    np.savez_compressed(features, features=np.zeros((1, 15, 14)))
    data = report()
    data.update({"protocol_sha256": sha256(path), "features_sha256": sha256(features), "source_sha256": "source",
                 "signature": "sig", "scores": {"model": data.pop("window_probabilities")}})
    result = tmp_path / "report.json"
    result.write_text(json.dumps(data))
    record = {"report": result.name, "report_sha256": sha256(result), "features": features.name, "features_sha256": sha256(features)}
    batch = {"protocol_sha256": sha256(path), "cohort": "calibration", "smoke_clips": None,
             "selection_sha256": None, "signature": "sig", "clips": {"a.mp4": record}}
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch))
    assert len(validate_batch(tmp_path, path, protocol, "calibration")[1]) == 1
    data["source_sha256"] = "changed"
    result.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="changed"):
        validate_batch(tmp_path, path, protocol, "calibration")
    record["report_sha256"] = sha256(result)
    batch_path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match="source"):
        validate_batch(tmp_path, path, protocol, "calibration")
    batch["clips"] = {}
    batch_path.write_text(json.dumps(batch))
    with pytest.raises(ValueError, match="every frozen"):
        validate_batch(tmp_path, path, protocol, "calibration")
