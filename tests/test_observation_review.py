import hashlib
import json

import cv2
import numpy as np
import pytest

from illegal_parking.observation_review import load_review_observation, save_observation_review, summarize_observation_reviews


@pytest.fixture
def observation(tmp_path):
    video = tmp_path / "clean.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 7.5, (64, 48))
    assert writer.isOpened()
    for index in range(5):
        writer.write(np.full((48, 64, 3), index * 40, dtype=np.uint8))
    writer.release()
    report = {"streaming_inference": True, "clean_video": str(video),
              "clean_video_sha256": hashlib.sha256(video.read_bytes()).hexdigest(),
              "frame_times_sec": [0, 0.2, 0.4, 2, 2.2], "event_clock": "decoder observation clock",
              "source_mode": "mjpeg", "model_sha256": "model", "detector_sha256": "detector",
              "hop_steps": 15, "min_positive_windows": 2, "threshold": 0.9,
              "events": [{"trigger_time_sec": 0.3}, {"trigger_time_sec": 1}, {"trigger_time_sec": 2.2}]}
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    return path


def save(path, output, **kwargs):
    return save_observation_review(path, output, reviewer_id="test-fixture-reviewer", label="normal", all_frames_reviewed=True, model_outputs_not_consulted=True, **kwargs)


def test_normal_exposure_excludes_capture_gaps_and_counts_each_episode_once(observation, tmp_path):
    path = tmp_path / "review.json"
    save(observation, path)
    result = summarize_observation_reviews([path])
    assert result["normal_camera_hours"] == pytest.approx(0.6 / 3600)
    assert result["normal_alert_episodes"] == 2
    assert result["normal_observations"][0]["excluded_gap_sec"] == pytest.approx(1.6)
    assert result["poisson_one_sided_upper_95_per_hour"] > result["alerts_per_normal_camera_hour"]


@pytest.mark.parametrize("label,complete,blinded", [("uncertain", True, True), ("normal", False, True), ("normal", True, False)])
def test_unreviewed_or_uncertain_exposure_never_becomes_a_zero_false_alarm_claim(observation, tmp_path, label, complete, blinded):
    path = tmp_path / "review.json"
    save_observation_review(observation, path, reviewer_id="test-fixture", label=label, all_frames_reviewed=complete, model_outputs_not_consulted=blinded)
    result = summarize_observation_reviews([path])
    assert result["normal_camera_hours"] == 0
    assert result["alerts_per_normal_camera_hour"] is None
    assert result["poisson_one_sided_upper_95_per_hour"] is None


def test_duplicate_reviews_do_not_duplicate_exposure_and_conflicts_are_excluded(observation, tmp_path):
    first, second, conflict = [tmp_path / name for name in ("first.json", "second.json", "conflict.json")]
    save(observation, first)
    save(observation, second)
    result = summarize_observation_reviews([first, second])
    assert len(result["normal_observations"]) == 1
    assert result["normal_camera_hours"] == pytest.approx(0.6 / 3600)
    save_observation_review(observation, conflict, reviewer_id="other-test-reviewer", label="incident_present", all_frames_reviewed=True, model_outputs_not_consulted=True)
    disputed = summarize_observation_reviews([first, conflict])
    assert disputed["normal_camera_hours"] == 0
    assert len(disputed["conflicting_observations"]) == 1


def test_media_tampering_and_source_report_changes_are_rejected(observation, tmp_path):
    annotation = tmp_path / "review.json"
    save(observation, annotation)
    report = json.loads(observation.read_text())
    report["threshold"] = 0.8
    observation.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="Source report changed"):
        summarize_observation_reviews([annotation])
    video = tmp_path / "clean.mp4"
    video.write_bytes(b"changed")
    with pytest.raises(ValueError, match="video hash"):
        load_review_observation(observation)


def test_zero_alert_bound_and_file_replay_exclusion(observation, tmp_path):
    report = json.loads(observation.read_text())
    report["events"] = []
    observation.write_text(json.dumps(report))
    annotation = tmp_path / "review.json"
    save(observation, annotation)
    result = summarize_observation_reviews([annotation])
    assert result["poisson_one_sided_upper_95_per_hour"] == pytest.approx(-np.log(0.05) / result["normal_camera_hours"])
    report["source_mode"] = "file"
    observation.write_text(json.dumps(report))
    replay = tmp_path / "replay.json"
    save(observation, replay)
    assert summarize_observation_reviews([replay])["normal_camera_hours"] == 0


def test_annotation_is_append_only_and_requires_attribution(observation, tmp_path):
    path = tmp_path / "review.json"
    save(observation, path)
    with pytest.raises(FileExistsError):
        save(observation, path)
    with pytest.raises(ValueError, match="reviewer"):
        save_observation_review(observation, path, reviewer_id=" ", label="normal", all_frames_reviewed=True, model_outputs_not_consulted=True)


def test_source_clock_must_match_actual_clean_video_frame_count(observation):
    report = json.loads(observation.read_text())
    report["frame_times_sec"] = [0, 1]
    observation.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="playback frames"):
        load_review_observation(observation)


def test_duplicate_media_cannot_supply_alternative_exposure_clocks(observation, tmp_path):
    first = tmp_path / "first_review.json"
    save(observation, first)
    alternative = tmp_path / "alternative_report.json"
    report = json.loads(observation.read_text())
    report["frame_times_sec"] = [0, 0.3, 0.6, 1, 1.4]
    alternative.write_text(json.dumps(report))
    second = tmp_path / "second_review.json"
    save(alternative, second)
    with pytest.raises(ValueError, match="conflicting observation clocks"):
        summarize_observation_reviews([first, second])
