from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from illegal_parking.accident_dataset import AccidentClip
from illegal_parking.incident_presentation import (
    normalized_presentation_times, presentation_grid, paired_presentation_windows,
    read_presentation_frames, valid_fixed_window,
)
from scripts.prepare_presentation_protocol import assign_partitions, probe_times


def clip(**kwargs):
    base = AccidentClip(Path("a.mp4"), "t-bone", 6, 90, (.1, .1, .2, .2),
                        "Taiwan", "intersection", "clear", "day", "Good", 300, 20, 360, 640, "train", "train")
    return replace(base, **kwargs)


@pytest.mark.parametrize("times", [[], [0, 0], [1, 0], [float("nan")], [float("inf")], [[0, 1]]])
def test_presentation_clock_rejects_ambiguous_timestamps(times):
    with pytest.raises(ValueError):
        normalized_presentation_times(times)


def test_grid_uses_presentation_time_instead_of_frame_index_rate():
    pts = np.cumsum(np.r_[0, np.tile([.04, .12], 100)]) + 5
    indices, targets = presentation_grid(pts)
    normalized = pts - 5
    assert np.all(normalized[indices] >= targets - 1e-6)
    assert np.all(normalized[np.maximum(0, indices[1:] - 1)] < targets[1:] + 1e-6)
    assert not np.array_equal(indices, np.ceil(targets / np.mean(np.diff(pts))).astype(int))


def test_canonical_clock_produces_matched_legacy_and_fixed_frames():
    rows, reason = paired_presentation_windows(clip(), np.arange(301) / 15)
    assert reason is None and len(rows) == 2
    for row in rows:
        assert row["legacy_indices"] == row["fixed_indices"]
        assert row["event_time_sec"] == 6
        assert len(row["fixed_indices"]) == 16
        assert row["fixed_times_sec"][-1] - row["fixed_times_sec"][0] == pytest.approx(2)


def test_supplied_terminal_annotation_is_not_silently_clamped():
    _, reason = paired_presentation_windows(clip(accident_frame=300), np.arange(301) / 15)
    assert reason == "legacy_event_after_last_sampled_frame"
    _, reason = paired_presentation_windows(clip(accident_frame=301), np.arange(301) / 15)
    assert reason == "supplied_annotation_outside_decoded_support"


def test_sparse_or_discontinuous_video_does_not_manufacture_fixed_cadence():
    assert paired_presentation_windows(clip(), np.arange(301) / 4)[1] == "fixed_event_cadence_or_lateness_violation"
    pts = np.arange(301) / 15
    pts[80:] += .8
    assert paired_presentation_windows(clip(), pts)[1] == "fixed_event_cadence_or_lateness_violation"


def test_fixed_window_rejects_repeated_frames_and_excess_lateness():
    assert not valid_fixed_window(np.zeros(16, dtype=int), np.arange(16) / 7.5, np.arange(32) / 15)
    assert not valid_fixed_window(np.arange(16), np.arange(16) / 7.5, np.arange(16) / 7.5 + .14)


def test_frozen_partitions_keep_clips_whole_and_tests_unchanged():
    rows = [{"path": f"{index}.mp4", "target": target, "iid_split": "train" if index < 20 else "test"}
            for index in range(25) for target in (0, 1)]
    assign_partitions(rows, "iid")
    first = [row["iid_partition"] for row in rows]
    assign_partitions(rows, "iid")
    assert first == [row["iid_partition"] for row in rows]
    sets = {part: {row["path"] for row in rows if row["iid_partition"] == part} for part in ("fit", "calibration", "test")}
    assert len(sets["calibration"]) == 4
    assert not (sets["fit"] & sets["calibration"] or sets["test"] & (sets["fit"] | sets["calibration"]))


def test_probe_sorts_pts_but_rejects_container_count_mismatch(monkeypatch):
    result = {"packets": [{"pts_time": value} for value in [.2, 0, .1]], "streams": [{"nb_frames": "3"}]}
    monkeypatch.setattr("scripts.prepare_presentation_protocol.subprocess.run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(result)))
    assert probe_times(Path("a"), Path("ffprobe")) == [0, .1, .2]
    result["streams"][0]["nb_frames"] = "4"
    with pytest.raises(ValueError, match="frame count"):
        probe_times(Path("a"), Path("ffprobe"))


def test_decoded_selected_frames_must_match_frozen_pts_and_indices(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "clock.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 48))
    assert writer.isOpened()
    for index in range(40):
        writer.write(np.full((48, 64, 3), index * 5, dtype=np.uint8))
    writer.release()
    selected = [0, 2, 6, 16, 30]
    frames = read_presentation_frames(path, selected, np.asarray(selected) / 15, target_width=32)
    assert len(frames) == 5 and frames[0].shape == (24, 32, 3)
    assert np.mean(frames[-1]) > np.mean(frames[0]) + 100
    with pytest.raises(ValueError, match="PTS"):
        read_presentation_frames(path, selected, np.asarray(selected) / 15 + .2)
    with pytest.raises(ValueError, match="strictly increasing"):
        read_presentation_frames(path, [0, 0], [0, 0])
