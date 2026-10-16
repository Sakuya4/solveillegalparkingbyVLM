from dataclasses import replace
from pathlib import Path
import json
from types import SimpleNamespace
from collections import Counter

import numpy as np
import pytest

from illegal_parking.accident_dataset import AccidentClip
from illegal_parking.incident_cadence import paired_cadence_windows, source_time_grid
from scripts.audit_incident_cadence import inspect_presentation_clock


def clip(**kwargs):
    base = AccidentClip(Path("real_videos/a.mp4"), "t-bone", 6, 90, (.1, .1, .2, .2),
                        "Taiwan", "intersection", "clear", "day", "Good", 300, 20, 360, 640, "train", "train")
    return replace(base, **kwargs)


@pytest.mark.parametrize("fps", [7.5, 7.966805, 15, 24.98, 29.97, 49.7])
def test_global_grid_selects_first_available_frame_without_duplicates(fps):
    indices = np.asarray(source_time_grid(300, fps))
    times = np.arange(len(indices)) / 7.5
    assert np.all(np.diff(indices) > 0)
    assert np.all(indices / fps + 1e-8 >= times)
    assert np.all((indices - 1) / fps < times + 1e-8)


@pytest.mark.parametrize("fps", [0, -1, float("nan"), float("inf"), 3.9])
def test_invalid_or_low_sampling_rate_cannot_create_artificial_frames(fps):
    with pytest.raises(ValueError):
        source_time_grid(100, fps)


def test_legacy_and_fixed_windows_match_at_the_canonical_source_rate():
    rows, reason = paired_cadence_windows(clip(), 15, 300)
    assert reason is None and len(rows) == 2
    for row in rows:
        assert row["legacy_indices"] == row["fixed_indices"]
        assert len(row["fixed_indices"]) == 16


def test_pairing_excludes_unobservable_events_and_container_disagreements():
    assert paired_cadence_windows(clip(), 3.9, 300) == ([], "source_fps_below_target")
    assert paired_cadence_windows(clip(), 15, 299) == ([], "container_metadata_frame_count_mismatch")
    assert paired_cadence_windows(clip(accident_frame=299), 15, 300) == ([], "event_after_last_sampled_frame")


def test_early_incident_does_not_invent_pre_event_negative_context():
    rows, reason = paired_cadence_windows(clip(accident_frame=3), 15, 300)
    assert reason is None and [row["label"] for row in rows] == ["incident"]


def test_extra_container_terminal_frame_does_not_change_legacy_support():
    regular, _ = paired_cadence_windows(clip(), 15, 300)
    extra, reason = paired_cadence_windows(clip(), 15, 301)
    assert reason is None
    for original, extended in zip(regular, extra):
        assert original["fixed_indices"] == extended["fixed_indices"]
        assert original["legacy_indices"] == extended["legacy_indices"]
        assert extended["support_frame_count"] == 300


def test_packet_clock_is_checked_against_decoded_pts_not_average_fps(monkeypatch):
    cv2 = pytest.importorskip("cv2")
    offsets = [0, .066667, .2]

    class Capture:
        index = -1

        def get(self, key):
            return 15 if key == cv2.CAP_PROP_FPS else 1000 * (2 + offsets[self.index])

        def read(self):
            self.index += 1
            return self.index < 3, None

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", lambda *args: Capture())
    monkeypatch.setattr("scripts.audit_incident_cadence.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"packets": [{"pts_time": 5 + value} for value in offsets]})))
    result = inspect_presentation_clock(Path("unused.mp4"), Path("ffprobe"))
    assert result["packet_frame_alignment_verified"]
    json.dumps(result, allow_nan=False)
    assert result["decoded_vs_packet_max_abs_error_sec"] < 1e-8
    assert result["frame_index_fps_clock_max_abs_error_sec"] == pytest.approx(.2 - 2 / 15)


@pytest.mark.parametrize("timestamps", [[], [0, 0], [float("nan")], [float("inf")]])
def test_invalid_packet_pts_cannot_certify_a_physical_clock(monkeypatch, timestamps):
    monkeypatch.setattr("scripts.audit_incident_cadence.subprocess.run", lambda *args, **kwargs:
                        SimpleNamespace(stdout=json.dumps({"packets": [{"pts_time": value} for value in timestamps]})))
    with pytest.raises(ValueError, match="timestamps"):
        inspect_presentation_clock(Path("unused.mp4"), Path("ffprobe"))


def test_published_corpus_audit_and_training_only_clock_traces_are_consistent():
    audit_path = Path(__file__).resolve().parents[1] / "docs/results/cadence_source_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    media = {row["path"]: row for row in audit["input_media"]}
    assert len(media) == audit["requested_clips"] == 2027
    counts = Counter(str(row["container_frame_count"] - row["metadata_frame_count"]) for row in media.values())
    assert dict(counts) == audit["frame_count_differences"]
    assert all(len(row["sha256"]) == 64 for row in media.values())
    assert len(audit["train_only_presentation_clock_audits"]) == 7
    for trace in audit["train_only_presentation_clock_audits"]:
        source = media[trace["path"]]
        assert source["iid_split"] == source["geographic_split"] == "train"
        pts = np.asarray(trace["presentation_timestamps_sec"])
        assert len(pts) == trace["decoded_frames"] == source["container_frame_count"]
        assert trace["packet_frame_alignment_verified"]
        error = np.max(np.abs(pts - pts[0] - np.arange(len(pts)) / source["source_fps"]))
        assert trace["frame_index_fps_clock_max_abs_error_sec"] == pytest.approx(error)
