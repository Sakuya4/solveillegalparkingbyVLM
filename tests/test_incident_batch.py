from __future__ import annotations

from pathlib import Path

from illegal_parking.accident_dataset import AccidentClip
from illegal_parking.incident_batch import report_filename, select_evaluation_clips


def _clip(path: str, iid_split: str, geographic_split: str, collision_type: str) -> AccidentClip:
    return AccidentClip(
        relative_path=Path(path),
        collision_type=collision_type,
        accident_time_sec=2.5,
        accident_frame=25,
        bbox_normalized=(0.2, 0.2, 0.4, 0.4),
        region="Taiwan",
        scene_layout="intersection",
        weather="normal",
        day_time="day",
        quality="Good",
        frame_count=100,
        duration_sec=10.0,
        height=720,
        width=1280,
        iid_split=iid_split,
        geographic_split=geographic_split,
    )


def test_select_evaluation_clips_uses_only_requested_test_split() -> None:
    clips = [
        _clip("real_videos/train.mp4", "train", "test", "rear-end"),
        _clip("real_videos/iid-a.mp4", "test", "train", "t-bone"),
        _clip("real_videos/iid-b.mp4", "test", "test", "rear-end"),
    ]

    selected = select_evaluation_clips(clips, split_scheme="iid", max_clips=None, seed=42)

    assert [clip.relative_path.as_posix() for clip in selected] == [
        "real_videos/iid-a.mp4",
        "real_videos/iid-b.mp4",
    ]


def test_select_evaluation_clips_is_reproducible_when_sampled() -> None:
    clips = [
        _clip(f"region-{index}/clip.mp4", "test", "test", "rear-end" if index % 2 else "t-bone")
        for index in range(12)
    ]

    first = select_evaluation_clips(clips, split_scheme="geographic", max_clips=5, seed=7)
    second = select_evaluation_clips(clips, split_scheme="geographic", max_clips=5, seed=7)

    assert [clip.relative_path for clip in first] == [clip.relative_path for clip in second]
    assert len(first) == 5


def test_report_filename_preserves_parent_path_to_avoid_collisions() -> None:
    assert report_filename(Path("taiwan/a.mp4")) != report_filename(Path("japan/a.mp4"))
    assert report_filename(Path("taiwan/a.mp4")).endswith(".json")
