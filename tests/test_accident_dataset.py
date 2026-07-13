from __future__ import annotations

from pathlib import Path

import pytest

from illegal_parking.accident_dataset import (
    build_temporal_windows,
    load_accident_manifest,
    stratified_sample_clips,
    summarize_accident_manifest,
    validate_accident_files,
)


CSV_HEADER = (
    "path,type,rollover,accident_time,accident_frame,center_x,center_y,"
    "x1,y1,x2,y2,region,scene_layout,weather,day_time,quality,no_frames,"
    "duration,height,width,split_in_distribution,split_geo_aware\n"
)


def _write_manifest(path: Path) -> None:
    path.write_text(
        CSV_HEADER
        + "real_videos/a.mp4,rear-end,0,4.0,60,0.5,0.5,0.4,0.4,0.6,0.6,"
        "Taiwan,intersection,rain,day,Good,150,10.0,720,1280,train,test\n"
        + "real_videos/b.mp4,t-bone,0,2.0,30,0.3,0.4,0.2,0.3,0.4,0.5,"
        "Japan,intersection,normal,night,Poor,90,6.0,1080,1920,test,train\n",
        encoding="utf-8",
    )


def test_load_accident_manifest_parses_typed_fields(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata-real.csv"
    _write_manifest(manifest)

    clips = load_accident_manifest(manifest)

    assert len(clips) == 2
    assert clips[0].relative_path == Path("real_videos/a.mp4")
    assert clips[0].collision_type == "rear-end"
    assert clips[0].accident_frame == 60
    assert clips[0].bbox_normalized == pytest.approx((0.4, 0.4, 0.6, 0.6))
    assert clips[0].frames_per_second == pytest.approx(15.0)
    assert clips[0].iid_split == "train"
    assert clips[0].geographic_split == "test"


def test_load_accident_manifest_rejects_invalid_normalized_bbox(tmp_path: Path) -> None:
    manifest = tmp_path / "invalid.csv"
    manifest.write_text(
        CSV_HEADER
        + "real_videos/a.mp4,rear-end,0,4.0,60,0.5,0.5,0.7,0.4,0.6,0.6,"
        "Taiwan,intersection,rain,day,Good,150,10.0,720,1280,train,test\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="normalized bbox"):
        load_accident_manifest(manifest)


def test_manifest_preserves_end_boundary_event_and_exposes_effective_frame(tmp_path: Path) -> None:
    manifest = tmp_path / "boundary.csv"
    manifest.write_text(
        CSV_HEADER
        + "real_videos/a.mp4,single,0,10.0,150,0.5,0.5,0.4,0.4,0.6,0.6,"
        "Taiwan,intersection,rain,day,Good,150,10.0,720,1280,train,test\n",
        encoding="utf-8",
    )

    clip = load_accident_manifest(manifest)[0]
    windows = build_temporal_windows(clip, window_frames=24, negative_gap_frames=6)

    assert clip.accident_frame == 150
    assert clip.effective_accident_frame == 149
    assert clip.accident_frame_adjusted is True
    assert windows[-1].accident_frame == 149
    assert windows[-1].source_accident_frame == 150
    assert windows[-1].start_frame <= 149 < windows[-1].end_frame


def test_build_temporal_windows_creates_normal_and_incident_examples(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata-real.csv"
    _write_manifest(manifest)
    clip = load_accident_manifest(manifest)[0]

    windows = build_temporal_windows(clip, window_frames=24, negative_gap_frames=6)

    assert [window.label for window in windows] == ["normal", "incident"]
    assert windows[0].end_frame <= clip.accident_frame - 6
    assert windows[0].end_frame <= windows[1].start_frame - 6
    assert windows[1].start_frame <= clip.accident_frame < windows[1].end_frame
    assert all(window.end_frame - window.start_frame == 24 for window in windows)


def test_build_temporal_windows_skips_normal_window_when_clip_is_too_short(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata-real.csv"
    _write_manifest(manifest)
    clip = load_accident_manifest(manifest)[1]

    windows = build_temporal_windows(clip, window_frames=24, negative_gap_frames=12)

    assert [window.label for window in windows] == ["incident"]


def test_summarize_accident_manifest_reports_splits_and_classes(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata-real.csv"
    _write_manifest(manifest)

    summary = summarize_accident_manifest(load_accident_manifest(manifest))

    assert summary.total_clips == 2
    assert summary.collision_types == {"rear-end": 1, "t-bone": 1}
    assert summary.iid_splits == {"test": 1, "train": 1}
    assert summary.geographic_splits == {"test": 1, "train": 1}
    assert summary.total_duration_sec == pytest.approx(16.0)
    assert summary.adjusted_accident_frame_count == 0


def test_validate_accident_files_reports_present_and_missing_clips(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata-real.csv"
    _write_manifest(manifest)
    clips = load_accident_manifest(manifest)
    existing = tmp_path / clips[0].relative_path
    existing.parent.mkdir(parents=True)
    existing.write_bytes(b"video")

    report = validate_accident_files(clips, tmp_path)

    assert report.expected_count == 2
    assert report.present_count == 1
    assert report.missing_count == 1
    assert report.missing_paths == ["real_videos/b.mp4"]


def test_stratified_sample_is_bounded_and_reproducible(tmp_path: Path) -> None:
    manifest = tmp_path / "metadata-real.csv"
    _write_manifest(manifest)
    clips = load_accident_manifest(manifest)

    first = stratified_sample_clips(clips, max_clips=1, seed=7)
    second = stratified_sample_clips(clips, max_clips=1, seed=7)

    assert len(first) == 1
    assert first == second
    assert stratified_sample_clips(clips, max_clips=10, seed=7) == clips
