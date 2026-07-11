from __future__ import annotations

import csv
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class AccidentClip:
    relative_path: Path
    collision_type: str
    accident_time_sec: float
    accident_frame: int
    bbox_normalized: tuple[float, float, float, float]
    region: str
    scene_layout: str
    weather: str
    day_time: str
    quality: str
    frame_count: int
    duration_sec: float
    height: int
    width: int
    iid_split: str
    geographic_split: str

    @property
    def frames_per_second(self) -> float:
        return self.frame_count / self.duration_sec

    @property
    def effective_accident_frame(self) -> int:
        return min(self.accident_frame, self.frame_count - 1)

    @property
    def accident_frame_adjusted(self) -> bool:
        return self.effective_accident_frame != self.accident_frame

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["relative_path"] = self.relative_path.as_posix()
        payload["frames_per_second"] = self.frames_per_second
        payload["effective_accident_frame"] = self.effective_accident_frame
        payload["accident_frame_adjusted"] = self.accident_frame_adjusted
        return payload


@dataclass(frozen=True)
class TemporalWindow:
    relative_path: Path
    start_frame: int
    end_frame: int
    label: Literal["normal", "incident"]
    collision_type: str
    accident_frame: int
    source_accident_frame: int
    iid_split: str
    geographic_split: str

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["relative_path"] = self.relative_path.as_posix()
        return payload


@dataclass(frozen=True)
class AccidentManifestSummary:
    total_clips: int
    total_duration_sec: float
    collision_types: dict[str, int]
    iid_splits: dict[str, int]
    geographic_splits: dict[str, int]
    regions: dict[str, int]
    adjusted_accident_frame_count: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AccidentFileValidation:
    expected_count: int
    present_count: int
    missing_count: int
    missing_paths: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def load_accident_manifest(path: str | Path) -> list[AccidentClip]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"ACCIDENT manifest has no rows: {manifest_path}")
    return [_parse_clip(row, row_number=index + 2) for index, row in enumerate(rows)]


def build_temporal_windows(
    clip: AccidentClip,
    window_frames: int,
    negative_gap_frames: int = 0,
) -> list[TemporalWindow]:
    if window_frames <= 1:
        raise ValueError("window_frames must be greater than one")
    if negative_gap_frames < 0:
        raise ValueError("negative_gap_frames must be non-negative")
    if clip.frame_count < window_frames:
        return []

    event_frame = clip.effective_accident_frame
    incident_start = event_frame - window_frames // 2
    incident_start = max(0, min(incident_start, clip.frame_count - window_frames))
    incident_end = incident_start + window_frames

    windows: list[TemporalWindow] = []
    normal_end = incident_start - negative_gap_frames
    normal_start = normal_end - window_frames
    if normal_start >= 0:
        windows.append(_window(clip, normal_start, normal_end, "normal"))

    windows.append(_window(clip, incident_start, incident_end, "incident"))
    return windows


def summarize_accident_manifest(clips: list[AccidentClip]) -> AccidentManifestSummary:
    return AccidentManifestSummary(
        total_clips=len(clips),
        total_duration_sec=sum(clip.duration_sec for clip in clips),
        collision_types=_counts(clip.collision_type for clip in clips),
        iid_splits=_counts(clip.iid_split for clip in clips),
        geographic_splits=_counts(clip.geographic_split for clip in clips),
        regions=_counts(clip.region for clip in clips),
        adjusted_accident_frame_count=sum(clip.accident_frame_adjusted for clip in clips),
    )


def validate_accident_files(
    clips: list[AccidentClip],
    dataset_root: str | Path,
) -> AccidentFileValidation:
    root = Path(dataset_root)
    missing_paths = [
        clip.relative_path.as_posix()
        for clip in clips
        if not (root / clip.relative_path).is_file()
    ]
    return AccidentFileValidation(
        expected_count=len(clips),
        present_count=len(clips) - len(missing_paths),
        missing_count=len(missing_paths),
        missing_paths=missing_paths,
    )


def _parse_clip(row: dict[str, str], row_number: int) -> AccidentClip:
    try:
        bbox = tuple(float(row[key]) for key in ("x1", "y1", "x2", "y2"))
        frame_count = int(row["no_frames"])
        duration_sec = float(row["duration"])
        accident_frame = int(row["accident_frame"])
        clip = AccidentClip(
            relative_path=Path(row["path"]),
            collision_type=row["type"].strip(),
            accident_time_sec=float(row["accident_time"]),
            accident_frame=accident_frame,
            bbox_normalized=bbox,
            region=row["region"].strip(),
            scene_layout=row["scene_layout"].strip(),
            weather=row["weather"].strip(),
            day_time=row["day_time"].strip(),
            quality=row["quality"].strip(),
            frame_count=frame_count,
            duration_sec=duration_sec,
            height=int(row["height"]),
            width=int(row["width"]),
            iid_split=row["split_in_distribution"].strip(),
            geographic_split=row["split_geo_aware"].strip(),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid ACCIDENT manifest row {row_number}: {exc}") from exc

    x1, y1, x2, y2 = clip.bbox_normalized
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(f"Invalid normalized bbox in ACCIDENT manifest row {row_number}: {clip.bbox_normalized}")
    if clip.duration_sec <= 0 or clip.frame_count <= 0:
        raise ValueError(f"Invalid clip duration/frame count in ACCIDENT manifest row {row_number}")
    if not 0 <= clip.accident_frame <= clip.frame_count:
        raise ValueError(f"Invalid accident_frame in ACCIDENT manifest row {row_number}")
    return clip


def _window(
    clip: AccidentClip,
    start_frame: int,
    end_frame: int,
    label: Literal["normal", "incident"],
) -> TemporalWindow:
    return TemporalWindow(
        relative_path=clip.relative_path,
        start_frame=start_frame,
        end_frame=end_frame,
        label=label,
        collision_type=clip.collision_type,
        accident_frame=clip.effective_accident_frame,
        source_accident_frame=clip.accident_frame,
        iid_split=clip.iid_split,
        geographic_split=clip.geographic_split,
    )


def _counts(values) -> dict[str, int]:
    return dict(sorted(Counter(values).items()))
