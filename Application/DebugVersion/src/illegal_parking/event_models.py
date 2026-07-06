from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DecisionStatus = Literal[
    "NO_EVENT",
    "OBSERVE",
    "CANDIDATE",
    "REVIEW_REQUIRED",
    "LIKELY_VIOLATION",
    "INSUFFICIENT_EVIDENCE",
]


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    def as_xyxy(self) -> tuple[int, int, int, int]:
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(frozen=True)
class TrackSnapshot:
    track_id: int
    class_name: str
    confidence: float
    bbox: BBox
    first_seen_ts: float
    last_seen_ts: float
    speed_px_per_sec: float
    stable_frame_count: int

    @property
    def dwell_time_sec(self) -> float:
        return max(0.0, self.last_seen_ts - self.first_seen_ts)


@dataclass(frozen=True)
class RuleEvidence:
    dwell_time_sec: float
    speed_px_per_sec: float
    redline_overlap_ratio: float
    in_no_parking_roi: bool
    stable_frame_count: int
    mask_source: str | None = None
    vehicle_mask_area: int = 0
    restricted_overlap_pixels: int = 0
    mask_restricted_overlap_ratio: float = 0.0
    mask_restricted_coverage_ratio: float = 0.0
    footprint_area: int = 0
    footprint_overlap_pixels: int = 0
    footprint_overlap_ratio: float = 0.0


@dataclass(frozen=True)
class ViolationCandidate:
    event_id: str
    track_id: int
    class_name: str
    bbox: BBox
    status: DecisionStatus
    is_candidate: bool
    rule_evidence: RuleEvidence
    reasons: list[str] = field(default_factory=list)
