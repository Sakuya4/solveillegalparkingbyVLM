from __future__ import annotations

from dataclasses import dataclass

from .event_models import RuleEvidence, TrackSnapshot, ViolationCandidate


@dataclass(frozen=True)
class ViolationEngineConfig:
    vehicle_classes: tuple[str, ...] = ("car", "truck", "bus", "motorcycle", "bicycle")
    min_confidence: float = 0.25
    dwell_threshold_sec: float = 30.0
    stationary_speed_threshold: float = 0.1
    min_redline_overlap_ratio: float = 0.003
    min_stable_frames: int = 3


class ViolationEngine:
    def __init__(self, config: ViolationEngineConfig | None = None):
        self.config = config or ViolationEngineConfig()

    def evaluate(
        self,
        track: TrackSnapshot,
        redline_overlap_ratio: float,
        in_no_parking_roi: bool,
    ) -> ViolationCandidate:
        reasons: list[str] = []
        class_name = track.class_name.lower()
        evidence = RuleEvidence(
            dwell_time_sec=track.dwell_time_sec,
            speed_px_per_sec=track.speed_px_per_sec,
            redline_overlap_ratio=redline_overlap_ratio,
            in_no_parking_roi=in_no_parking_roi,
            stable_frame_count=track.stable_frame_count,
        )

        if class_name not in self.config.vehicle_classes:
            reasons.append("class is not configured as vehicle")
            return self._candidate(track, "NO_EVENT", False, evidence, reasons)

        if track.confidence < self.config.min_confidence:
            reasons.append("confidence below threshold")
            return self._candidate(track, "NO_EVENT", False, evidence, reasons)

        if track.stable_frame_count < self.config.min_stable_frames:
            reasons.append("track not stable yet")
            return self._candidate(track, "OBSERVE", False, evidence, reasons)

        has_redline_overlap = redline_overlap_ratio >= self.config.min_redline_overlap_ratio
        has_restricted_zone = has_redline_overlap or in_no_parking_roi
        if has_redline_overlap:
            reasons.append("redline overlap")
        if in_no_parking_roi:
            reasons.append("inside no-parking roi")
        if not has_restricted_zone:
            reasons.append("no restricted-zone evidence")
            return self._candidate(track, "OBSERVE", False, evidence, reasons)

        if track.dwell_time_sec < self.config.dwell_threshold_sec:
            reasons.append("dwell time below threshold")
            return self._candidate(track, "OBSERVE", False, evidence, reasons)

        reasons.append("dwell time threshold met")
        if track.speed_px_per_sec <= self.config.stationary_speed_threshold:
            reasons.append("stationary speed threshold met")

        return self._candidate(track, "CANDIDATE", True, evidence, reasons)

    def _candidate(
        self,
        track: TrackSnapshot,
        status,
        is_candidate: bool,
        evidence: RuleEvidence,
        reasons: list[str],
    ) -> ViolationCandidate:
        return ViolationCandidate(
            event_id=f"track_{track.track_id}_{int(track.last_seen_ts)}",
            track_id=track.track_id,
            class_name=track.class_name.lower(),
            bbox=track.bbox,
            status=status,
            is_candidate=is_candidate,
            rule_evidence=evidence,
            reasons=reasons,
        )
