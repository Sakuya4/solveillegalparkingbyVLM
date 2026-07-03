from __future__ import annotations

from dataclasses import dataclass

from .detectors import DetectionRecord
from .event_models import BBox, TrackSnapshot


@dataclass
class _TrackState:
    track_id: int
    detection: DetectionRecord
    first_seen_ts: float
    last_seen_ts: float
    last_center: tuple[int, int]
    speed_px_per_sec: float = 0.0
    stable_frame_count: int = 1
    lost_count: int = 0


class SimpleIouTracker:
    def __init__(self, iou_threshold: float = 0.3, max_lost: int = 10):
        self.iou_threshold = iou_threshold
        self.max_lost = max_lost
        self._next_id = 1
        self._tracks: dict[int, _TrackState] = {}

    def update(self, detections: list[DetectionRecord], timestamp_sec: float) -> list[TrackSnapshot]:
        for track in self._tracks.values():
            track.lost_count += 1

        matched_track_ids: set[int] = set()
        for detection in detections:
            track = self._match_track(detection, matched_track_ids)
            if track is None:
                track = _TrackState(
                    track_id=self._next_id,
                    detection=detection,
                    first_seen_ts=timestamp_sec,
                    last_seen_ts=timestamp_sec,
                    last_center=detection.bbox.center,
                )
                self._next_id += 1
                self._tracks[track.track_id] = track
            else:
                elapsed = max(1e-6, timestamp_sec - track.last_seen_ts)
                new_center = detection.bbox.center
                dx = new_center[0] - track.last_center[0]
                dy = new_center[1] - track.last_center[1]
                track.speed_px_per_sec = ((dx * dx + dy * dy) ** 0.5) / elapsed
                track.detection = detection
                track.last_seen_ts = timestamp_sec
                track.last_center = new_center
                track.stable_frame_count += 1
                track.lost_count = 0

            matched_track_ids.add(track.track_id)

        for track_id in list(self._tracks):
            if self._tracks[track_id].lost_count > self.max_lost:
                del self._tracks[track_id]

        return [
            TrackSnapshot(
                track_id=track.track_id,
                class_name=track.detection.class_name,
                confidence=track.detection.confidence,
                bbox=track.detection.bbox,
                first_seen_ts=track.first_seen_ts,
                last_seen_ts=track.last_seen_ts,
                speed_px_per_sec=track.speed_px_per_sec,
                stable_frame_count=track.stable_frame_count,
            )
            for track in self._tracks.values()
            if track.lost_count == 0
        ]

    def _match_track(self, detection: DetectionRecord, used_ids: set[int]) -> _TrackState | None:
        best_track = None
        best_iou = 0.0
        for track in self._tracks.values():
            if track.track_id in used_ids:
                continue
            score = _bbox_iou(track.detection.bbox, detection.bbox)
            if score > best_iou:
                best_iou = score
                best_track = track
        if best_iou >= self.iou_threshold:
            return best_track
        return None


def _bbox_iou(a: BBox, b: BBox) -> float:
    inter_x1 = max(a.x1, b.x1)
    inter_y1 = max(a.y1, b.y1)
    inter_x2 = min(a.x2, b.x2)
    inter_y2 = min(a.y2, b.y2)
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    intersection = inter_w * inter_h
    union = a.area + b.area - intersection
    if union <= 0:
        return 0.0
    return intersection / union
