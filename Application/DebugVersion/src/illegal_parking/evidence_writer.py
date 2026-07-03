from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .event_models import BBox, TrackSnapshot, ViolationCandidate


class EvidenceWriter:
    def __init__(self, output_dir: str | Path):
        self.output_dir = Path(output_dir)
        self.events_dir = self.output_dir / "events"
        self.events_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.output_dir / "events.jsonl"

    def write_event(
        self,
        camera_id: str,
        frame_index: int,
        frame_bgr: np.ndarray,
        track: TrackSnapshot,
        candidate: ViolationCandidate,
        redline_mask: np.ndarray | None = None,
    ) -> dict:
        import cv2

        event_dir = self.events_dir / candidate.event_id
        event_dir.mkdir(parents=True, exist_ok=True)

        original_rel = Path("events") / candidate.event_id / "original.jpg"
        annotated_rel = Path("events") / candidate.event_id / "annotated.jpg"
        crop_rel = Path("events") / candidate.event_id / "vehicle_crop.jpg"

        cv2.imwrite(str(self.output_dir / original_rel), frame_bgr)
        cv2.imwrite(str(self.output_dir / annotated_rel), _draw_bbox(frame_bgr, track.bbox))
        cv2.imwrite(str(self.output_dir / crop_rel), _crop(frame_bgr, track.bbox))

        artifacts = {
            "original_frame": str(original_rel).replace("\\", "/"),
            "annotated_frame": str(annotated_rel).replace("\\", "/"),
            "vehicle_crop": str(crop_rel).replace("\\", "/"),
        }
        if redline_mask is not None:
            mask_rel = Path("events") / candidate.event_id / "redline_mask.png"
            cv2.imwrite(str(self.output_dir / mask_rel), redline_mask)
            artifacts["redline_mask"] = str(mask_rel).replace("\\", "/")

        record = {
            "event_id": candidate.event_id,
            "camera_id": camera_id,
            "frame_index": frame_index,
            "track_id": candidate.track_id,
            "class_name": candidate.class_name,
            "bbox": list(candidate.bbox.as_xyxy()),
            "status": candidate.status,
            "is_candidate": candidate.is_candidate,
            "reasons": candidate.reasons,
            "rule_evidence": asdict(candidate.rule_evidence),
            "artifacts": artifacts,
        }

        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def _crop(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x1 = max(0, min(bbox.x1, w - 1))
    y1 = max(0, min(bbox.y1, h - 1))
    x2 = max(0, min(bbox.x2, w))
    y2 = max(0, min(bbox.y2, h))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((1, 1, 3), dtype=frame_bgr.dtype)
    return crop


def _draw_bbox(frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
    import cv2

    out = frame_bgr.copy()
    cv2.rectangle(out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 0, 255), 2)
    return out
