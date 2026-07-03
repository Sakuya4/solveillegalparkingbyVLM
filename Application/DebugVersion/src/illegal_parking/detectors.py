from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .event_models import BBox


@dataclass(frozen=True)
class DetectionRecord:
    bbox: BBox
    confidence: float
    class_id: int
    class_name: str


class Detector(Protocol):
    def detect(self, frame_bgr: np.ndarray) -> list[DetectionRecord]:
        ...


class StaticDetector:
    def __init__(self, detections: list[DetectionRecord]):
        self._detections = detections

    def detect(self, frame_bgr: np.ndarray) -> list[DetectionRecord]:
        return list(self._detections)


class UltralyticsYoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", imgsz: int = 640, conf: float = 0.25):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.imgsz = imgsz
        self.conf = conf

    def detect(self, frame_bgr: np.ndarray) -> list[DetectionRecord]:
        records: list[DetectionRecord] = []
        results = self.model.predict(source=frame_bgr, imgsz=self.imgsz, conf=self.conf, verbose=False)
        for result in results:
            names = getattr(result, "names", {}) or {}
            for box in getattr(result, "boxes", []):
                xyxy = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                records.append(
                    DetectionRecord(
                        bbox=BBox(int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])),
                        confidence=confidence,
                        class_id=class_id,
                        class_name=str(names.get(class_id, class_id)).lower(),
                    )
                )
        return records
