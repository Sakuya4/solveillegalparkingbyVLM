from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .event_models import BBox


@dataclass(frozen=True)
class SceneEvidence:
    redline_overlap_ratio: float
    in_no_parking_roi: bool = False


class SceneAnalyzer(Protocol):
    def analyze(self, frame_bgr: np.ndarray, bbox: BBox) -> SceneEvidence:
        ...

    def get_last_mask(self) -> np.ndarray | None:
        ...


class StaticSceneAnalyzer:
    def __init__(self, redline_overlap_ratio: float = 0.0, in_no_parking_roi: bool = False):
        self._evidence = SceneEvidence(redline_overlap_ratio, in_no_parking_roi)

    def analyze(self, frame_bgr: np.ndarray, bbox: BBox) -> SceneEvidence:
        return self._evidence

    def get_last_mask(self) -> np.ndarray | None:
        return None


class RedlineSceneAnalyzer:
    def __init__(self, band_px: int = 30, min_ratio: float = 0.003):
        self.band_px = band_px
        self.min_ratio = min_ratio
        self._last_mask: np.ndarray | None = None

    def analyze(self, frame_bgr: np.ndarray, bbox: BBox) -> SceneEvidence:
        from utils.red_line_detector import bbox_hits_redline, detect_red_lines

        self._last_mask = detect_red_lines(frame_bgr)
        _overlap, ratio = bbox_hits_redline(
            bbox.as_xyxy(),
            self._last_mask,
            band_px=self.band_px,
            min_ratio=self.min_ratio,
        )
        return SceneEvidence(redline_overlap_ratio=ratio, in_no_parking_roi=False)

    def get_last_mask(self) -> np.ndarray | None:
        return self._last_mask
