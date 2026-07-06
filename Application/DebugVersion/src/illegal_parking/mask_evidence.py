from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .event_models import BBox


@dataclass(frozen=True)
class MaskEvidence:
    source: str
    vehicle_mask_area: int
    restricted_overlap_pixels: int
    restricted_overlap_ratio: float
    restricted_coverage_ratio: float


class VehicleMaskProvider(Protocol):
    source_name: str

    def predict_mask(self, frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
        ...


class BBoxVehicleMaskProvider:
    source_name = "bbox"

    def predict_mask(self, frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1 = max(0, min(bbox.x1, w))
        y1 = max(0, min(bbox.y1, h))
        x2 = max(0, min(bbox.x2, w))
        y2 = max(0, min(bbox.y2, h))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 1
        return mask


class StaticVehicleMaskProvider:
    source_name = "static_mask"

    def __init__(self, mask: np.ndarray):
        self.mask = _as_binary_mask(mask)

    def predict_mask(self, frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        if self.mask.shape != (h, w):
            raise ValueError("Static mask shape must match frame height and width.")
        return self.mask.copy()


class SamPromptMaskProvider:
    source_name = "sam_prompt"

    def __init__(self, checkpoint: str, model_type: str = "vit_b", device: str = "cuda"):
        try:
            from segment_anything import SamPredictor, sam_model_registry
        except ImportError as exc:
            raise RuntimeError(
                "segment-anything is required for SamPromptMaskProvider. "
                "Install it from the official Meta repository and provide a checkpoint."
            ) from exc

        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        sam.to(device=device)
        self.predictor = SamPredictor(sam)

    def predict_mask(self, frame_bgr: np.ndarray, bbox: BBox) -> np.ndarray:
        frame_rgb = frame_bgr[:, :, ::-1]
        self.predictor.set_image(frame_rgb)
        masks, scores, _ = self.predictor.predict(
            box=np.array(bbox.as_xyxy(), dtype=np.float32),
            multimask_output=True,
        )
        best_index = int(np.argmax(scores))
        return masks[best_index].astype(np.uint8)


class MaskEvidenceAnalyzer:
    def __init__(self, vehicle_mask_provider: VehicleMaskProvider | None = None):
        self.vehicle_mask_provider = vehicle_mask_provider or BBoxVehicleMaskProvider()

    def analyze(
        self,
        frame_bgr: np.ndarray,
        bbox: BBox,
        restricted_mask: np.ndarray | None,
        in_no_parking_roi: bool,
    ) -> MaskEvidence:
        vehicle_mask = _as_binary_mask(self.vehicle_mask_provider.predict_mask(frame_bgr, bbox))
        vehicle_area = int(vehicle_mask.sum())
        if vehicle_area == 0:
            return MaskEvidence(
                source=self.vehicle_mask_provider.source_name,
                vehicle_mask_area=0,
                restricted_overlap_pixels=0,
                restricted_overlap_ratio=0.0,
                restricted_coverage_ratio=0.0,
            )

        if in_no_parking_roi:
            restricted = vehicle_mask
        elif restricted_mask is None:
            restricted = np.zeros_like(vehicle_mask, dtype=np.uint8)
        else:
            restricted = _as_binary_mask(restricted_mask)
            if restricted.shape != vehicle_mask.shape:
                raise ValueError("Restricted mask shape must match vehicle mask shape.")

        overlap_pixels = int(np.logical_and(vehicle_mask, restricted).sum())
        restricted_area = int(restricted.sum())
        return MaskEvidence(
            source=self.vehicle_mask_provider.source_name,
            vehicle_mask_area=vehicle_area,
            restricted_overlap_pixels=overlap_pixels,
            restricted_overlap_ratio=overlap_pixels / vehicle_area,
            restricted_coverage_ratio=overlap_pixels / restricted_area if restricted_area else 0.0,
        )


def _as_binary_mask(mask: np.ndarray) -> np.ndarray:
    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")
    return (mask > 0).astype(np.uint8)
