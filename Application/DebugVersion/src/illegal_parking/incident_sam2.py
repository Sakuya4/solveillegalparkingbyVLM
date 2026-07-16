from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .incident_candidate_roi import CandidateRegion


class UltralyticsSam2Refiner:
    source_name = "sam2"

    def __init__(
        self,
        checkpoint: str = "sam2.1_t.pt",
        device: str | None = None,
        min_mask_area_ratio: float = 0.05,
        model: Any | None = None,
    ) -> None:
        if not 0.0 <= min_mask_area_ratio <= 1.0:
            raise ValueError("min_mask_area_ratio must be between zero and one")
        if model is None:
            try:
                from ultralytics import SAM
            except ImportError as exc:
                raise RuntimeError(
                    "Ultralytics with SAM2 support is required when a SAM2 checkpoint is enabled."
                ) from exc
            model = SAM(checkpoint)
        self.model = model
        self.device = device
        self.min_mask_area_ratio = min_mask_area_ratio

    def refine(self, frame_bgr: np.ndarray, candidate: CandidateRegion) -> CandidateRegion:
        if frame_bgr.ndim != 3 or frame_bgr.shape[2] != 3:
            raise ValueError("SAM2 refinement expects a BGR frame with three channels")
        height, width = frame_bgr.shape[:2]
        prompt = [
            candidate.x1 * width,
            candidate.y1 * height,
            candidate.x2 * width,
            candidate.y2 * height,
        ]
        arguments: dict[str, Any] = {
            "source": frame_bgr,
            "bboxes": prompt,
            "verbose": False,
        }
        if self.device is not None:
            arguments["device"] = self.device
        results = self.model.predict(**arguments)
        if not results or getattr(results[0], "masks", None) is None:
            return candidate
        mask_data = _as_numpy(results[0].masks.data)
        if mask_data.ndim == 2:
            mask_data = mask_data[None, ...]
        if mask_data.ndim != 3 or len(mask_data) == 0:
            return candidate

        prompt_slices = _prompt_slices(candidate, width, height)
        masks = [_resize_mask(mask, width, height) for mask in mask_data]
        selected = max(masks, key=lambda mask: int(mask[prompt_slices].sum()))
        clipped = np.zeros((height, width), dtype=np.uint8)
        clipped[prompt_slices] = (selected[prompt_slices] > 0).astype(np.uint8)
        ys, xs = np.nonzero(clipped)
        if not len(xs):
            return candidate

        refined_bbox = (
            float(xs.min() / width),
            float(ys.min() / height),
            float((xs.max() + 1) / width),
            float((ys.max() + 1) / height),
        )
        refined_area = (refined_bbox[2] - refined_bbox[0]) * (refined_bbox[3] - refined_bbox[1])
        if refined_area < candidate.area * self.min_mask_area_ratio:
            return candidate
        return CandidateRegion(
            *refined_bbox,
            score=candidate.score,
            source=f"{candidate.source}_sam2",
            track_ids=candidate.track_ids,
        )


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape != (height, width):
        binary = cv2.resize(binary, (width, height), interpolation=cv2.INTER_NEAREST)
    return binary


def _prompt_slices(
    candidate: CandidateRegion,
    width: int,
    height: int,
) -> tuple[slice, slice]:
    left = max(0, min(width - 1, int(np.floor(candidate.x1 * width))))
    top = max(0, min(height - 1, int(np.floor(candidate.y1 * height))))
    right = max(left + 1, min(width, int(np.ceil(candidate.x2 * width))))
    bottom = max(top + 1, min(height, int(np.ceil(candidate.y2 * height))))
    return slice(top, bottom), slice(left, right)
