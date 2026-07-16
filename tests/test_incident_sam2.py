from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from illegal_parking.incident_candidate_roi import CandidateRegion
from illegal_parking.incident_sam2 import UltralyticsSam2Refiner


@dataclass
class _Masks:
    data: np.ndarray


@dataclass
class _Result:
    masks: _Masks | None


class _FakeSam2:
    def __init__(self, mask: np.ndarray | None):
        self.mask = mask
        self.calls: list[dict] = []

    def predict(self, source, bboxes, verbose):
        self.calls.append({"source": source, "bboxes": bboxes, "verbose": verbose})
        masks = None if self.mask is None else _Masks(self.mask)
        return [_Result(masks=masks)]


def test_sam2_refiner_uses_candidate_as_pixel_box_prompt() -> None:
    mask = np.zeros((1, 80, 100), dtype=np.uint8)
    mask[0, 30:60, 25:65] = 1
    model = _FakeSam2(mask)
    refiner = UltralyticsSam2Refiner(model=model)
    candidate = CandidateRegion(0.2, 0.25, 0.8, 0.9, score=0.8, source="track_motion", track_ids=(1, 2))

    refined = refiner.refine(np.zeros((80, 100, 3), dtype=np.uint8), candidate)

    assert model.calls[0]["bboxes"] == [20.0, 20.0, 80.0, 72.0]
    assert refined.bbox == pytest.approx((0.25, 0.375, 0.65, 0.75))
    assert refined.source == "track_motion_sam2"
    assert refined.track_ids == (1, 2)


@pytest.mark.parametrize("mask", [None, np.zeros((1, 80, 100), dtype=np.uint8)])
def test_sam2_refiner_keeps_candidate_when_no_mask_is_available(mask: np.ndarray | None) -> None:
    candidate = CandidateRegion(0.2, 0.25, 0.8, 0.9, score=0.8, source="track_motion")
    refiner = UltralyticsSam2Refiner(model=_FakeSam2(mask))

    refined = refiner.refine(np.zeros((80, 100, 3), dtype=np.uint8), candidate)

    assert refined is candidate
