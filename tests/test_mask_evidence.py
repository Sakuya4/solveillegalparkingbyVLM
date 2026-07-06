import numpy as np

from illegal_parking.event_models import BBox
from illegal_parking.mask_evidence import (
    BBoxVehicleMaskProvider,
    MaskEvidenceAnalyzer,
    StaticVehicleMaskProvider,
)


def test_bbox_mask_provider_creates_clipped_vehicle_mask():
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    provider = BBoxVehicleMaskProvider()

    mask = provider.predict_mask(frame, BBox(-2, 2, 5, 9))

    assert mask.shape == (8, 10)
    assert mask.dtype == np.uint8
    assert int(mask.sum()) == 30


def test_mask_evidence_uses_vehicle_mask_overlap_with_restricted_mask():
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    vehicle_mask = np.zeros((8, 10), dtype=np.uint8)
    vehicle_mask[2:6, 2:7] = 1
    redline_mask = np.zeros((8, 10), dtype=np.uint8)
    redline_mask[4:8, 4:9] = 255
    analyzer = MaskEvidenceAnalyzer(StaticVehicleMaskProvider(vehicle_mask))

    evidence = analyzer.analyze(
        frame_bgr=frame,
        bbox=BBox(2, 2, 7, 6),
        restricted_mask=redline_mask,
        in_no_parking_roi=False,
    )

    assert evidence.vehicle_mask_area == 20
    assert evidence.restricted_overlap_pixels == 6
    assert evidence.restricted_overlap_ratio == 0.3
    assert evidence.restricted_coverage_ratio == 0.3
    assert evidence.source == "static_mask"


def test_mask_evidence_can_mark_roi_without_redline_pixels():
    frame = np.zeros((8, 10, 3), dtype=np.uint8)
    analyzer = MaskEvidenceAnalyzer(BBoxVehicleMaskProvider())

    evidence = analyzer.analyze(
        frame_bgr=frame,
        bbox=BBox(1, 1, 5, 5),
        restricted_mask=None,
        in_no_parking_roi=True,
    )

    assert evidence.vehicle_mask_area == 16
    assert evidence.restricted_overlap_pixels == 16
    assert evidence.restricted_overlap_ratio == 1.0
    assert evidence.restricted_coverage_ratio == 1.0
    assert evidence.source == "bbox"
