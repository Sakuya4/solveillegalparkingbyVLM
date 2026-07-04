import numpy as np

from utils.red_line_detector import bbox_hits_redline


def test_bbox_hits_redline_checks_contact_band_below_vehicle():
    red_mask = np.zeros((100, 100), dtype=np.uint8)
    red_mask[72:75, 20:80] = 255

    overlapped, ratio = bbox_hits_redline(
        bbox=(20, 20, 80, 70),
        red_mask=red_mask,
        band_px=10,
        min_ratio=0.003,
    )

    assert overlapped is True
    assert ratio > 0
