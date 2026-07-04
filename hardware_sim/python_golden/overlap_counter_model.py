from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OverlapPixel:
    in_bbox_band: bool
    is_red_pixel: bool


@dataclass(frozen=True)
class OverlapCounterResult:
    band_pixels: int
    red_pixels: int
    overlap_hit: bool


def count_overlap(pixels: list[OverlapPixel], min_red_pixels: int) -> OverlapCounterResult:
    band_pixels = 0
    red_pixels = 0
    for pixel in pixels:
        if not pixel.in_bbox_band:
            continue
        band_pixels += 1
        if pixel.is_red_pixel:
            red_pixels += 1

    return OverlapCounterResult(
        band_pixels=band_pixels,
        red_pixels=red_pixels,
        overlap_hit=red_pixels >= min_red_pixels,
    )
