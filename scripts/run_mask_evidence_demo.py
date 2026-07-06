from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.event_models import BBox
from illegal_parking.mask_evidence import (
    BBoxVehicleMaskProvider,
    MaskEvidenceAnalyzer,
    SamPromptMaskProvider,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a mask-evidence demo for violation-event review.")
    parser.add_argument("--image", help="Optional input image. If omitted, a synthetic traffic frame is generated.")
    parser.add_argument("--bbox", help="Vehicle bbox as x1,y1,x2,y2. Defaults to the synthetic vehicle bbox.")
    parser.add_argument("--restricted-rect", help="Restricted zone rect as x1,y1,x2,y2. Defaults to synthetic red-line band.")
    parser.add_argument("--sam-checkpoint", help="Optional SAM checkpoint. If omitted, bbox mask fallback is used.")
    parser.add_argument("--sam-model-type", default="vit_b")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", default="outputs/mask_evidence_demo")
    args = parser.parse_args()

    import cv2

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise ValueError(f"Could not read image: {args.image}")
        bbox = _parse_bbox(args.bbox) if args.bbox else _default_bbox_for(frame)
        restricted_rect = _parse_bbox(args.restricted_rect) if args.restricted_rect else _default_restricted_rect_for(frame)
    else:
        frame, bbox, restricted_rect = _synthetic_frame()

    restricted_mask = _rect_mask(frame.shape[:2], restricted_rect)
    provider = (
        SamPromptMaskProvider(
            checkpoint=args.sam_checkpoint,
            model_type=args.sam_model_type,
            device=args.device,
        )
        if args.sam_checkpoint
        else BBoxVehicleMaskProvider()
    )
    analyzer = MaskEvidenceAnalyzer(provider)
    evidence = analyzer.analyze(
        frame_bgr=frame,
        bbox=bbox,
        restricted_mask=restricted_mask,
        in_no_parking_roi=False,
    )
    vehicle_mask = provider.predict_mask(frame, bbox)

    cv2.imwrite(str(output_dir / "original.jpg"), frame)
    cv2.imwrite(str(output_dir / "bbox_overlay.jpg"), _draw_bbox_and_zone(frame, bbox, restricted_rect))
    cv2.imwrite(str(output_dir / "vehicle_mask.png"), _mask_to_image(vehicle_mask))
    cv2.imwrite(str(output_dir / "restricted_mask.png"), _mask_to_image(restricted_mask))
    cv2.imwrite(str(output_dir / "overlap_overlay.jpg"), _draw_overlap(frame, vehicle_mask, restricted_mask))

    evidence_record = asdict(evidence)
    evidence_record["mask_source"] = evidence_record.pop("source")
    record = {
        **evidence_record,
        "bbox": list(bbox.as_xyxy()),
        "restricted_rect": list(restricted_rect.as_xyxy()),
        "artifacts": {
            "original": "original.jpg",
            "bbox_overlay": "bbox_overlay.jpg",
            "vehicle_mask": "vehicle_mask.png",
            "restricted_mask": "restricted_mask.png",
            "overlap_overlay": "overlap_overlay.jpg",
        },
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    return 0


def _synthetic_frame() -> tuple[np.ndarray, BBox, BBox]:
    import cv2

    frame = np.zeros((480, 720, 3), dtype=np.uint8)
    frame[:, :] = (48, 48, 48)
    cv2.rectangle(frame, (0, 310), (719, 360), (36, 36, 36), -1)
    cv2.line(frame, (0, 340), (719, 340), (0, 0, 255), 8)
    cv2.line(frame, (0, 250), (719, 250), (220, 220, 220), 2)
    cv2.line(frame, (0, 420), (719, 420), (220, 220, 220), 2)

    bbox = BBox(270, 250, 470, 380)
    cv2.rectangle(frame, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (55, 130, 220), -1)
    cv2.rectangle(frame, (300, 270), (440, 320), (35, 80, 140), -1)
    cv2.circle(frame, (315, 380), 24, (20, 20, 20), -1)
    cv2.circle(frame, (425, 380), 24, (20, 20, 20), -1)
    restricted_rect = BBox(0, 332, 720, 349)
    return frame, bbox, restricted_rect


def _parse_bbox(value: str) -> BBox:
    parts = [int(part.strip()) for part in value.split(",")]
    if len(parts) != 4:
        raise ValueError("BBox values must use x1,y1,x2,y2 format.")
    return BBox(*parts)


def _default_bbox_for(frame: np.ndarray) -> BBox:
    h, w = frame.shape[:2]
    return BBox(w // 3, h // 3, (w * 2) // 3, (h * 2) // 3)


def _default_restricted_rect_for(frame: np.ndarray) -> BBox:
    h, w = frame.shape[:2]
    return BBox(0, int(h * 0.68), w, int(h * 0.72))


def _rect_mask(shape_hw: tuple[int, int], rect: BBox) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    x1 = max(0, min(rect.x1, w))
    y1 = max(0, min(rect.y1, h))
    x2 = max(0, min(rect.x2, w))
    y2 = max(0, min(rect.y2, h))
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = 1
    return mask


def _mask_to_image(mask: np.ndarray) -> np.ndarray:
    return (mask > 0).astype(np.uint8) * 255


def _draw_bbox_and_zone(frame: np.ndarray, bbox: BBox, restricted_rect: BBox) -> np.ndarray:
    import cv2

    out = frame.copy()
    cv2.rectangle(out, (restricted_rect.x1, restricted_rect.y1), (restricted_rect.x2, restricted_rect.y2), (0, 0, 255), 3)
    cv2.rectangle(out, (bbox.x1, bbox.y1), (bbox.x2, bbox.y2), (0, 255, 255), 3)
    return out


def _draw_overlap(frame: np.ndarray, vehicle_mask: np.ndarray, restricted_mask: np.ndarray) -> np.ndarray:
    out = frame.copy()
    vehicle = vehicle_mask > 0
    restricted = restricted_mask > 0
    overlap = np.logical_and(vehicle, restricted)
    out[vehicle] = (0.55 * out[vehicle] + np.array([0, 180, 255]) * 0.45).astype(np.uint8)
    out[restricted] = (0.55 * out[restricted] + np.array([0, 0, 255]) * 0.45).astype(np.uint8)
    out[overlap] = (0, 255, 0)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
