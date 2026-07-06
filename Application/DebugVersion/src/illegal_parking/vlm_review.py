from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VlmReviewRequest:
    task: str
    prompt: str
    image_paths: list[str]
    evidence: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "prompt": self.prompt,
            "image_paths": self.image_paths,
            "evidence": self.evidence,
        }


def build_redline_parking_review_request(
    evidence: dict[str, Any],
    image_dir: str | Path,
) -> VlmReviewRequest:
    image_root = Path(image_dir)
    artifacts = evidence.get("artifacts", {})
    image_paths = [
        str(image_root / artifacts[name])
        for name in ("original", "bbox_overlay", "overlap_overlay")
        if name in artifacts
    ]
    prompt = _build_prompt(evidence)
    return VlmReviewRequest(
        task="redline_parking_review",
        prompt=prompt,
        image_paths=image_paths,
        evidence=evidence,
    )


def _build_prompt(evidence: dict[str, Any]) -> str:
    footprint_ratio = float(evidence.get("footprint_overlap_ratio", 0.0))
    footprint_pixels = int(evidence.get("footprint_overlap_pixels", 0))
    mask_source = str(evidence.get("mask_source", "unknown"))
    redline_margin = int(evidence.get("restricted_line_margin_px", 0))
    return (
        "You are reviewing a privacy-redacted traffic event. "
        "Do not infer or recover the license plate or hidden timestamp text. "
        "Use the original image, bbox overlay, and overlap overlay to decide whether "
        "the vehicle appears stopped on or beside a red no-parking curb line. "
        f"The mask source is {mask_source}. "
        f"The red-line contact band uses {redline_margin}px margin on both sides. "
        f"The vehicle bottom footprint overlaps the restricted red-line band by "
        f"{footprint_pixels} pixels, ratio {footprint_ratio:.4f}. "
        "Return a JSON object with keys: likely_violation, confidence, visual_reasons, "
        "missing_evidence, and human_review_needed."
    )
