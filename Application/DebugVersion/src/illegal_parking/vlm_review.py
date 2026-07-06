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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "VlmReviewRequest":
        return cls(
            task=str(payload["task"]),
            prompt=str(payload["prompt"]),
            image_paths=[str(path) for path in payload.get("image_paths", [])],
            evidence=dict(payload.get("evidence", {})),
        )


@dataclass(frozen=True)
class VlmReviewResult:
    likely_violation: bool
    confidence: float
    visual_reasons: list[str]
    missing_evidence: list[str]
    human_review_needed: bool
    provider: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "likely_violation": self.likely_violation,
            "confidence": self.confidence,
            "visual_reasons": self.visual_reasons,
            "missing_evidence": self.missing_evidence,
            "human_review_needed": self.human_review_needed,
            "provider": self.provider,
        }


@dataclass(frozen=True)
class OfflineReviewThresholds:
    confirm_footprint_overlap_ratio: float = 0.1
    review_footprint_overlap_ratio: float = 0.03
    min_overlap_pixels: int = 1


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


def review_redline_parking_offline(
    request: VlmReviewRequest,
    thresholds: OfflineReviewThresholds = OfflineReviewThresholds(),
) -> VlmReviewResult:
    evidence = request.evidence
    footprint_ratio = float(evidence.get("footprint_overlap_ratio", 0.0))
    footprint_pixels = int(evidence.get("footprint_overlap_pixels", 0))
    mask_source = str(evidence.get("mask_source", "unknown"))
    missing_evidence = _find_missing_evidence(request)
    visual_reasons = _build_visual_reasons(evidence)

    has_contact = footprint_pixels >= thresholds.min_overlap_pixels
    likely_violation = has_contact and footprint_ratio >= thresholds.confirm_footprint_overlap_ratio
    near_contact = has_contact and footprint_ratio >= thresholds.review_footprint_overlap_ratio

    if likely_violation:
        confidence = min(0.95, 0.68 + footprint_ratio * 1.1)
    elif near_contact:
        confidence = min(0.69, 0.45 + footprint_ratio * 1.4)
    else:
        confidence = 0.25

    if mask_source == "bbox":
        missing_evidence.append("SAM or instance mask evidence; bbox-only review is weaker.")
        confidence = min(confidence, 0.62)
    if not has_contact:
        missing_evidence.append("Vehicle bottom footprint does not overlap the restricted red-line band.")

    human_review_needed = bool(missing_evidence) or confidence < 0.7 or not likely_violation
    return VlmReviewResult(
        likely_violation=likely_violation,
        confidence=round(confidence, 4),
        visual_reasons=visual_reasons,
        missing_evidence=missing_evidence,
        human_review_needed=human_review_needed,
        provider="offline_evidence_reviewer",
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


def _build_visual_reasons(evidence: dict[str, Any]) -> list[str]:
    footprint_ratio = float(evidence.get("footprint_overlap_ratio", 0.0))
    footprint_pixels = int(evidence.get("footprint_overlap_pixels", 0))
    redline_margin = int(evidence.get("restricted_line_margin_px", 0))
    reasons = [
        f"Vehicle bottom footprint overlaps restricted band by {footprint_pixels} pixels.",
        f"Footprint overlap ratio is {footprint_ratio:.4f}.",
    ]
    if "restricted_line" in evidence:
        reasons.append(f"Restricted red-line band uses {redline_margin}px margin on both sides.")
    return reasons


def _find_missing_evidence(request: VlmReviewRequest) -> list[str]:
    missing: list[str] = []
    evidence = request.evidence
    artifact_names = set(evidence.get("artifacts", {}).keys())
    if "original" not in artifact_names and "original_frame" not in artifact_names:
        missing.append("Original privacy-redacted image artifact.")
    if "overlap_overlay" not in artifact_names:
        missing.append("Overlap overlay artifact.")
    if "restricted_line" not in evidence and "restricted_rect" not in evidence:
        missing.append("Restricted zone geometry.")
    if not request.image_paths:
        missing.append("Review image paths.")
    return missing
