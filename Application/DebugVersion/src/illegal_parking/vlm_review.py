from __future__ import annotations

import json
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

    @classmethod
    def from_dict(cls, payload: dict[str, Any], provider: str | None = None) -> "VlmReviewResult":
        return cls(
            likely_violation=_as_bool(payload["likely_violation"]),
            confidence=float(payload["confidence"]),
            visual_reasons=_as_string_list(payload.get("visual_reasons", [])),
            missing_evidence=_as_string_list(payload.get("missing_evidence", [])),
            human_review_needed=_as_bool(payload["human_review_needed"]),
            provider=str(provider or payload.get("provider", "unknown_vlm")),
        )


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


def build_traffic_incident_review_request(
    evidence: dict[str, Any],
    image_dir: str | Path,
    include_model_context: bool = True,
) -> VlmReviewRequest:
    image_root = Path(image_dir)
    artifacts = evidence.get("artifacts", {})
    image_paths = [
        str(image_root / artifacts[name])
        for name in ("before", "trigger", "after")
        if name in artifacts
    ]
    request_evidence = dict(evidence)
    if not include_model_context:
        for key in (
            "model_probability",
            "decision_threshold",
            "consecutive_positive_windows",
            "required_consecutive_windows",
        ):
            request_evidence.pop(key, None)
    return VlmReviewRequest(
        task="traffic_incident_review",
        prompt=_build_incident_prompt(evidence, include_model_context),
        image_paths=image_paths,
        evidence=request_evidence,
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


def review_traffic_incident_offline(request: VlmReviewRequest) -> VlmReviewResult:
    evidence = request.evidence
    probability = float(evidence.get("model_probability", 0.0))
    threshold = float(evidence.get("decision_threshold", 1.0))
    positive_windows = int(evidence.get("consecutive_positive_windows", 0))
    required_windows = int(evidence.get("required_consecutive_windows", 1))
    missing_evidence = _find_incident_missing_evidence(request)

    score_confirmed = probability >= threshold
    persistence_confirmed = positive_windows >= required_windows
    likely_incident = score_confirmed and persistence_confirmed

    if not score_confirmed:
        missing_evidence.append(
            f"Model probability {probability:.4f} is below decision threshold {threshold:.4f}."
        )
    if not persistence_confirmed:
        missing_evidence.append(
            f"Temporal persistence is {positive_windows} windows; {required_windows} are required."
        )

    if likely_incident:
        confidence = min(0.95, max(0.5, probability))
    else:
        confidence = min(0.49, max(0.1, probability * 0.5))

    visual_reasons = [
        f"Incident candidate model probability is {probability:.4f} at threshold {threshold:.4f}.",
        f"Candidate persists for {positive_windows} consecutive windows; {required_windows} are required.",
    ]
    return VlmReviewResult(
        likely_violation=likely_incident,
        confidence=round(confidence, 4),
        visual_reasons=visual_reasons,
        missing_evidence=missing_evidence,
        human_review_needed=True,
        provider="offline_incident_evidence_reviewer",
    )


def review_request_offline(request: VlmReviewRequest) -> VlmReviewResult:
    if request.task == "redline_parking_review":
        return review_redline_parking_offline(request)
    if request.task == "traffic_incident_review":
        return review_traffic_incident_offline(request)
    raise ValueError(f"Unsupported offline review task: {request.task}")


def parse_vlm_review_result_text(text: str, provider: str) -> VlmReviewResult:
    payload = json.loads(_extract_json_object(text))
    return VlmReviewResult.from_dict(payload, provider=provider)


def build_unparsed_vlm_review_result(raw_text: str, provider: str, error: str) -> VlmReviewResult:
    snippet = " ".join(raw_text.split())[:240]
    return VlmReviewResult(
        likely_violation=False,
        confidence=0.0,
        visual_reasons=[f"Unparsed VLM response: {snippet}"] if snippet else [],
        missing_evidence=[f"VLM response could not be normalized: {error}"],
        human_review_needed=True,
        provider=provider,
    )


def compare_vlm_review_results(
    results: list[VlmReviewResult],
    baseline_provider: str = "offline_evidence_reviewer",
) -> list[dict[str, Any]]:
    if not results:
        return []
    baseline = next((result for result in results if result.provider == baseline_provider), results[0])
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "provider": result.provider,
                "likely_violation": result.likely_violation,
                "confidence": result.confidence,
                "human_review_needed": result.human_review_needed,
                "agrees_with_baseline": result.likely_violation == baseline.likely_violation,
                "confidence_delta_from_baseline": round(result.confidence - baseline.confidence, 4),
                "missing_evidence_count": len(result.missing_evidence),
            }
        )
    return rows


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


def _build_incident_prompt(
    evidence: dict[str, Any], include_model_context: bool = True
) -> str:
    probability = float(evidence.get("model_probability", 0.0))
    threshold = float(evidence.get("decision_threshold", 1.0))
    positive_windows = int(evidence.get("consecutive_positive_windows", 0))
    required_windows = int(evidence.get("required_consecutive_windows", 1))
    visual_task = (
        "You are reviewing an ordered, privacy-redacted traffic event: before, trigger, and after. "
        "Do not infer or recover license plates, hidden timestamps, faces, or location text. "
        "Assess only visible temporal evidence of a collision, abrupt road conflict, stopped hazard, "
        "or another abnormal road incident that requires operator attention. "
    )
    model_context = (
        f"The candidate model probability is {probability:.4f}, threshold {threshold:.4f}, and the "
        f"candidate persists for {positive_windows} windows out of {required_windows} required. "
        "These model values select evidence and are not ground truth. "
        if include_model_context
        else "This is a blinded visual review; model scores and thresholds are intentionally hidden. "
    )
    return (
        visual_task
        + model_context
        + "For this task, likely_violation means "
        "a likely traffic incident that requires operator review. Return a JSON object with keys: "
        "likely_violation, confidence, visual_reasons, missing_evidence, and human_review_needed."
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


def _find_incident_missing_evidence(request: VlmReviewRequest) -> list[str]:
    missing: list[str] = []
    artifacts = request.evidence.get("artifacts", {})
    for artifact_name in ("before", "trigger", "after"):
        if artifact_name not in artifacts:
            missing.append(f"{artifact_name.title()} event image artifact.")
    if not bool(request.evidence.get("privacy_redacted", False)):
        missing.append("Verified privacy redaction for review images.")
    if len(request.image_paths) < 3:
        missing.append("Ordered before, trigger, and after image paths.")
    return missing


def _extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("VLM response does not contain a JSON object.")
    return cleaned[start : end + 1]


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "yes", "1"}:
            return True
        if normalized in {"false", "no", "0"}:
            return False
        raise ValueError(f"Unsupported boolean string: {value}")
    return bool(value)


def _as_string_list(value: Any) -> list[str]:
    if value is None or value is False:
        return []
    if isinstance(value, str):
        if value.strip().lower() in {"", "false", "none", "null", "no"}:
            return []
        return [value]
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(value)]
