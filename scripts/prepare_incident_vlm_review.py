from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Application" / "DebugVersion" / "src"
sys.path.insert(0, str(SRC))

from illegal_parking.incident_vlm_evidence import select_review_context
from illegal_parking.vlm_review import build_traffic_incident_review_request


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a privacy-safe temporal VLM review package from incident inference."
    )
    parser.add_argument("--inference-report", required=True)
    parser.add_argument("--annotated-video", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.inference_report).read_text(encoding="utf-8"))
    if not report.get("privacy"):
        raise ValueError("Inference report must document privacy redaction before evidence export.")

    selection = select_review_context(report)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_names = ("before.jpg", "trigger.jpg", "after.jpg")
    _extract_frames(
        Path(args.annotated_video),
        selection.frame_indices,
        tuple(output_dir / name for name in artifact_names),
    )

    evidence = {
        "event_type": "traffic_incident",
        "model_probability": selection.model_probability,
        "decision_threshold": float(report.get("threshold", 0.5)),
        "consecutive_positive_windows": selection.consecutive_positive_windows,
        "required_consecutive_windows": int(report.get("min_positive_windows", 1)),
        "confirmed_window_index": selection.window_index,
        "frame_indices": list(selection.frame_indices),
        "privacy_redacted": True,
        "privacy_method": str(report["privacy"]),
        "annotation_free_inference": bool(report.get("annotation_free_inference", False)),
        "artifacts": {
            "before": artifact_names[0],
            "trigger": artifact_names[1],
            "after": artifact_names[2],
        },
    }
    evidence_path = output_dir / "evidence.json"
    evidence_path.write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")

    request = build_traffic_incident_review_request(evidence, output_dir)
    request_path = output_dir / "vlm_review_request.json"
    request_path.write_text(
        json.dumps(request.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"evidence": str(evidence_path), "request": str(request_path)}, indent=2))
    return 0


def _extract_frames(
    video_path: Path,
    frame_indices: tuple[int, int, int],
    output_paths: tuple[Path, Path, Path],
) -> None:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open annotated video: {video_path}")
    try:
        for frame_index, output_path in zip(frame_indices, output_paths):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"Could not read frame {frame_index} from {video_path}")
            if not cv2.imwrite(str(output_path), frame):
                raise ValueError(f"Could not write review frame: {output_path}")
    finally:
        capture.release()


if __name__ == "__main__":
    raise SystemExit(main())
