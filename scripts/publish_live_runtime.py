from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.observation_review import load_review_observation, MAX_OBSERVATION_GAP_SEC


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_summary(report, protocol, decisions):
    if not report.get("streaming_inference") or report["source_mode"] != "page" or report["event_clock"] != "decoder observation clock":
        raise ValueError("A continuous public-camera observation is required")
    if report["input"] != protocol["camera_page"] or report["requested_duration_sec"] != protocol["requested_observation_sec"]:
        raise ValueError("Camera request differs from the predeclared protocol")
    for key in ("display_width", "feature_width_limit", "sequence_steps", "hop_steps", "min_positive_windows", "threshold"):
        if report[key] != protocol["inference"][key]:
            raise ValueError("Inference contract changed")
    if report["model_sha256"] != protocol["reference_model_sha256"] or report["detector_sha256"] != protocol["detector_sha256"]:
        raise ValueError("Reference weights changed")
    times = report["frame_times_sec"]
    if len(times) != report["sampled_frames"] or len(times) < 2 or any(not math.isfinite(t) or t < 0 for t in times) or any(b <= a for a, b in zip(times, times[1:])):
        raise ValueError("Sampled frame clocks must be aligned and increasing")
    if ([row["window_start"] for row in decisions] != report["window_starts"] or
            [row["probability"] for row in decisions] != report["window_probabilities"] or
            [row["review_active"] for row in decisions] != report["window_review_flags"]):
        raise ValueError("Decision log differs from its source report")
    gaps = [b - a for a, b in zip(times, times[1:])]
    if len(report["context_gaps"]) != sum(value > MAX_OBSERVATION_GAP_SEC for value in gaps):
        raise ValueError("Context-reset records disagree with sampled-frame gaps")
    events = [{key: event[key] for key in ("event_id", "trigger_time_sec", "probability")} for event in report["events"]]
    if any(event["trigger_time_sec"] not in times for event in events):
        raise ValueError("Event time is absent from sampled frames")
    return {"source": report["input"], "started_at_utc": report["started_at_utc"],
            "requested_observation_sec": report["requested_duration_sec"],
            "initialization_before_observation_sec": report["initialization_before_observation_sec"],
            "camera_model_warmup_sec": report["camera_model_warmup_sec"], "elapsed_sec": report["elapsed_sec"],
            "end_reason": report["end_reason"], "source_error": report["source_error"],
            "sampled_frames": len(times), "frame_times_sec": times, "source_span_sec": times[-1] - times[0],
            "effective_sampled_fps": (len(times) - 1) / (times[-1] - times[0]),
            "eligible_acquisition_sec": sum(value for value in gaps if value <= MAX_OBSERVATION_GAP_SEC),
            "excluded_gap_sec": sum(value for value in gaps if value > MAX_OBSERVATION_GAP_SEC),
            "max_observation_gap_sec": MAX_OBSERVATION_GAP_SEC, "context_gaps": report["context_gaps"],
            "acquisition": report["acquisition"], "processing_sec": report["processing_sec"],
            "arrival_to_decision_sec": report["arrival_to_decision_sec"], "runtime": report["runtime"],
            "window_count": len(decisions), "decisions": decisions, "candidate_alerts": events,
            "model_sha256": report["model_sha256"], "detector_sha256": report["detector_sha256"],
            "clean_video_sha256": report["clean_video_sha256"], "inference": protocol["inference"],
            "observation_label": "UNASSIGNED", "normal_exposure_added_hours": 0,
            "clock": protocol["clock"], "media_policy": protocol["media_policy"],
            "interpretation": "Public-endpoint runtime observation; candidate counts are not ground-truth events or a false-alarm rate"}


def main():
    parser = argparse.ArgumentParser(description="Publish hash-verified public-camera timing and numeric traces without imagery.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Do not overwrite a published observation")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    for name, expected in protocol["implementation_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Predeclared runtime implementation changed")
    report, _ = load_review_observation(args.report)
    path = args.report.parent / "decisions.jsonl"
    if sha(path) != report["decisions_sha256"]:
        raise ValueError("Decision log hash changed")
    decisions = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    result = runtime_summary(report, protocol, decisions)
    result.update({"protocol_sha256": sha(args.protocol), "source_report_sha256": sha(args.report),
                   "decisions_sha256": sha(path), "output_video_sha256": sha(Path(report["output_video"])),
                   "publisher_sha256": sha(Path(__file__))})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result[key] for key in ("sampled_frames", "window_count", "source_span_sec", "eligible_acquisition_sec", "processing_sec", "arrival_to_decision_sec", "end_reason")}, indent=2))


if __name__ == "__main__":
    main()
