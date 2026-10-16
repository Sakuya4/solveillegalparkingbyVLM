import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.observation_review import load_review_observation, MAX_OBSERVATION_GAP_SEC


def main():
    parser = argparse.ArgumentParser(description="Publish numeric review-ready observation inventory without assigning normal labels.")
    parser.add_argument("--report", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/review_ready_observations.json")
    args = parser.parse_args()
    observations = []
    for path in args.report:
        report, _ = load_review_observation(path)
        decisions_path = path.parent / "decisions.jsonl"
        if hashlib.sha256(decisions_path.read_bytes()).hexdigest() != report["decisions_sha256"]:
            raise ValueError("Window decision log no longer matches its observation report")
        decisions = [json.loads(line) for line in decisions_path.read_text(encoding="utf-8").splitlines() if line]
        times = report["frame_times_sec"]
        differences = [b - a for a, b in zip(times, times[1:])]
        observations.append({"source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                             "clean_video_sha256": report["clean_video_sha256"], "model_sha256": report["model_sha256"],
                             "detector_sha256": report["detector_sha256"], "started_at_utc": report["started_at_utc"],
                             "source_mode": report["source_mode"], "source_clock": report["event_clock"],
                             "source": report["input"] if report["source_mode"] != "file" else "reference file replay",
                             "sampled_frames": report["sampled_frames"], "source_span_sec": times[-1] - times[0],
                             "requested_duration_sec": report.get("requested_duration_sec"), "end_reason": report["end_reason"], "source_error": report["source_error"],
                             "context_gaps": report.get("context_gaps", []),
                             "first_decision_after_observation_start_wall_sec": decisions[0]["available_wall_sec"] if decisions else None,
                             "initialization_before_observation_sec": report.get("initialization_before_observation_sec"),
                             "observable_sec_at_gap_limit": sum(value for value in differences if value <= MAX_OBSERVATION_GAP_SEC),
                             "excluded_gap_sec": sum(value for value in differences if value > MAX_OBSERVATION_GAP_SEC),
                             "model_alert_episodes": len(report["events"]), "acquisition": report.get("acquisition"),
                             "arrival_to_decision_sec": report.get("arrival_to_decision_sec"),
                             "review_status": "unassigned", "normal_exposure_added_hours": 0})
    result = {"observations": observations, "max_observation_gap_sec": MAX_OBSERVATION_GAP_SEC,
              "scope": "Acquired review-ready observations only; no normal/incident ground-truth label or normal-exposure claim is inferred from acquisition or model output",
              "media_policy": "Public-camera videos and images remain local; only numerical provenance is published"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
