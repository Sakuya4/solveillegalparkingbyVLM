import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def publish_review(report_path: Path, output: Path):
    report = json.loads(report_path.read_text(encoding="utf-8"))
    run = report["evidence_review_runs"][-1]
    source = Path(run["source_report"])
    if hashlib.sha256(source.read_bytes()).hexdigest() != run["source_report_sha256"]:
        raise ValueError("Saved camera source changed")
    original = json.loads(source.read_text(encoding="utf-8"))
    if not report.get("streaming_inference") or original["events"] != [{key: value for key, value in event.items() if key not in {"vlm_review", "vlm_raw_path", "review_disposition", "review_disposition_policy_version"}} for event in report["events"]]:
        raise ValueError("Review must preserve every original streaming event")
    events = []
    for index in run["event_indices"]:
        event = report["events"][index]
        review = event["vlm_review"]
        if review.get("generation_contract_version") != 2:
            raise ValueError("Publication requires explicitly forwarded generation parameters")
        events.append({"event_index": index, "trigger_time_sec": event["trigger_time_sec"],
                       "temporal_probability": event["probability"], "review_disposition": event["review_disposition"],
                       "policy_version": event["review_disposition_policy_version"], "model_output": review,
                       "raw_response_sha256": hashlib.sha256(Path(event["vlm_raw_path"]).read_bytes()).hexdigest(),
                       "annotation_status": "unassigned"})
    result = {"source_report_sha256": run["source_report_sha256"],
              "reviewed_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
              "source_model_sha256": report["model_sha256"], "detector_sha256": report["detector_sha256"],
              "review_duration_including_model_load_sec": run["elapsed_sec_including_model_load"],
              "detector_rerun": run["detector_rerun"], "event_clock": report["event_clock"],
              "events": events, "reviewed_normal_exposure_added_hours": 0,
              "scope": "Saved streaming-alert evidence reviewed by a VLM; model descriptions are not independently verified labels, and this session is not added to the normal-camera false-alarm denominator",
              "media_policy": "Numeric results and model responses only; public-camera imagery and video are retained locally"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description="Publish numeric saved-camera review outcomes without public-camera imagery.")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/streaming_alert_vlm_review.json")
    args = parser.parse_args()
    result = publish_review(args.report, args.output)
    print(f"Published {len(result['events'])} reviewed alerts; independent labels remain unassigned")


if __name__ == "__main__":
    main()
