from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Publish camera transport and native-workflow measurements without redistributing imagery.")
    parser.add_argument("--capture-report", type=Path, required=True)
    parser.add_argument("--desktop-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/native_cctv_observations.json")
    args = parser.parse_args()
    native = json.loads(args.desktop_report.read_text(encoding="utf-8"))
    standalone = json.loads(args.capture_report.read_text(encoding="utf-8"))
    rows = []
    for name, capture, path in (("transport_capture", standalone, args.capture_report), ("native_capture_and_analysis", native["capture"], args.desktop_report)):
        keys = ("source_page", "transport", "started_at_utc", "requested_duration_sec", "observed_wall_duration_sec", "valid_observation_span_sec", "successful_frames", "decoded_frames", "failed_reads", "effective_fps", "unique_exact_frames", "exact_duplicate_rate", "new_frame_transitions_per_sec", "source_metadata", "output_video_sha256")
        row = {"workflow": name, "source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), **{key: capture[key] for key in keys if key in capture}}
        if name == "native_capture_and_analysis":
            row.update({"analysis_sec": native["elapsed_sec"], "review_events": len(native["events"]), "event_clock": native["event_clock"], "model_sha256": native["model_sha256"], "runtime": native["runtime"]})
        rows.append(row)
    result = {"measurement": "Persistent public-camera ingestion and native observation-to-analysis workflow", "annotation_status": "unassigned", "timing_contract": "Observation precedes batch analysis; event records retain original observation timestamps", "observations": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
