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
    parser.add_argument("--stream-report", type=Path, action="append", default=[])
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
    streams = []
    for path in args.stream_report:
        stream = json.loads(path.read_text(encoding="utf-8"))
        if stream.get("streaming_inference") is not True or stream["source_mode"] == "file":
            raise ValueError("Streaming camera reports must contain online camera inference")
        times = stream["frame_times_sec"]
        if len(times) != stream["sampled_frames"] or len(times) < 2 or any(a >= b for a, b in zip(times, times[1:])):
            raise ValueError("Streaming timestamps must align with increasing sampled frames")
        streams.append({"source_report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                        "source_page": stream["input"], "started_at_utc": stream["started_at_utc"],
                        "sampled_frames": len(times), "valid_observation_span_sec": times[-1] - times[0],
                        "effective_fps": (len(times) - 1) / (times[-1] - times[0]),
                        "review_events": len(stream["events"]),
                        **{key: stream[key] for key in ("acquisition", "arrival_to_decision_sec", "processing_sec", "event_clock", "model_sha256", "detector_sha256", "runtime", "buffers", "decisions_sha256", "feature_width_limit", "display_width")}})
    result = {"measurement": "Persistent public-camera ingestion, native observation-to-analysis, and continuous causal inference", "annotation_status": "unassigned", "timing_contract": "Capture sessions precede batch analysis; streaming sessions emit window-end decisions while acquisition remains active; original observation timestamps are retained", "observations": rows, "streaming_sessions": streams}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
