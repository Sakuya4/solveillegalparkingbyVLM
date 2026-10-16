from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description="Verify and publish real-detector CPU batch/streaming parity.")
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--stream-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/streaming_reference_parity.json")
    args = parser.parse_args()
    batch, stream = [json.loads(path.read_text(encoding="utf-8")) for path in (args.batch_report, args.stream_report)]
    keys = ("model_sha256", "input_sha256", "detector_sha256", "sampled_frames", "window_starts", "window_review_flags", "min_positive_windows", "threshold")
    if stream.get("streaming_inference") is not True or stream["source_mode"] != "file" or any(batch[key] != stream[key] for key in keys):
        raise ValueError("Batch and streaming provenance, frame selection, and decisions must agree")
    a, b = np.asarray(batch["window_probabilities"]), np.asarray(stream["window_probabilities"])
    np.testing.assert_allclose(a, b, atol=1e-5, rtol=0)
    if [event["input_frame_index"] for event in batch["events"]] != [event["input_frame_index"] for event in stream["events"]]:
        raise ValueError("Batch and streaming event frame identities differ")
    result = {"input_filename": Path(stream["input"]).name,
              **{key: stream[key] for key in ("model_sha256", "input_sha256", "detector_sha256", "sampled_frames", "min_positive_windows", "runtime", "buffers")},
              "batch_report_sha256": hashlib.sha256(args.batch_report.read_bytes()).hexdigest(),
              "stream_report_sha256": hashlib.sha256(args.stream_report.read_bytes()).hexdigest(),
              "windows": len(a), "probability_max_abs_difference": float(np.max(np.abs(a - b))),
              "window_flag_agreement": float(np.mean(np.asarray(batch["window_review_flags"]) == stream["window_review_flags"])),
              "matched_events": len(stream["events"]), "event_frame_identity_agreement": True,
              "feature_width_limit": stream["feature_width_limit"], "display_width": stream["display_width"],
              "stream_processing_sec": stream["processing_sec"],
              "events": [{key: event[key] for key in ("input_frame_index", "trigger_time_sec", "decision_available_wall_sec", "evidence_available_wall_sec", "post_context_truncated")} for event in stream["events"]],
              "scope": "Sequential CPU inference and batch inference on identical decoded source frames; file replay throughput is distinct from camera arrival-to-decision latency"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
