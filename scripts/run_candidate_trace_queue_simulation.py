from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_sim.python_golden.candidate_trace_workload import (
    build_candidate_flags,
    load_candidate_trace,
    repeat_candidate_flags,
)
from hardware_sim.python_golden.npu_queue_model import EdgeQueueProfile, simulate_edge_trace_queues


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay model-produced candidate ROI traces through the edge queue model."
    )
    parser.add_argument("--trace", default="outputs/accident/candidate_roi_trace_500.jsonl")
    parser.add_argument("--score-threshold", type=float, default=0.95)
    parser.add_argument("--trace-fps", type=float, default=7.5)
    parser.add_argument("--camera-count", type=int, default=4)
    parser.add_argument("--detector-latency-ms", type=float, default=3.9)
    parser.add_argument("--temporal-latency-ms", type=float, default=3.0)
    parser.add_argument("--npu-queue-capacity", type=int, default=8)
    parser.add_argument("--review-latency-ms", type=float, default=300.0)
    parser.add_argument("--review-queue-capacity", type=int, default=8)
    parser.add_argument("--output", default="outputs/hardware/candidate_trace_queue_report.json")
    parser.add_argument("--flags-output", default="outputs/hardware/candidate_trace_flags.txt")
    args = parser.parse_args()
    if args.trace_fps <= 0:
        raise SystemExit("--trace-fps must be positive")

    trace_path = _resolve(args.trace)
    records = load_candidate_trace(trace_path)
    source_flags = build_candidate_flags(records, args.score_threshold)
    queue_flags = repeat_candidate_flags(source_flags, args.camera_count)
    duration_sec = len(source_flags) / args.trace_fps
    candidate_rate = float(source_flags.mean())
    profile = EdgeQueueProfile(
        camera_count=args.camera_count,
        camera_fps=args.trace_fps,
        duration_sec=duration_sec,
        detector_latency_ms=args.detector_latency_ms,
        temporal_latency_ms=args.temporal_latency_ms,
        npu_queue_capacity=args.npu_queue_capacity,
        candidate_probability=candidate_rate,
        review_latency_ms=args.review_latency_ms,
        review_queue_capacity=args.review_queue_capacity,
    )
    result = simulate_edge_trace_queues(profile, queue_flags)
    payload = {
        "workload": {
            "source": str(trace_path),
            "semantics": "annotation-free YOLO/ByteTrack plus motion candidate proposal score",
            "trace_records": len(records),
            "score_threshold": args.score_threshold,
            "source_candidate_count": int(source_flags.sum()),
            "source_candidate_rate": candidate_rate,
            "camera_replication": args.camera_count,
        },
        "profile": profile.__dict__,
        "result": result.to_dict(),
    }

    output_path = _resolve(args.output)
    flags_path = _resolve(args.flags_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flags_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    flags_path.write_text("\n".join("1" if flag else "0" for flag in queue_flags) + "\n", encoding="ascii")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
