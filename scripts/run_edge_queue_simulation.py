from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_sim.python_golden.npu_queue_model import (
    EdgeQueueProfile,
    simulate_camera_capacity_sweep,
    simulate_edge_queues,
)


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Simulate shared edge NPU and review queues.")
    parser.add_argument("--camera-count", type=int, default=4)
    parser.add_argument("--camera-fps", type=float, default=15.0)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--detector-latency-ms", type=float, default=29.0)
    parser.add_argument("--temporal-latency-ms", type=float, default=3.0)
    parser.add_argument("--npu-queue-capacity", type=int, default=8)
    parser.add_argument("--candidate-probability", type=float, default=0.01)
    parser.add_argument("--review-latency-ms", type=float, default=300.0)
    parser.add_argument("--review-queue-capacity", type=int, default=8)
    parser.add_argument("--output", default="outputs/hardware/edge_queue_report.json")
    parser.add_argument("--sweep-max-cameras", type=int, default=0)
    args = parser.parse_args()

    profile = EdgeQueueProfile(
        camera_count=args.camera_count,
        camera_fps=args.camera_fps,
        duration_sec=args.duration_sec,
        detector_latency_ms=args.detector_latency_ms,
        temporal_latency_ms=args.temporal_latency_ms,
        npu_queue_capacity=args.npu_queue_capacity,
        candidate_probability=args.candidate_probability,
        review_latency_ms=args.review_latency_ms,
        review_queue_capacity=args.review_queue_capacity,
    )
    result = simulate_edge_queues(profile)
    payload = {"profile": profile.__dict__, "result": result.to_dict()}
    if args.sweep_max_cameras:
        payload["camera_capacity_sweep"] = simulate_camera_capacity_sweep(profile, args.sweep_max_cameras)
    output_path = _resolve(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _resolve(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


if __name__ == "__main__":
    raise SystemExit(main())
