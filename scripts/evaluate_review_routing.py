from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from hardware_sim.python_golden.candidate_trace_workload import build_candidate_flags, load_candidate_trace
from hardware_sim.python_golden.npu_queue_model import EdgeQueueProfile, simulate_edge_trace_queues
from hardware_sim.python_golden.review_routing import coalesce_candidate_flags, meter_review_flags


def main():
    parser = argparse.ArgumentParser(description="Compare causal proposal coalescing and measured-service review metering.")
    parser.add_argument("--trace", type=Path, default=ROOT / "outputs/accident/candidate_roi_trace_500.jsonl")
    parser.add_argument("--review-report", type=Path, default=ROOT / "docs/results/incident_vlm_qwen_greedy_batch.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/review_routing.json")
    parser.add_argument("--systemc-executable", type=Path)
    args = parser.parse_args()
    records = load_candidate_trace(args.trace)
    measured = json.loads(args.review_report.read_text(encoding="utf-8"))
    review_seconds = float(measured["mean_review_sec"])
    if not math.isfinite(review_seconds) or review_seconds <= 0:
        raise ValueError("Measured review service time must be finite and positive")
    cameras, fps = 4, 7.5
    raw = build_candidate_flags(records, 0.95)
    merged = coalesce_candidate_flags(records, raw, max_gap_steps=15, minimum_iou=0.1)
    raw_multi = np.repeat(raw[:, None], cameras, axis=1)
    merged_multi = np.repeat(merged[:, None], cameras, axis=1)
    profile = EdgeQueueProfile(cameras, fps, len(records) / fps, 3.9, 3.0, 8, float(raw.mean()), review_seconds * 1000, 8)
    plans = [
        ("all_proposals", raw_multi, None),
        ("causal_overlap_merge", merged_multi, None),
        ("all_proposals_metered", raw_multi, 1.0),
        ("merge_metered_80pct", merged_multi, 0.8),
        ("merge_metered_100pct", merged_multi, 1.0),
    ]
    labels = {"all_proposals": "All proposals", "causal_overlap_merge": "Overlap merge", "all_proposals_metered": "Metered proposals", "merge_metered_80pct": "Merge + 80% capacity", "merge_metered_100pct": "Merge + 100% capacity"}
    rows = []
    for name, proposals, utilization in plans:
        admitted = proposals if utilization is None else meter_review_flags(proposals, camera_fps=fps, rate_per_sec=utilization / review_seconds, burst=cameras)
        result = simulate_edge_trace_queues(replace(profile, candidate_probability=float(admitted.mean())), admitted.reshape(-1)).to_dict()
        if result["dropped_frames"] or result["pending_frames"]:
            raise ValueError("Review-only accounting requires complete front-end processing")
        counts = {
            "raw_proposals": int(raw_multi.sum()), "coalesced_proposals": int(raw_multi.sum() - proposals.sum()),
            "budget_denied": int(proposals.sum() - admitted.sum()),
            "admitted_proposals": int(admitted.sum()),
            "admitted_per_camera": admitted.sum(axis=0).astype(int).tolist(),
        }
        accounted = counts["coalesced_proposals"] + counts["budget_denied"] + result["completed_reviews"] + result["pending_reviews"] + result["dropped_candidates"]
        if accounted != counts["raw_proposals"] or result["generated_candidates"] != counts["admitted_proposals"]:
            raise AssertionError("Proposal conservation failed")
        row = {"policy": name, "label": labels[name], "meter_capacity_fraction": utilization, **counts, "queue": result}
        if args.systemc_executable:
            flag_path = ROOT / "outputs/hardware/review_routing" / f"{name}_flags.txt"
            flag_path.parent.mkdir(parents=True, exist_ok=True)
            flag_path.write_text("".join("1\n" if flag else "0\n" for flag in admitted.reshape(-1)), encoding="ascii")
            command = [str(args.systemc_executable.resolve()), "--camera-count", str(cameras), "--camera-fps", str(fps), "--duration-sec", str(profile.duration_sec), "--detector-latency-ms", str(profile.detector_latency_ms), "--temporal-latency-ms", str(profile.temporal_latency_ms), "--npu-queue-capacity", str(profile.npu_queue_capacity), "--review-latency-ms", str(profile.review_latency_ms), "--review-queue-capacity", str(profile.review_queue_capacity), "--candidate-flags", flag_path.relative_to(ROOT).as_posix()]
            replay = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=True)
            counters = {}
            for line in replay.stdout.splitlines():
                key, separator, value = line.partition("=")
                if separator and value.strip().isdigit():
                    counters[key.strip()] = int(value)
            keys = ("generated_frames", "processed_frames", "pending_frames", "dropped_frames", "generated_candidates", "completed_reviews", "pending_reviews", "dropped_candidates")
            differences = {key: counters.get(key) for key in keys if counters.get(key) != result[key]}
            if differences:
                raise AssertionError(f"SystemC replay differs for {name}: {differences}")
            version = re.search(r"SystemC\s+(\d+\.\d+\.\d+)", replay.stderr)
            row["systemc"] = {"version": version.group(1) if version else None, "counters": counters, "python_parity": True}
        rows.append(row)
    report = {
        "protocol": "Causal source-local overlap coalescing and strict shared token-bucket admission",
        "trace_sha256": hashlib.sha256(args.trace.read_bytes()).hexdigest(),
        "review_report_sha256": hashlib.sha256(args.review_report.read_bytes()).hexdigest(),
        "routing_inputs": ["candidate_available", "candidate_score", "candidate_bbox", "path", "frame_index"],
        "source_window_selection": "Concatenated sampled benchmark windows; replay is a capacity workload, not continuous traffic exposure",
        "nominal_camera_fps": fps, "camera_replication": cameras,
        "duration_sec": profile.duration_sec, "review_seconds": review_seconds,
        "front_end_service_ms": {"detector": 3.9, "temporal": 3.0},
        "candidate_threshold": 0.95, "merge_gap_steps": 15, "minimum_iou": 0.1,
        "source_frame_stride": 2, "token_burst": cameras, "queue_capacity": 8,
        "endpoint": "Proposal routing and queue accounting; incident recall is evaluated separately",
        "policies": rows,
        "sources": ["https://www.rfc-editor.org/rfc/rfc3290.html", "https://www.cs.princeton.edu/~ravian/publications/reducto_sigcomm20.pdf"],
    }
    if args.systemc_executable:
        report["systemc_executable_sha256"] = hashlib.sha256(args.systemc_executable.read_bytes()).hexdigest()
        report["systemc_source_sha256"] = hashlib.sha256((ROOT / "hardware_sim/systemc/npu_queue_sim.cpp").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
