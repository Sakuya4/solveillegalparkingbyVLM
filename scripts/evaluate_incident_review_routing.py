from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))

from illegal_parking.accident_dataset import load_accident_manifest
from illegal_parking.incident_evaluation import match_report_to_accident_clip
from hardware_sim.python_golden.incident_review_workload import build_incident_review_workload, summarize_review_retention
from hardware_sim.python_golden.npu_queue_model import EdgeQueueProfile, simulate_edge_trace_queues
from hardware_sim.python_golden.review_routing import meter_review_flags


def main():
    parser = argparse.ArgumentParser(description="Replay TCN-gated incident episodes and measure event retention after routing.")
    parser.add_argument("--batch-dir", type=Path, default=ROOT / "outputs/accident/event_batch_iid_100")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data/raw/accident/full/extracted/metadata-real.csv")
    parser.add_argument("--review-report", type=Path, default=ROOT / "docs/results/incident_vlm_qwen_greedy_batch.json")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/incident_review_routing_iid.json")
    parser.add_argument("--systemc-executable", type=Path)
    args = parser.parse_args()
    summary_path = args.batch_dir / "batch_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary["failure_count"] or summary["requested_clips"] != summary["completed_clips"] + summary["resumed_clips"]:
        raise ValueError("Review replay requires a complete failure-free event batch")
    clips = load_accident_manifest(args.metadata)
    reports, hashes = [], []
    for name in summary["report_paths"]:
        path = args.batch_dir / "reports" / Path(name).name
        report = json.loads(path.read_text(encoding="utf-8"))
        reports.append((report, match_report_to_accident_clip(report, clips)))
        hashes.append({"report": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    measured = json.loads(args.review_report.read_text(encoding="utf-8"))
    service = float(measured["mean_review_sec"])
    if not np.isfinite(service) or service <= 0:
        raise ValueError("Measured service time must be finite and positive")
    fps, rows = 7.5, []
    for gate in (1, 2):
        flags, matching, clip_count = build_incident_review_workload(reports, gate)
        for cameras in (1, 4):
            raw = np.repeat(flags[:, None], cameras, axis=1)
            for policy, fraction in (("all_episodes", None), ("metered_80pct", 0.8), ("metered_100pct", 1.0)):
                admitted = raw if fraction is None else meter_review_flags(raw, camera_fps=fps, rate_per_sec=fraction / service, burst=cameras)
                profile = EdgeQueueProfile(cameras, fps, len(flags) / fps, 3.9, 3.0, 8, float(admitted.mean()), service * 1000, 8)
                details = []
                queue = simulate_edge_trace_queues(profile, admitted.reshape(-1), review_events=details).to_dict()
                if queue["pending_frames"] or queue["dropped_frames"]:
                    raise ValueError("This review-only comparison requires complete front-end processing")
                if queue["generated_candidates"] != int(admitted.sum()):
                    raise AssertionError("Admitted candidate conservation failed")
                row = {"min_positive_windows": gate, "camera_count": cameras, "policy": policy,
                       "raw_episodes": int(raw.sum()), "budget_denied": int(raw.sum() - admitted.sum()),
                       "admitted_per_camera": admitted.sum(axis=0).tolist(), "queue": queue,
                       "event_retention": summarize_review_retention(raw, admitted, matching, details, clips=clip_count, fps=fps)}
                if args.systemc_executable:
                    name = f"{summary['split_scheme']}_gate{gate}_cam{cameras}_{policy}"
                    path = ROOT / "outputs/hardware/incident_review_routing" / f"{name}.txt"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text("".join("1\n" if flag else "0\n" for flag in admitted.reshape(-1)), encoding="ascii")
                    command = [str(args.systemc_executable.resolve()), "--camera-count", str(cameras), "--camera-fps", str(fps), "--duration-sec", str(profile.duration_sec), "--detector-latency-ms", "3.9", "--temporal-latency-ms", "3.0", "--npu-queue-capacity", "8", "--review-latency-ms", str(profile.review_latency_ms), "--review-queue-capacity", "8", "--candidate-flags", path.relative_to(ROOT).as_posix()]
                    replay = subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60)
                    counters = {key.strip(): int(value) for line in replay.stdout.splitlines() for key, separator, value in [line.partition("=")] if separator and value.strip().isdigit()}
                    keys = ("generated_frames", "processed_frames", "pending_frames", "dropped_frames", "generated_candidates", "completed_reviews", "pending_reviews", "dropped_candidates")
                    if any(counters.get(key) != queue[key] for key in keys):
                        raise AssertionError(f"SystemC counter mismatch: {name}")
                    row["systemc_python_parity"] = True
                rows.append(row)
    output = {"split_scheme": summary["split_scheme"], "clips": len(reports), "source_batch_sha256": hashlib.sha256(summary_path.read_bytes()).hexdigest(),
              "source_reports": hashes, "metadata_sha256": hashlib.sha256(args.metadata.read_bytes()).hexdigest(),
              "model_sha256": hashlib.sha256(Path(summary["model"]).read_bytes()).hexdigest(),
              "review_report_sha256": hashlib.sha256(args.review_report.read_bytes()).hexdigest(), "review_service_sec": service,
              "nominal_camera_fps": fps, "workload_duration_sec": len(flags) / fps,
              "front_end_service_ms": {"detector": 3.9, "temporal": 3.0}, "policies": rows,
              "protocol": "Causal TCN persistence and episode starts; fixed benchmark clip order, synchronized camera replication, zero inter-clip gap; post-hoc annotation matching at -1/+3 seconds",
              "endpoint": "Retention of annotation-matched detected episodes through admission and simulated review-service completion, not VLM classification accuracy",
              "timing_scope": "Constant review service equals measured Qwen mean; front-end service times are scenario parameters, not Snapdragon measurements; outstanding reviews remain pending at observation end"}
    if args.systemc_executable:
        output["systemc_executable_sha256"] = hashlib.sha256(args.systemc_executable.read_bytes()).hexdigest()
        output["systemc_source_sha256"] = hashlib.sha256((ROOT / "hardware_sim/systemc/npu_queue_sim.cpp").read_bytes()).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps([{key: row[key] for key in ("min_positive_windows", "camera_count", "policy", "event_retention")} for row in rows], indent=2))


if __name__ == "__main__":
    main()
