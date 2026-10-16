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
    parser = argparse.ArgumentParser(description="Replay source-bound empirical VLM service variability against the constant-mean scenario.")
    parser.add_argument("--replicates", type=int, default=10)
    parser.add_argument("--systemc-executable", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/results/review_service_variability.json")
    args = parser.parse_args()
    if args.replicates <= 0:
        parser.error("--replicates must be positive")
    measured_path = ROOT / "docs/results/incident_vlm_qwen_greedy_batch.json"
    measured = json.loads(measured_path.read_text(encoding="utf-8"))
    provenance_path = ROOT / "docs/results/vlm_review_dispositions.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    services, source_hashes = [], []
    for case in provenance["source_cases"]:
        path = ROOT / "outputs/validation/incident_vlm_paired" / case["case_id"] / "qwen_greedy_result.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != case["report_sha256"]:
            raise ValueError("Measured review source changed")
        value = float(json.loads(path.read_text(encoding="utf-8"))["elapsed_sec"])
        if not np.isfinite(value) or value <= 0:
            raise ValueError("Measured review durations must be finite and positive")
        services.append(value * 1000)
        source_hashes.append({"case_id": case["case_id"], "source_sha256": case["report_sha256"], "review_sec": value})
    if len(services) != measured["total_cases"] or not np.isclose(np.mean(services) / 1000, measured["mean_review_sec"], rtol=0, atol=1e-8):
        raise ValueError("Empirical service samples do not reproduce the published mean")
    mean_ms, fps = float(np.mean(services)), 7.5
    clips = load_accident_manifest(ROOT / "data/raw/accident/full/extracted/metadata-real.csv")
    studies, summaries, replay_hashes = [], [], {}
    for split, directory in (("iid", "event_batch_iid_100"), ("geographic", "event_batch_geographic_100_20261016")):
        batch_path = ROOT / "outputs/accident" / directory / "batch_summary.json"
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if batch["failure_count"] or batch["requested_clips"] != batch["completed_clips"] + batch["resumed_clips"]:
            raise ValueError("Service sensitivity requires complete source batches")
        reports = []
        for name in batch["report_paths"]:
            path = batch_path.parent / "reports" / Path(name.replace("\\", "/")).name
            report = json.loads(path.read_text(encoding="utf-8"))
            reports.append((report, match_report_to_accident_clip(report, clips)))
        replay_hashes[split] = hashlib.sha256(batch_path.read_bytes()).hexdigest()
        for gate in (1, 2):
            flags, matching, count = build_incident_review_workload(reports, gate)
            for cameras in (1, 4):
                raw = np.repeat(flags[:, None], cameras, axis=1)
                for policy in ("all_episodes", "metered_100pct"):
                    admitted = raw if policy == "all_episodes" else meter_review_flags(raw, camera_fps=fps, rate_per_sec=1000 / mean_ms, burst=cameras)
                    profile = EdgeQueueProfile(cameras, fps, len(flags) / fps, 3.9, 3.0, 8, float(admitted.mean()), mean_ms, 8)
                    rows = []
                    for seed in [None, *range(42, 42 + args.replicates)]:
                        service = None if seed is None else np.random.default_rng(seed).choice(services, size=admitted.size)
                        details = []
                        queue = simulate_edge_trace_queues(profile, admitted.reshape(-1), review_service_ms=service, review_events=details).to_dict()
                        retention = summarize_review_retention(raw, admitted, matching, details, clips=count, fps=fps)
                        row = {"split": split, "gate": gate, "cameras": cameras, "policy": policy, "service_mode": "constant_mean" if seed is None else "empirical_resampling", "seed": seed, "queue": queue, "event_retention": retention}
                        if args.systemc_executable and seed in (None, 42):
                            name = f"{split}_gate{gate}_cam{cameras}_{policy}_{seed}"
                            base = ROOT / "outputs/hardware/service_variability"
                            base.mkdir(parents=True, exist_ok=True)
                            flag_path = base / f"{name}_flags.txt"
                            flag_path.write_text("".join("1\n" if flag else "0\n" for flag in admitted.reshape(-1)), encoding="ascii")
                            command = [str(args.systemc_executable.resolve()), "--camera-count", str(cameras), "--camera-fps", str(fps), "--duration-sec", str(profile.duration_sec), "--detector-latency-ms", "3.9", "--temporal-latency-ms", "3.0", "--review-latency-ms", str(mean_ms), "--candidate-flags", flag_path.relative_to(ROOT).as_posix()]
                            if service is not None:
                                service_path = base / f"{name}_service.txt"
                                service_path.write_text("".join(f"{value:.12f}\n" for value in service), encoding="ascii")
                                command.extend(["--review-service-ms", service_path.relative_to(ROOT).as_posix()])
                                row["service_trace_sha256"] = hashlib.sha256(service_path.read_bytes()).hexdigest()
                            replay = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, check=True)
                            counters = {key.strip(): int(value) for line in replay.stdout.splitlines() for key, separator, value in [line.partition("=")] if separator and value.strip().isdigit()}
                            keys = ("generated_frames", "processed_frames", "pending_frames", "dropped_frames", "generated_candidates", "completed_reviews", "pending_reviews", "dropped_candidates")
                            if any(counters.get(key) != queue[key] for key in keys):
                                raise AssertionError(f"SystemC service replay mismatch: {name}")
                            row["systemc_python_parity"] = True
                        rows.append(row)
                    studies.extend(rows)
                    summary = {key: rows[0][key] for key in ("split", "gate", "cameras", "policy")}
                    for metric in ("completed_reviews", "dropped_candidates", "pending_reviews"):
                        values = [row["queue"][metric] for row in rows[1:]]
                        summary[metric] = {"constant_mean": rows[0]["queue"][metric], "empirical_median": float(np.median(values)), "empirical_min": min(values), "empirical_max": max(values)}
                    for label, key in (("matched_completed_events", "retained_after_completed_service"), ("review_p95_sec", "completed_review_latency_sec")):
                        values = [row["event_retention"][key]["p95"] if key == "completed_review_latency_sec" else row["event_retention"][key] for row in rows]
                        summary[label] = {"constant_mean": values[0], "empirical_median": float(np.median(values[1:])), "empirical_min": min(values[1:]), "empirical_max": max(values[1:])}
                    summaries.append(summary)
                    print(f"Completed service scenarios: {split}, gate={gate}, cameras={cameras}, {policy}", flush=True)
    result = {"review_model": measured["model"], "model_revision": measured["model_revision"],
              "measured_service_sec": {"count": len(services), "mean": mean_ms / 1000, "std": float(np.std(services, ddof=1) / 1000), "minimum": min(services) / 1000, "maximum": max(services) / 1000},
              "service_samples": source_hashes, "source_batches_sha256": replay_hashes,
              "replicates": args.replicates, "assignment_seeds": list(range(42, 42 + args.replicates)),
              "assignment": "Independent empirical draws indexed by source frame, shared across routing policies; each policy's admissions remain fixed within its paired constant/variable comparison",
              "timing_contract": "Completion-before-arrival FIFO semantics at equal timestamps; positive half-up frame counts; endpoint completions included",
              "scope": "Sensitivity of benchmark queue/event retention to a 20-review empirical service distribution; scenario ranges are not population confidence intervals, field prevalence, or Snapdragon measurements",
              "front_end_service_ms": {"detector": 3.9, "temporal": 3.0}, "nominal_camera_fps": fps,
              "source_hops": {"iid": 5, "geographic": 15}, "summaries": summaries, "scenarios": studies}
    if args.systemc_executable:
        result["systemc_executable_sha256"] = hashlib.sha256(args.systemc_executable.read_bytes()).hexdigest()
        result["systemc_source_sha256"] = hashlib.sha256((ROOT / "hardware_sim/systemc/npu_queue_sim.cpp").read_bytes()).hexdigest()
        result["systemc_parity_checks"] = sum(row.get("systemc_python_parity", False) for row in studies)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
