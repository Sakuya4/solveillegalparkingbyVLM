from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def workflow_steps() -> list[tuple[str, list[str]]]:
    dense = "outputs/validation/incident_vlm_paired_dense8"
    sequence = "data/processed/accident/candidate_roi_sequences_full.npz"
    steps = [
        ("greedy_qwen", ["scripts/run_incident_vlm_batch.py", "review", "--provider", "qwen"]),
        ("greedy_llava", ["scripts/run_incident_vlm_batch.py", "review", "--provider", "llava"]),
        ("dense_evidence", ["scripts/run_incident_vlm_batch.py", "prepare", "--frames", "8", "--output", dense]),
        ("dense_qwen", ["scripts/run_incident_vlm_batch.py", "review", "--provider", "qwen", "--output", dense]),
        ("full_candidate_features", ["scripts/extract_accident_candidate_roi_features.py", "--metadata", "data/raw/accident/metadata-real.csv", "--dataset-root", "data/raw/accident/full/extracted", "--threads", "2", "--output", "data/processed/accident/candidate_roi_features_full.csv", "--sequence-output", sequence, "--trace-output", "outputs/accident/candidate_roi_trace_full.jsonl", "--report", "outputs/accident/candidate_roi_features_full_report.json"]),
    ]
    for feature in ("candidate", "global"):
        for split in ("iid", "geographic"):
            stem = f"{feature}_full_tcn_{split}_fpr20"
            steps.append((stem, ["scripts/train_incident_tcn.py", "--sequences", sequence, "--split-scheme", split, "--include-prefix", feature + "_", "--target-train-fpr", "0.2", "--epochs", "40", "--seed", "42", "--threads", "2", "--device", "cuda", "--model-output", f"models/{stem}.pt", "--report-output", f"outputs/accident/{stem}_report.json", "--predictions-output", f"outputs/accident/{stem}_predictions.csv"]))
    return steps


def representation_steps() -> list[tuple[str, list[str]]]:
    features = "data/processed/accident/videomae_small_features_full.csv"
    steps = [("videomae_full_features", ["scripts/extract_accident_videomae_features.py", "--max-clips", "2027", "--batch-size", "8", "--threads", "2", "--output", features, "--report", "outputs/accident/videomae_full_features_report.json"])]
    for split in ("iid", "geographic"):
        stem = f"videomae_full_frozen_{split}_fpr20"
        steps.append((stem, ["scripts/train_incident_motion_baseline.py", "--features", features, "--include-prefix", "videomae_", "--split-scheme", split, "--target-train-fpr", "0.2", "--seed", "42", "--model-output", f"models/{stem}.json", "--report-output", f"outputs/accident/{stem}_report.json", "--predictions-output", f"outputs/accident/{stem}_predictions.csv"]))
    for split in ("iid", "geographic"):
        stem = f"videomae_full_finetuned_{split}_fpr20"
        steps.append((stem, ["scripts/finetune_incident_videomae.py", "--max-clips", "2027", "--split-scheme", split, "--epochs", "15", "--batch-size", "8", "--threads", "2", "--seed", "42", "--cache-dir", "data/processed/accident/videomae_pixel_cache_full", "--model-output", f"models/{stem}.pt", "--report-output", f"outputs/accident/{stem}_report.json", "--predictions-output", f"outputs/accident/{stem}_predictions.csv"]))
    return steps


def resolution_steps() -> list[tuple[str, list[str]]]:
    steps = []
    for split, suffix in (("iid", "iid"), ("geographic", "geo")):
        directory = f"outputs/accident/event_resolution320_{split}_100"
        steps.append((f"resolution320_{split}", ["scripts/run_accident_event_batch.py", "--split-scheme", split, "--model", f"models/candidate500_tcn_only_{suffix}_fpr20.pt", "--max-clips", "100", "--sampling-seed", "42", "--threads", "2", "--device", "0", "--feature-width", "320", "--output-dir", directory]))
        for profile, gate in (("responsive", "1"), ("conservative", "2")):
            steps.append((f"resolution320_{split}_{profile}", ["scripts/evaluate_incident_reports.py", "--metadata", "data/raw/accident/full/extracted/metadata-real.csv", "--report-glob", directory + "/reports/*.json", "--min-positive-windows", gate, "--output", directory + f"/{profile}_evaluation.json"]))
    return steps


def main():
    parser = argparse.ArgumentParser(description="Run the bounded, sequential research queue with resumable checkpoints.")
    parser.add_argument("--deadline", required=True, help="Timezone-aware ISO deadline, e.g. 2026-10-17T04:00:00+00:00")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--keep-awake-until-deadline", action="store_true")
    parser.add_argument("--phase", choices=("initial", "representations", "resolution"), default="initial")
    args = parser.parse_args()
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None:
        parser.error("--deadline must include a timezone")
    steps = {"initial": workflow_steps, "representations": representation_steps, "resolution": resolution_steps}[args.phase]()
    if args.dry_run:
        print(json.dumps(steps, indent=2))
        return
    directory = ROOT / "outputs/research_runs"
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory / ("overnight_status.json" if args.phase == "initial" else f"{args.phase}_status.json")
    import psutil
    for other_path in directory.glob("*status.json"):
        if other_path == state_path:
            continue
        other = json.loads(other_path.read_text(encoding="utf-8"))
        child_pid = other.get("child_pid")
        if other.get("status") == "running" and child_pid and psutil.pid_exists(child_pid):
            expected = Path(other["command"][1]).name
            if any(Path(part).name == expected for part in psutil.Process(child_pid).cmdline()[1:]):
                raise RuntimeError(f"Research child {child_pid} is already active; inspect {other_path}")
    previous = json.loads(state_path.read_text(encoding="utf-8")) if state_path.exists() else {}
    if previous.get("status") == "running":
        import psutil
        pid = previous.get("pid")
        if pid and psutil.pid_exists(pid) and any("run_overnight_research.py" in part for part in psutil.Process(pid).cmdline()):
            raise RuntimeError(f"Research runner {pid} is already active")
    completed = previous.get("completed_steps", [])
    state = {"pid": os.getpid(), "status": "running", "deadline": deadline.isoformat(), "completed_steps": completed, "current_step": None, "child_pid": None}

    def persist():
        state["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        temporary = state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)

    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONUNBUFFERED": "1", "PYTHONIOENCODING": "utf-8"})
    child = None
    keep_awake = False
    if args.keep_awake_until_deadline and os.name == "nt":
        import ctypes
        keep_awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
    state["temporary_system_wake_request"] = keep_awake
    persist()
    try:
        reuse_prefix = True
        for step_index, (name, arguments) in enumerate(steps):
            signature = hashlib.sha256(json.dumps(arguments).encode() + (ROOT / arguments[0]).read_bytes()).hexdigest()
            if reuse_prefix and any(row["name"] == name and row["signature"] == signature for row in completed):
                continue
            if reuse_prefix:
                completed[:] = completed[:step_index]
            reuse_prefix = False
            if datetime.now(timezone.utc) >= deadline:
                state["status"] = "deadline_reached"
                break
            state["current_step"] = name
            state["command"] = [sys.executable, *arguments]
            state["log"] = str((directory / f"{name}.log").relative_to(ROOT))
            started = time.perf_counter()
            with (directory / f"{name}.log").open("w", encoding="utf-8") as log:
                child = subprocess.Popen([sys.executable, *arguments], cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
                state["child_pid"] = child.pid
                persist()
                while child.poll() is None:
                    if datetime.now(timezone.utc) >= deadline:
                        child.terminate()
                        state["status"] = "deadline_reached"
                        break
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        persist()
                if state["status"] == "deadline_reached":
                    break
                if child.returncode:
                    raise RuntimeError(f"Step {name} exited with code {child.returncode}; inspect {state['log']}")
            completed.append({"name": name, "signature": signature, "elapsed_sec": time.perf_counter() - started})
            persist()
        else:
            state["status"] = "complete"
            state["current_step"] = None
            state["child_pid"] = None
            persist()
            if keep_awake:
                state["wake_request_purpose"] = "Allow scheduled research continuations until the requested deadline"
                persist()
                while datetime.now(timezone.utc) < deadline:
                    time.sleep(30)
    except BaseException as exc:
        state["status"] = "failed"
        state["error"] = str(exc)
        raise
    finally:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        state["child_pid"] = None
        if keep_awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            state["temporary_system_wake_request"] = False
        persist()


if __name__ == "__main__":
    main()
