from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
import numpy as np
from scripts.run_continuous_context_study import load_study, load_models, DIRECTORY, PROTOCOL, EXECUTION, sha, dump, now

PLAN = ROOT / "docs/results/continuous_context_latency_protocol.json"
OUTPUT = ROOT / "docs/results/continuous_context_latency.json"


def summarize_timings(values):
    samples = np.asarray(values, dtype=float)
    if samples.ndim != 1 or not len(samples) or not np.all(np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("CPU timings must be finite nonnegative samples")
    return {"calls": len(samples), "median_ms": float(np.median(samples) * 1000),
            "p95_ms": float(np.percentile(samples, 95) * 1000), "mean_ms": float(samples.mean() * 1000)}


def freeze():
    if PLAN.exists():
        raise FileExistsError("Latency input contracts are immutable")
    protocol, execution, _ = load_study()
    cache = ROOT / execution["calibration_cache"]["directory"]
    batch_path = cache / "batch.json"
    if sha(batch_path) != execution["calibration_cache"]["batch_sha256"]:
        raise ValueError("Declared TRAIN calibration cache changed")
    batch = json.loads(batch_path.read_text())
    inputs = []
    for name, definition in sorted(batch["clips"].items()):
        report_path = cache / definition["report"]
        report = json.loads(report_path.read_text())
        if report["window_starts"]:
            inputs.append({"path": name, "source_sha256": report["source_sha256"], "window_offset": 0,
                           "features_path": (cache / definition["features"]).relative_to(ROOT).as_posix(),
                           "features_sha256": definition["features_sha256"], "report_sha256": sha(report_path)})
        if len(inputs) == 10:
            break
    if len(inputs) != 10:
        raise ValueError("Ten supported TRAIN calibration sources are required")
    dump(PLAN, {"created_at_utc": now(), "protocol_sha256": sha(PROTOCOL), "execution_sha256": sha(EXECUTION),
                "benchmark_sha256": sha(Path(__file__)), "inputs": inputs,
                "input_selection": "First valid context from each of the first ten supported calibration-union source names; not selected by scores or labels",
                "warmup_calls_per_model": 20, "timed_calls_per_model": 1000, "batch_size": 1, "threads": 2,
                "schedule": "Deterministic round-robin over ten identical source contexts for all eight models",
                "scope": "Temporal-only CPU predict_probabilities including standardization; excludes detector, decoder, network, queues and VLM. Not Snapdragon/NPU latency.",
                "isolation": "Run only after research extraction/fitting/replay jobs finish; OS workload lock held. Other operating-system activity is not controlled."})
    print(sha(PLAN))


def run():
    import torch
    from threadpoolctl import threadpool_limits
    from illegal_parking.gpu_job import exclusive_gpu_job
    from illegal_parking.incident_tcn_inference import IncidentTcnPredictor
    from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite measured latency")
    protocol, execution, _ = load_study()
    plan = json.loads(PLAN.read_text())
    if plan["protocol_sha256"] != sha(PROTOCOL) or plan["execution_sha256"] != sha(EXECUTION) or plan["benchmark_sha256"] != sha(Path(__file__)):
        raise ValueError("Frozen latency contract changed")
    torch.set_num_threads(2)
    threadpool_limits(2)
    contexts = []
    for definition in plan["inputs"]:
        path = ROOT / definition["features_path"]
        if sha(path) != definition["features_sha256"]:
            raise ValueError("Latency input tensor changed")
        with np.load(path, allow_pickle=False) as data:
            if data["feature_names"].tolist() != list(CANDIDATE_MOTION_FEATURE_NAMES):
                raise ValueError("Latency feature channels changed")
            contexts.append(data["features"][definition["window_offset"]].copy())
    results = []
    with exclusive_gpu_job(ROOT):
        for key, definition in load_models().items():
            predictor = IncidentTcnPredictor.from_checkpoint(ROOT / definition["path"], "cpu")
            columns = [CANDIDATE_MOTION_FEATURE_NAMES.index(name) for name in predictor.feature_names]
            inputs = [matrix[:, columns][None, :, :] for matrix in contexts]
            for index in range(plan["warmup_calls_per_model"]):
                predictor.predict_probabilities(inputs[index % len(inputs)])
            timings = []
            for index in range(plan["timed_calls_per_model"]):
                started = time.perf_counter()
                predictor.predict_probabilities(inputs[index % len(inputs)])
                timings.append(time.perf_counter() - started)
            results.append({"model": key, "model_sha256": definition["sha256"], "feature_channels": len(columns),
                            **summarize_timings(timings), "timings_sec": timings})
    dump(OUTPUT, {"protocol_sha256": sha(PLAN), "models_sha256": sha(DIRECTORY / "models.json"),
                  "completed_at_utc": now(), "results": results, "scope": plan["scope"], "threads": 2})
    print(json.dumps([{key: row[key] for key in ("model", "calls", "median_ms", "p95_ms")} for row in results], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predeclare and measure matched-context temporal CPU inference costs.")
    parser.add_argument("stage", choices=("freeze", "run"))
    freeze() if parser.parse_args().stage == "freeze" else run()
