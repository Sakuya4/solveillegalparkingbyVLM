from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.incident_event_discrimination import model_diagnostic, paired_ranking_interval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs(root, protocol):
    inventory, grouped = {}, defaultdict(list)
    for key in ("inventory", "contexts"):
        definition = protocol["inputs"][key]
        path = root / definition["path"]
        if digest(path) != definition["sha256"]:
            raise ValueError("Frozen TRAIN-only diagnostic input changed")
        with path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if len(rows) != definition["rows"]:
            raise ValueError("Diagnostic row counts changed")
        for row in rows:
            if row["cohort"] != "calibration" or row["model"] not in protocol["model_clip_paths"] or row["path"] not in protocol["model_clip_paths"][row["model"]]:
                raise ValueError("Test or unassigned rows cannot enter diagnostic analysis")
            identity = (row["model"], row["path"])
            if row["split"] != protocol["models"][row["model"]]["split"]:
                raise ValueError("Model-specific official-TRAIN membership changed")
            if key == "inventory":
                if identity in inventory:
                    raise ValueError("Duplicate diagnostic clip")
                inventory[identity] = row
            else:
                if identity not in inventory or row["source_sha256"] != inventory[identity]["source_sha256"]:
                    raise ValueError("Orphan or mismatched diagnostic context")
                grouped[identity].append(row)
    expected = {(key, path) for key, names in protocol["model_clip_paths"].items() for path in names}
    if set(inventory) != expected:
        raise ValueError("The full calibration source denominator is required")
    for identity, row in inventory.items():
        if len(grouped[identity]) != int(row["valid_contexts"]):
            raise ValueError("A valid diagnostic score was dropped or added")
    return inventory, grouped


def analyze(root, protocol):
    inventory, grouped = load_inputs(root, protocol)
    rows, outcomes = [], {}
    for key, definition in protocol["models"].items():
        names = protocol["model_clip_paths"][key]
        clips = [inventory[(key, name)] for name in names]
        contexts = {name: grouped[(key, name)] for name in names}
        summary, outcomes[key] = model_diagnostic(clips, contexts, protocol["thresholds"][key])
        rows.append({"model": key, "split": definition["split"], "summary": summary})
    pairs = []
    for split in ("iid", "geographic"):
        for feature in ("candidate", "global"):
            for metric in ("auroc", "average_precision"):
                pairs.append({"split": split, "feature": feature, "metric": metric, "contrast": "Fixed-time minus source-frame within-clip mean",
                              "interval": paired_ranking_interval(outcomes[f"fixed_{feature}_{split}"], outcomes[f"legacy_{feature}_{split}"], metric,
                                                                  repeats=protocol["contrasts"]["resamples"], seed=protocol["contrasts"]["seed"])})
    return rows, outcomes, pairs


def main():
    parser = argparse.ArgumentParser(description="TRAIN-only event-local ranking and controller diagnostics; no fitting or test evaluation.")
    parser.add_argument("--freeze-execution", action="store_true")
    args = parser.parse_args()
    base = ROOT / "docs/results"
    protocol_path = base / "event_discrimination_protocol.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol["status"] != "frozen_before_event_local_metric_computation":
        raise ValueError("A frozen diagnostic specification is required")
    execution_path = base / "event_discrimination_execution.json"
    code = [Path(__file__), ROOT / "Application/DebugVersion/src/illegal_parking/incident_event_discrimination.py"]
    if args.freeze_execution:
        if execution_path.exists():
            raise FileExistsError("Diagnostic execution freeze is immutable")
        record = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "protocol_sha256": digest(protocol_path),
                  "implementation_sha256": {path.relative_to(ROOT).as_posix(): digest(path) for path in code}}
        execution_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
        print(json.dumps(record, indent=2))
        return
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    if execution["protocol_sha256"] != digest(protocol_path) or any(digest(ROOT / name) != value for name, value in execution["implementation_sha256"].items()):
        raise ValueError("Diagnostic code or protocol changed after the execution freeze")
    if {name: importlib.metadata.version(name) for name in protocol["versions"]} != protocol["versions"]:
        raise ValueError("Ranking runtime differs from the frozen specification")
    output, trace_path = base / "event_discrimination_results.json", base / "event_discrimination_outcomes.csv"
    if output.exists() or trace_path.exists():
        raise FileExistsError("Completed diagnostic publications are immutable")
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    started = datetime.now(timezone.utc)
    if started <= datetime.fromisoformat(execution["created_at_utc"]):
        raise ValueError("Execution freeze must precede real-data analysis")
    rows, outcomes, pairs = analyze(ROOT, protocol)
    traces = []
    for key, values in outcomes.items():
        for row in values:
            metrics = row["within_clip"] or {}
            traces.append({"model": key, "split": protocol["models"][key]["split"], "path": row["path"], "source_sha256": row["source_sha256"],
                           "annotation_pts_sec": row["annotation_pts_sec"], **{name + "_contexts": count for name, count in row["contexts"].items()},
                           "support": row["support"], **{name: metrics.get(name) for name in ("auroc", "average_precision", "target_prevalence", "ap_minus_prevalence", "max_target_minus_max_prefix")},
                           "gate1_attainable": int(row["gate1_attainable"]), "gate2_attainable": int(row["gate2_attainable"]),
                           **{name + "_category": value["category"] for name, value in row["policies"].items()}})
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
        writer.writeheader()
        writer.writerows(traces)
    result = {"protocol_sha256": digest(protocol_path), "execution_sha256": digest(execution_path),
              "started_at_utc": started.isoformat(), "completed_at_utc": datetime.now(timezone.utc).isoformat(),
              "inputs": protocol["inputs"], "trace_sha256": digest(trace_path), "trace_rows": len(traces),
              "models": rows, "paired_ranking_contrasts": pairs, "scope": protocol["scope"]}
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"models": rows, "paired_ranking_contrasts": pairs}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
