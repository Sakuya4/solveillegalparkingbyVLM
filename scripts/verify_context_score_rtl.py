from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.run_continuous_context_study import PROTOCOL, SELECTION, sha, dump, now
from scripts.verify_tcn_confidence_rtl import float_review_flags, tool_path
from hardware_sim.python_golden.tcn_confidence_model import context_decisions, UNORM_MAX
from illegal_parking.incident_presentation_events import presentation_episode_indices, evaluate_presentation_event

PLAN = ROOT / "docs/results/continuous_context_rtl_protocol.json"
OUTPUT = ROOT / "docs/results/continuous_context_rtl.json"
RESULTS = ROOT / "docs/results/continuous_context_results.json"


def audit_stream(starts, scores, threshold):
    tokens, decisions = context_decisions(starts, scores, threshold)
    flags = float_review_flags(starts, scores, threshold)
    episodes = set(presentation_episode_indices(starts, scores, threshold))
    quantized = [row["probability_code"] / UNORM_MAX for row in decisions]
    quantized_threshold = tokens[0][3] / UNORM_MAX
    summary = {"decisions": len(scores), "positive_changes": sum((value >= threshold) != row["positive"] for value, row in zip(scores, decisions)),
               "review_changes": sum(flag != row["review_flag"] for flag, row in zip(flags, decisions)),
               "episode_changes": sum((index in episodes) != row["episode_pulse"] for index, row in enumerate(decisions)),
               "maximum_probability_error": max((abs(a - b) for a, b in zip(scores, quantized)), default=0.)}
    return tokens, summary, quantized, quantized_threshold


def freeze():
    if PLAN.exists() or RESULTS.exists():
        raise FileExistsError("Freeze hardware verification before new study results")
    paths = [Path(__file__), ROOT / "scripts/verify_tcn_confidence_rtl.py", ROOT / "hardware_sim/python_golden/tcn_confidence_model.py",
             ROOT / "hardware_sim/rtl/tcn_confidence_gate.v", ROOT / "hardware_sim/testbench/tcn_confidence_gate_tb.v"]
    dump(PLAN, {"created_at_utc": now(), "training_protocol_sha256": sha(PROTOCOL),
                "implementation_sha256": {path.relative_to(ROOT).as_posix(): sha(path) for path in paths},
                "encoding": "UNORM16 floor(value*65535+0.5); same encoding for scores and TRAIN-selected thresholds",
                "endpoints": ["Integer-golden/RTL cycle mismatches", "Float/integer positive, review and episode changes", "Float/integer event-detection and trigger-time changes"],
                "cohorts": "Every selected model and corresponding calibration/test source, including empty sources; no score-dependent exclusion",
                "scope": "Post-inference control verification only; not Snapdragon NPU compute, synthesis, timing or power"})
    print(sha(PLAN))


def verify():
    if OUTPUT.exists():
        raise FileExistsError("Do not overwrite hardware results")
    plan = json.loads(PLAN.read_text())
    if plan["training_protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Hardware training protocol changed")
    for name, expected in plan["implementation_sha256"].items():
        if sha(ROOT / name) != expected:
            raise ValueError("Frozen hardware implementation changed")
    results, protocol, selection = (json.loads(path.read_text()) for path in (RESULTS, PROTOCOL, SELECTION))
    if results["selection_sha256"] != sha(SELECTION) or results["protocol_sha256"] != sha(PROTOCOL):
        raise ValueError("Verified study and selected policies do not match")
    definition = results["files"]["contexts"]
    source = ROOT / definition["path"]
    if sha(source) != definition["sha256"]:
        raise ValueError("Public confidence trace changed")
    groups = defaultdict(list)
    with source.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            groups[(row["cohort"], row["model"], row["path"])].append(row)
    tokens, summaries = [], []
    for key, policy in selection["policies"].items():
        if policy["selected"] is None:
            summaries.append({"model": key, "status": "unselected"})
            continue
        split = key.rsplit("_", 1)[-1]
        threshold = policy["selected"]["threshold"]
        for cohort in ("calibration", "test"):
            counts = defaultdict(int)
            maximum = 0.
            for name in protocol["memberships"][split][cohort + "_clip_paths"]:
                rows = sorted(groups.get((cohort, key, name), []), key=lambda row: int(row["start"]))
                starts, scores = [int(row["start"]) for row in rows], [float(row["probability"]) for row in rows]
                local, audit, quantized, encoded_threshold = audit_stream(starts, scores, threshold)
                tokens.extend(local)
                for field in ("decisions", "positive_changes", "review_changes", "episode_changes"):
                    counts[field] += audit[field]
                maximum = max(maximum, audit["maximum_probability_error"])
                report = {"path": name, "annotation_free_inference": True, "clock": "normalized_presentation_time",
                          "window_starts": starts, "window_end_times_sec": [float(row["end_pts_sec"]) for row in rows], "window_probabilities": scores}
                original = evaluate_presentation_event(report, protocol["annotations"][name], threshold)
                encoded = evaluate_presentation_event({**report, "window_probabilities": quantized}, protocol["annotations"][name], encoded_threshold)
                counts["event_changes"] += original.detected != encoded.detected
                counts["trigger_time_changes"] += original.trigger_time_sec != encoded.trigger_time_sec
                counts["source_runs"] += 1
            summaries.append({"model": key, "cohort": cohort, **dict(counts), "maximum_probability_error": maximum})
    folder = ROOT / "outputs/hardware/continuous_context"
    folder.mkdir(parents=True, exist_ok=True)
    vectors = folder / "vectors.txt"
    vectors.write_text("".join(" ".join(map(str, token)) + "\n" for token in tokens), encoding="ascii", newline="\n")
    compiler, runtime = tool_path("iverilog"), tool_path("vvp")
    executable = folder / "controller.vvp"
    subprocess.run([str(compiler), "-g2012", "-s", "tcn_confidence_gate_tb", "-o", str(executable),
                    str(ROOT / "hardware_sim/rtl/tcn_confidence_gate.v"), str(ROOT / "hardware_sim/testbench/tcn_confidence_gate_tb.v")], check=True, capture_output=True, timeout=30)
    completed = subprocess.run([str(runtime), executable.relative_to(ROOT).as_posix(), "+vectors=" + vectors.relative_to(ROOT).as_posix()],
                               cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    if f"PASS tcn_confidence_gate_tb tokens={len(tokens)}" not in completed.stdout:
        raise ValueError("The controller did not verify the entire trace")
    dump(OUTPUT, {"protocol_sha256": sha(PLAN), "results_sha256": sha(RESULTS), "selection_sha256": sha(SELECTION),
                  "completed_at_utc": now(), "tokens_checked": len(tokens), "rtl_golden_mismatches": 0, "summaries": summaries,
                  "vectors_sha256": sha(vectors), "compiler_sha256": sha(compiler), "runtime_sha256": sha(runtime),
                  "simulation_output": completed.stdout.strip(), "scope": plan["scope"]})
    print(json.dumps({"tokens_checked": len(tokens), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify new selected temporal models with the existing bit-accurate confidence RTL.")
    parser.add_argument("stage", choices=("freeze", "verify"))
    freeze() if parser.parse_args().stage == "freeze" else verify()
