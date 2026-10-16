from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from scripts.prepare_presentation_protocol import sha256
from scripts.replay_presentation_events import load_event_protocol, validate_batch
from scripts.evaluate_presentation_events import model_reports
from illegal_parking.incident_tcn_inference import persistent_positive_flags
from illegal_parking.incident_presentation_events import evaluate_presentation_event, presentation_episode_indices
from hardware_sim.python_golden.tcn_confidence_model import UNORM_MAX, context_decisions


def float_review_flags(starts, probabilities, threshold):
    flags = []
    first = 0
    boundaries = [offset for offset in range(1, len(starts)) if starts[offset] != starts[offset - 1] + 15] + [len(starts)]
    for end in boundaries:
        flags.extend(persistent_positive_flags(probabilities[first:end], threshold, 2).tolist())
        first = end
    return flags


def tool_path(name):
    value = shutil.which(name)
    fallback = Path("C:/iverilog/bin") / (name + ".exe")
    if not value and fallback.exists():
        value = str(fallback)
    if not value:
        raise FileNotFoundError(f"Install {name} before publishing RTL results")
    return Path(value).resolve()


def main():
    parser = argparse.ArgumentParser(description="Verify actual frozen TCN confidence streams against a bit-accurate RTL persistence controller.")
    parser.add_argument("--iverilog", type=Path)
    parser.add_argument("--vvp", type=Path)
    args = parser.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(2)
    protocol_path = ROOT / "docs/results/presentation_event_protocol.json"
    protocol = load_event_protocol(protocol_path)
    selection_path = ROOT / "docs/results/presentation_event_selection.json"
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    control_path = ROOT / "docs/results/presentation_event_controls.json"
    control = json.loads(control_path.read_text(encoding="utf-8"))
    if control["protocol_sha256"] != sha256(protocol_path) or selection["protocol_sha256"] != sha256(protocol_path):
        raise ValueError("Only the frozen, prediction-verified study can supply controller traces")
    vector_path = ROOT / "hardware_sim/test_vectors/tcn_confidence_real.txt"
    trace_path = ROOT / "docs/results/tcn_confidence_rtl_trace.csv"
    output = ROOT / "docs/results/tcn_confidence_rtl.json"
    if any(path.exists() for path in (trace_path, output)):
        raise FileExistsError("Completed real-trace verification artifacts are immutable")
    tokens, rows, summaries = [], [], []
    for cohort in ("calibration", "test"):
        directory = ROOT / f"outputs/accident/presentation_events_{cohort}"
        _, reports = validate_batch(directory, protocol_path, protocol, cohort, sha256(selection_path) if cohort == "test" else None)
        if sha256(directory / "batch.json") != control[cohort + "_batch_sha256"]:
            raise ValueError("Raw confidence traces no longer match the checkpoint/feature audit")
        for key, definition in protocol["models"].items():
            policy = selection["policies"][key]["selected"]
            if policy is None:
                summaries.append({"cohort": cohort, "model": key, "status": "unselected"})
                continue
            threshold = policy["threshold"]
            counts = defaultdict(int)
            maximum_error, threshold_error = 0., 0.
            for report in model_reports(protocol, reports, key, cohort):
                starts, probabilities = report["window_starts"], report["window_probabilities"]
                local_tokens, decisions = context_decisions(starts, probabilities, threshold)
                line_indices = [len(tokens) + offset + 1 for offset, token in enumerate(local_tokens) if token[0]]
                tokens.extend(local_tokens)
                float_flags = float_review_flags(starts, probabilities, threshold)
                float_episodes = set(presentation_episode_indices(starts, probabilities, threshold))
                quantized = [decision["probability_code"] / UNORM_MAX for decision in decisions]
                threshold_code = local_tokens[0][3]
                quantized_threshold = threshold_code / UNORM_MAX
                threshold_error = abs(quantized_threshold - threshold)
                reference_flags = float_review_flags(starts, quantized, quantized_threshold)
                reference_episodes = set(presentation_episode_indices(starts, quantized, quantized_threshold))
                if [decision["review_flag"] for decision in decisions] != reference_flags or {offset for offset, decision in enumerate(decisions) if decision["episode_pulse"]} != reference_episodes:
                    raise ValueError("Integer golden disagrees with the frozen gate semantics on quantized values")
                for offset, (probability, decision) in enumerate(zip(probabilities, decisions)):
                    error = abs(quantized[offset] - probability)
                    maximum_error = max(maximum_error, error)
                    counts["positive_decision_changes"] += int((probability >= threshold) != decision["positive"])
                    counts["review_flag_changes"] += int(float_flags[offset] != decision["review_flag"])
                    counts["episode_pulse_changes"] += int((offset in float_episodes) != decision["episode_pulse"])
                    rows.append({"vector_line": line_indices[offset], "cohort": cohort, "model": key,
                                 "path": report["path"], "source_sha256": report["source_sha256"],
                                 "start_slot": starts[offset], "decision_pts_sec": report["window_end_times_sec"][offset],
                                 "probability": probability, "threshold": threshold,
                                 "probability_unorm16": decision["probability_code"], "threshold_unorm16": threshold_code,
                                 "float_review": int(float_flags[offset]), "integer_review": int(decision["review_flag"]),
                                 "float_episode": int(offset in float_episodes), "integer_episode": int(decision["episode_pulse"])})
                annotation = protocol["annotations"][report["path"]]
                original = evaluate_presentation_event(report, annotation, threshold)
                integer = evaluate_presentation_event({**report, "window_probabilities": quantized}, annotation, quantized_threshold)
                counts["event_detection_changes"] += int(original.detected != integer.detected)
                counts["trigger_time_changes"] += int(original.trigger_time_sec != integer.trigger_time_sec)
                counts["clips"] += 1
                counts["valid_contexts"] += len(starts)
                counts["reset_tokens"] += sum(token[1] for token in local_tokens)
                counts["idle_tokens"] += sum(not token[0] and not token[1] for token in local_tokens)
            summaries.append({"cohort": cohort, "model": key, "status": "verified", **dict(counts),
                              "maximum_probability_error": maximum_error, "threshold_error": threshold_error})
    vector_text = "".join(" ".join(map(str, token)) + "\n" for token in tokens)
    if vector_path.exists():
        if vector_path.read_text(encoding="ascii") != vector_text:
            raise ValueError("Partial verification vectors changed; preserve and investigate them")
    else:
        vector_path.write_text(vector_text, encoding="ascii", newline="\n")
    compiler, runtime = (args.iverilog or tool_path("iverilog")), (args.vvp or tool_path("vvp"))
    build = ROOT / "outputs/hardware"
    build.mkdir(parents=True, exist_ok=True)
    executable = build / "tcn_confidence_gate.vvp"
    rtl = ROOT / "hardware_sim/rtl/tcn_confidence_gate.v"
    bench = ROOT / "hardware_sim/testbench/tcn_confidence_gate_tb.v"
    subprocess.run([str(compiler), "-g2012", "-s", "tcn_confidence_gate_tb", "-o", str(executable), str(rtl), str(bench)], check=True, capture_output=True, text=True, timeout=30)
    simulation = subprocess.run([str(runtime), executable.relative_to(ROOT).as_posix(), "+vectors=" + vector_path.relative_to(ROOT).as_posix()], cwd=ROOT, check=True, capture_output=True, text=True, timeout=60)
    if f"PASS tcn_confidence_gate_tb tokens={len(tokens)}" not in simulation.stdout:
        raise ValueError("The self-checking testbench did not verify the complete token stream")
    with trace_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    version = subprocess.run([str(compiler), "-V"], capture_output=True, text=True, timeout=30)
    record = {"protocol_sha256": sha256(protocol_path), "selection_sha256": sha256(selection_path),
              "prediction_audit_sha256": sha256(control_path), "vectors_sha256": sha256(vector_path),
              "trace_sha256": sha256(trace_path), "compiler": version.stdout.splitlines()[0],
              "compiler_sha256": sha256(compiler), "runtime_sha256": sha256(runtime),
              "implementation_sha256": {path.relative_to(ROOT).as_posix(): sha256(path) for path in (Path(__file__), rtl, bench, ROOT / "hardware_sim/python_golden/tcn_confidence_model.py")},
              "tokens_checked": len(tokens), "model_decisions": len(rows), "rtl_golden_mismatches": 0,
              "simulation_output": simulation.stdout.strip(), "summaries": summaries,
              "encoding": "UNORM16: floor(value * 65535 + 0.5), endpoints exact, reconstruction error <= 0.5/65535",
              "scope": "Confidence/persistence controller verification on frozen real-model streams; not NPU arithmetic, target timing, power, synthesis or Snapdragon measurements"}
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: record[key] for key in ("tokens_checked", "model_decisions", "rtl_golden_mismatches", "compiler")}, indent=2))


if __name__ == "__main__":
    main()
