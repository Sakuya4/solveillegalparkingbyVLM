from __future__ import annotations

import csv
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Application/DebugVersion/src"))
from illegal_parking.research_statistics import paired_cluster_coverage_interval


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_same(actual, expected):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            raise ValueError("Published metric keys differ")
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (float, int)) and not isinstance(expected, bool):
        if not math.isclose(actual, expected, abs_tol=1e-12, rel_tol=1e-12):
            raise ValueError(f"Metric mismatch: {actual} != {expected}")
    elif actual != expected:
        raise ValueError(f"Published value differs: {actual} != {expected}")


def clip_outcome(clip, contexts, threshold, constant=None):
    # Independent implementation: no production gate or event evaluator import.
    run, previous, episodes = 0, None, []
    event = float(clip["annotation_pts_sec"])
    if not math.isfinite(event) or event < 0 or not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("A finite event anchor and probability threshold are required")
    prefix, positive = 0, 0
    previous_time = -1.
    for row in contexts:
        start, end = int(row["start_slot"]), float(row["end_pts_sec"])
        score = float(row["probability"]) if constant is None else float(constant)
        if (start < 0 or start % 15 or (previous is not None and start <= previous)
                or not math.isfinite(end) or end < 0 or end <= previous_time or not math.isfinite(score) or not 0 <= score <= 1):
            raise ValueError("Invalid public context grid or score")
        if previous is None or start != previous + 15:
            run = 0
        run = run + 1 if score >= threshold else 0
        if run == 2:
            episodes.append(end)
        if end < event - 1:
            prefix += 1
            positive += score >= threshold
        previous, previous_time = start, end
    matching = [time for time in episodes if event - 1 <= time <= event + 3]
    trigger = matching[0] if matching else None
    return {"detected": int(bool(matching)), "trigger_pts_sec": trigger,
            "signed_delay_sec": trigger - event if trigger is not None else None,
            "early_episodes": sum(time < event - 1 for time in episodes),
            "late_episodes": sum(time > event + 3 for time in episodes),
            "prefix_contexts": prefix, "prefix_positives": positive}


def summarize(clips, outcomes):
    delays = [row["signed_delay_sec"] for row in outcomes if row["detected"]]
    delay = dict.fromkeys(("mean", "median", "p95", "minimum", "maximum"))
    if delays:
        delay = dict(zip(delay, map(float, (np.mean(delays), np.median(delays), np.percentile(delays, 95), min(delays), max(delays)))))
    support = [row for row in outcomes if row["prefix_contexts"]]
    fractions = [row["prefix_positives"] / row["prefix_contexts"] for row in support]
    contexts = sum(row["prefix_contexts"] for row in support)
    positives = sum(row["prefix_positives"] for row in support)
    detected = sum(row["detected"] for row in outcomes)
    early, late = (sum(row[field] for row in outcomes) for field in ("early_episodes", "late_episodes"))
    types = defaultdict(list)
    for clip, outcome in zip(clips, outcomes):
        types[clip["collision_type"]].append(outcome["detected"])
    summary = {"clip_count": len(clips), "detected_clips": detected, "missed_clips": len(clips) - detected,
               "event_recall": detected / len(clips), "early_alert_episodes": early, "late_alert_episodes": late,
               "trigger_delay_sec": delay, "by_collision_type": {name: {"clip_count": len(values), "detected_clips": sum(values),
                   "event_recall": sum(values) / len(values)} for name, values in sorted(types.items())},
               "off_target_episodes_per_clip": (early + late) / len(clips),
               "rejected_windows": sum(int(clip["rejected_contexts"]) for clip in clips),
               "clips_without_valid_windows": sum(int(clip["valid_contexts"]) == 0 for clip in clips)}
    prefix = {"clips_with_support": len(support), "clips_without_support": len(clips) - len(support),
              "pre_event_contexts": contexts, "positive_pre_event_contexts": positives,
              "macro_positive_fraction": float(np.mean(fractions)) if support else None,
              "pooled_positive_fraction": positives / contexts if contexts else None}
    return summary, prefix


def read_public(root):
    base = root / "docs/results"
    manifest = json.loads((base / "positive_context_trace.json").read_text(encoding="utf-8"))
    for field, name in (("protocol", "positive_exposure_protocol.json"), ("selection", "positive_exposure_selection.json"), ("results", "positive_exposure_results.json")):
        if digest(base / name) != manifest[field + "_sha256"]:
            raise ValueError("Public trace provenance changed")
    for definition in manifest["files"].values():
        if digest(root / definition["path"]) != definition["sha256"]:
            raise ValueError("Public trace bytes changed")
    def rows(key):
        with (root / manifest["files"][key]["path"]).open(encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    inventory, contexts = rows("inventory"), rows("contexts")
    if len(inventory) != manifest["inventory_rows"] or len(contexts) != manifest["context_rows"]:
        raise ValueError("Public trace row counts changed")
    clips = {}
    grouped = defaultdict(list)
    identity = lambda row: (row["cohort"], row["model"], row["path"])
    for row in inventory:
        key = identity(row)
        if key in clips:
            raise ValueError("Duplicate public clip identity")
        clips[key] = row
    for row in contexts:
        key = identity(row)
        if key not in clips or any(row[field] != clips[key][field] for field in ("split", "source_sha256")):
            raise ValueError("Orphan or mismatched public context")
        grouped[key].append(row)
    protocol = json.loads((base / "positive_exposure_protocol.json").read_text(encoding="utf-8"))
    expected = {(cohort, key, path) for key, definition in protocol["models"].items() for cohort in ("calibration", "test")
                for path in protocol["cohorts"][definition["split"]][cohort + "_clip_paths"]}
    if set(clips) != expected:
        raise ValueError("All frozen clips must remain in the public denominator")
    sources = {row["path"]: row["sha256"] for row in protocol["input_media"]}
    for key, row in clips.items():
        annotation = protocol["annotations"][row["path"]]
        if (len(grouped[key]) != int(row["valid_contexts"]) or row["source_sha256"] != sources[row["path"]]
                or row["split"] != protocol["models"][row["model"]]["split"]
                or row["collision_type"] != annotation["collision_type"]):
            raise ValueError("Context inventory differs from frozen sources or annotations")
        assert_same(float(row["annotation_pts_sec"]), annotation["event_time_sec"])
    return protocol, manifest, clips, grouped


def audit(root=ROOT):
    protocol, manifest, inventory, contexts = read_public(root)
    base = root / "docs/results"
    selection = json.loads((base / "positive_exposure_selection.json").read_text(encoding="utf-8"))
    results = json.loads((base / "positive_exposure_results.json").read_text(encoding="utf-8"))
    trace_path = base / "positive_exposure_outcomes.csv"
    if digest(trace_path) != results["trace_sha256"]:
        raise ValueError("Published outcomes changed")
    with trace_path.open(encoding="utf-8", newline="") as handle:
        published = list(csv.DictReader(handle))
    traced = {(row["model"], row["policy"], row["split"], row["path"]): row for row in published}
    if len(traced) != len(published) or len(published) != results["trace_rows"]:
        raise ValueError("Duplicate or incomplete outcome trace")
    computed = {}
    checked, curves = 0, 0
    def evaluate(key, cohort, threshold, constant=None, trace_model=None, policy=None):
        nonlocal checked
        names = protocol["cohorts"][protocol["models"][key]["split"]][cohort + "_clip_paths"]
        clips = [inventory[(cohort, key, path)] for path in names]
        values = [clip_outcome(clip, contexts[(cohort, key, clip["path"])], threshold, constant) for clip in clips]
        if policy:
            for clip, value in zip(clips, values):
                row = traced[(trace_model or key, policy, clip["split"], clip["path"])]
                for field in ("source_sha256", "annotation_pts_sec", "valid_contexts", "rejected_contexts"):
                    assert_same(row[field], clip[field])
                assert_same(float(row["threshold"]), threshold)
                for field in ("detected", "trigger_pts_sec", "signed_delay_sec", "early_episodes", "late_episodes"):
                    actual = float(row[field]) if row[field] else None
                    assert_same(actual, value[field])
                checked += 1
        return (*summarize(clips, values), values)
    for key in protocol["models"]:
        candidates = []
        for expected in selection["policies"][key]["curve"]:
            summary, prefix, _ = evaluate(key, "calibration", expected["threshold"])
            assert_same(summary, expected["summary"])
            assert_same(prefix, expected["pre_event"])
            constraints = protocol["constraints"]
            if (prefix["clips_with_support"] >= constraints["minimum_supported_calibration_clips"] and prefix["macro_positive_fraction"] is not None
                    and prefix["macro_positive_fraction"] <= constraints["macro_pre_event_positive_cap"]
                    and summary["off_target_episodes_per_clip"] <= constraints["episode_budget_per_clip"]):
                candidates.append(expected)
            curves += 1
        chosen = max(candidates, key=lambda row: (row["summary"]["event_recall"], -row["summary"]["off_target_episodes_per_clip"], row["threshold"])) if candidates else None
        assert_same(chosen, selection["policies"][key]["selected"])
    for row in results["policies"]:
        summary, prefix, values = evaluate(row["model"], "test", row["threshold"], policy=row["policy"])
        assert_same(summary, row["test"])
        assert_same(prefix, row["pre_event"])
        assert_same(summary["off_target_episodes_per_clip"] > results["constraints"]["episode_budget_per_clip"], row["episode_budget_exceeded"])
        assert_same(prefix["macro_positive_fraction"] > results["constraints"]["macro_pre_event_positive_cap"], row["pre_event_cap_exceeded"])
        computed[(row["model"], row["policy"])] = [value["detected"] for value in values]
    for split, controls in results["controls"].items():
        key = next(key for key, definition in protocol["models"].items() if definition["split"] == split)
        for name, constant in (("clock_only_positive", 1), ("always_negative", 0)):
            summary, prefix, _ = evaluate(key, "test", .5, constant, name, "control")
            assert_same(summary, controls[name]["test"])
            assert_same(prefix, controls[name]["pre_event"])
    for row in results["paired_recall_contrasts"]:
        a, b = (computed[(row["model"], policy)] for policy in ("positive_context", "onset_only"))
        assert_same(paired_cluster_coverage_interval(np.ones(len(a)), np.asarray(a), np.asarray(b)), row["paired_interval"])
    if checked != len(published):
        raise ValueError("Every published outcome must be independently checked")
    return {"calibration_curve_points": curves, "per_clip_outcomes": checked, "paired_intervals": len(results["paired_recall_contrasts"]),
            "inventory_rows": manifest["inventory_rows"], "context_rows": manifest["context_rows"],
            "trace_manifest_sha256": digest(base / "positive_context_trace.json"),
            "auditor_sha256": digest(Path(__file__)), "scope": "Public CSV/JSON only; independent gate, prefix, event and aggregate reconstruction; no video, checkpoint, detector or GPU required"}


if __name__ == "__main__":
    result = audit()
    print(json.dumps(result, indent=2))
    (ROOT / "docs/results/positive_context_audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
