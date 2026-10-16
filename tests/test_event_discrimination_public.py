import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.analyze_event_discrimination import analyze, load_inputs
from illegal_parking.incident_presentation_events import evaluate_presentation_event

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "docs/results"


def pairwise_auc(labels, scores):
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    return float(np.mean((positives[:, None] > negatives) + .5 * (positives[:, None] == negatives)))


def tied_average_precision(labels, scores):
    # Threshold groups, rather than arbitrary tie ordering, define AP increments.
    total = labels.sum()
    previous_tp, ap = 0, 0.
    for threshold in sorted(set(scores), reverse=True):
        selected = scores >= threshold
        tp = int(labels[selected].sum())
        ap += (tp - previous_tp) / total * tp / selected.sum()
        previous_tp = tp
    return ap


def test_published_diagnostics_reproduce_with_public_calibration_inputs_only():
    protocol = json.loads((BASE / "event_discrimination_protocol.json").read_text())
    result = json.loads((BASE / "event_discrimination_results.json").read_text())
    rows, _, pairs = analyze(ROOT, protocol)
    assert result["models"] == rows and result["paired_ranking_contrasts"] == pairs
    assert result["trace_rows"] == 708
    assert hashlib.sha256((BASE / "event_discrimination_outcomes.csv").read_bytes()).hexdigest() == result["trace_sha256"]
    execution = json.loads((BASE / "event_discrimination_execution.json").read_text())
    assert datetime.fromisoformat(protocol["created_at_utc"]) < datetime.fromisoformat(execution["created_at_utc"]) < datetime.fromisoformat(result["started_at_utc"])
    for path, digest in execution["implementation_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest


def test_independent_rank_formulas_and_production_gate_agree_for_every_public_clip():
    protocol = json.loads((BASE / "event_discrimination_protocol.json").read_text())
    inventory, contexts = load_inputs(ROOT, protocol)
    with (BASE / "event_discrimination_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        traces = list(csv.DictReader(handle))
    assert len({(row["model"], row["path"]) for row in traces}) == len(traces) == 708
    for row in traces:
        identity = (row["model"], row["path"])
        clip = inventory[identity]
        entries = contexts[identity]
        times = np.asarray([float(entry["end_pts_sec"]) for entry in entries])
        scores = np.asarray([float(entry["probability"]) for entry in entries])
        event = float(clip["annotation_pts_sec"])
        keep = times <= event + 3
        labels = (times[keep] >= event - 1).astype(int)
        if len(np.unique(labels)) == 2:
            assert float(row["auroc"]) == pytest.approx(pairwise_auc(labels, scores[keep]), abs=1e-12)
            assert float(row["average_precision"]) == pytest.approx(tied_average_precision(labels, scores[keep]), abs=1e-12)
        else:
            assert row["auroc"] == row["average_precision"] == ""
        report = {"path": clip["path"], "clock": "normalized_presentation_time", "annotation_free_inference": True,
                  "window_starts": [int(entry["start_slot"]) for entry in entries], "window_end_times_sec": times.tolist(), "window_probabilities": scores.tolist()}
        annotation = {"path": clip["path"], "event_time_sec": event, "collision_type": clip["collision_type"]}
        for policy, threshold in protocol["thresholds"][row["model"]].items():
            outcome = evaluate_presentation_event(report, annotation, threshold)
            assert outcome.detected == (row[policy + "_category"] == "detected")
            possible = [index for index in range(1, len(entries)) if report["window_starts"][index] - report["window_starts"][index - 1] == 15 and event - 1 <= times[index] <= event + 3]
            positive_pairs = [index for index in possible if min(scores[index - 1], scores[index]) >= threshold]
            expected = ("detected" if outcome.detected else "structurally_unreachable" if not possible else
                        "no_target_positive_pair" if not positive_pairs else "onset_suppression")
            assert row[policy + "_category"] == expected


def test_diagnostic_categories_reconcile_to_prior_training_selections_not_test_results():
    result = json.loads((BASE / "event_discrimination_results.json").read_text())
    selection = json.loads((BASE / "positive_exposure_selection.json").read_text())
    for row in result["models"]:
        summary = row["summary"]
        capped = summary["policy_decomposition"]["positive_context"]
        assert capped["detected"] == selection["policies"][row["model"]]["selected"]["summary"]["detected_clips"]
        for policy in summary["policy_decomposition"].values():
            assert sum(policy.values()) == summary["clip_count"]
        assert summary["gate2_attainable_clips"] <= summary["gate1_attainable_clips"] <= summary["clip_count"]
        assert summary["constant_control"]["average_precision"] == summary["pooled"]["target_prevalence"]
    assert all(row["interval"]["lower_95"] < 0 < row["interval"]["upper_95"] for row in result["paired_ranking_contrasts"])


def test_public_clip_trace_reconstructs_macro_rankings_and_paired_intervals():
    result = json.loads((BASE / "event_discrimination_results.json").read_text())
    with (BASE / "event_discrimination_outcomes.csv").open(newline="", encoding="utf-8") as handle:
        traces = list(csv.DictReader(handle))
    supported = {}
    for model in result["models"]:
        rows = [row for row in traces if row["model"] == model["model"] and row["support"] == "both"]
        supported[model["model"]] = {row["path"]: row for row in rows}
        for metric, expected in model["summary"]["within_clip_macro"].items():
            assert np.mean([float(row[metric]) for row in rows]) == pytest.approx(expected, abs=1e-12)
    for pair in result["paired_ranking_contrasts"]:
        a = supported[f'fixed_{pair["feature"]}_{pair["split"]}']
        b = supported[f'legacy_{pair["feature"]}_{pair["split"]}']
        assert set(a) == set(b)
        differences = np.asarray([float(a[path][pair["metric"]]) - float(b[path][pair["metric"]]) for path in sorted(a)])
        rng = np.random.default_rng(42)
        estimates = [differences[rng.integers(0, len(differences), len(differences))].mean() for _ in range(2000)]
        interval = pair["interval"]
        assert interval["difference"] == pytest.approx(differences.mean(), abs=1e-12)
        assert [interval["lower_95"], interval["upper_95"]] == pytest.approx(np.percentile(estimates, [2.5, 97.5]), abs=1e-12)
