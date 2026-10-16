import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

from illegal_parking.research_statistics import paired_cluster_coverage_interval

RESULTS = Path(__file__).resolve().parents[1] / "docs/results"


def load(name):
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def trace_groups():
    with (RESULTS / "presentation_event_outcomes.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[(row["cohort"], row["split"], row["method"])].append(row)
    return rows, groups


def test_every_published_endpoint_recomputes_from_public_per_clip_records():
    rows, groups = trace_groups()
    protocol = load("presentation_event_protocol.json")
    selection = load("presentation_event_selection.json")
    comparison = load("presentation_event_comparison.json")
    controls = load("presentation_event_controls.json")
    test_models = {row["key"]: row for row in comparison["models"]}
    assert len(rows) == 2130 and len(groups) == 24
    for (cohort, split, method), records in groups.items():
        names = protocol["cohorts"][split][cohort + "_clip_paths"]
        assert len(records) == len(names) and sorted(row["path"] for row in records) == sorted(names)
        if method in protocol["models"]:
            expected = selection["policies"][method]["selected"]["summary"] if cohort == "calibration" else test_models[method]["test"]
        else:
            expected = controls["controls"][split][cohort][method]
        assert sum(int(row["detected"]) for row in records) / len(records) == pytest.approx(expected["event_recall"])
        assert sum(int(row["early_episodes"]) + int(row["late_episodes"]) for row in records) / len(records) == pytest.approx(expected["off_target_episodes_per_clip"])
        assert sum(int(row["rejected_contexts"]) for row in records) == expected["rejected_windows"]
        delays = [float(row["signed_delay_sec"]) for row in records if row["signed_delay_sec"]]
        if delays:
            assert float(np.median(delays)) == pytest.approx(expected["trigger_delay_sec"]["median"])
        else:
            assert expected["trigger_delay_sec"]["median"] is None
        for row in records:
            assert bool(row["signed_delay_sec"]) == bool(int(row["detected"]))
            if row["signed_delay_sec"]:
                assert float(row["trigger_pts_sec"]) - float(row["annotation_pts_sec"]) == pytest.approx(float(row["signed_delay_sec"]))


def test_public_selection_and_trace_hashes_match_the_frozen_study():
    def digest(name):
        return hashlib.sha256((RESULTS / name).read_bytes()).hexdigest()

    manifest = load("presentation_event_outcomes.json")
    comparison = load("presentation_event_comparison.json")
    controls = load("presentation_event_controls.json")
    assert manifest["outcomes_sha256"] == digest("presentation_event_outcomes.csv")
    assert manifest["prediction_audit_sha256"] == digest("presentation_event_controls.json")
    assert manifest["protocol_sha256"] == digest("presentation_event_protocol.json")
    assert manifest["selection_sha256"] == comparison["selection_sha256"] == digest("presentation_event_selection.json")
    assert controls["prediction_verification"]["calibration"]["maximum_absolute_error"] <= 1e-7
    assert controls["prediction_verification"]["test"]["maximum_absolute_error"] <= 1e-7


def test_all_paired_event_intervals_reproduce_from_public_trace_rows():
    _, groups = trace_groups()
    protocol = load("presentation_event_protocol.json")
    comparisons = load("presentation_event_comparison.json")
    controls = load("presentation_event_controls.json")

    def scores(split, method):
        indexed = {row["path"]: float(row["detected"]) for row in groups[("test", split, method)]}
        return np.asarray([indexed[name] for name in protocol["cohorts"][split]["test_clip_paths"]])

    for row in comparisons["paired_recall_contrasts"]:
        a, b = scores(row["split"], f'fixed_{row["feature"]}_{row["split"]}'), scores(row["split"], f'legacy_{row["feature"]}_{row["split"]}')
        assert paired_cluster_coverage_interval(np.ones(len(a)), a, b) == row["paired_interval"]
    for row in controls["model_comparisons"]:
        split = protocol["models"][row["model"]]["split"]
        a, b = scores(split, row["model"]), scores(split, "clock_only_positive")
        assert paired_cluster_coverage_interval(np.ones(len(a)), a, b) == row["paired_recall_vs_clock"]
