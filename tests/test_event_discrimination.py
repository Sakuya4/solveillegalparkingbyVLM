import csv
import hashlib

import numpy as np
import pytest

from illegal_parking.incident_event_discrimination import clip_diagnostic, model_diagnostic, paired_ranking_interval, ranking_metrics
from scripts.prepare_event_discrimination import calibration_rows
from scripts.analyze_event_discrimination import load_inputs


def clip(name="a", event=6, valid=4):
    return {"path": name, "source_sha256": "source", "annotation_pts_sec": event, "valid_contexts": valid, "rejected_contexts": 0}


def contexts(scores=(.1, .2, .9, .8), starts=(0, 15, 30, 45), times=(2, 4, 6, 8)):
    return [dict(start_slot=start, end_pts_sec=time, probability=score) for start, time, score in zip(starts, times, scores)]


def test_boundary_labels_use_strict_prefix_and_inclusive_target_band():
    result = clip_diagnostic(clip(event=5), contexts(times=(2, 4, 8, 10)), {"p": .5})
    assert result["contexts"] == {"valid": 4, "prefix": 1, "target": 2, "post": 1, "rejected": 0}
    assert result["support"] == "both"


def test_perfect_reversed_and_constant_rankings_keep_orientation_and_ties():
    assert ranking_metrics([0, 0, 1, 1], [.1, .2, .9, .8])["auroc"] == 1
    assert ranking_metrics([0, 0, 1, 1], [.9, .8, .1, .2])["auroc"] == 0
    constant = ranking_metrics([0, 0, 0, 1], [.5] * 4)
    assert constant == {"auroc": .5, "average_precision": .25, "target_prevalence": .25, "ap_minus_prevalence": 0}
    assert ranking_metrics([], [])["auroc"] is None
    assert ranking_metrics([1], [.5])["average_precision"] is None


def test_miss_categories_separate_structural_pair_and_sustained_onset_failures():
    positive = clip_diagnostic(clip(), contexts(scores=(.9,) * 4), {"p": .5})
    assert positive["policies"]["p"]["category"] == "onset_suppression"
    assert positive["gate2_attainable"]
    missing_pair = clip_diagnostic(clip(), contexts(scores=(.1, .1, .9, .1)), {"p": .5})
    assert missing_pair["policies"]["p"]["category"] == "no_target_positive_pair"
    detected = clip_diagnostic(clip(), contexts(), {"p": .5})
    assert detected["policies"]["p"]["category"] == "detected"
    assert detected["policies"]["p"]["trigger_pts_sec"] == 8
    unreachable = clip_diagnostic(clip(valid=1), contexts((.9,), (0,), (6,)), {"p": .5})
    assert unreachable["gate1_attainable"] and not unreachable["gate2_attainable"]
    assert unreachable["policies"]["p"]["category"] == "structurally_unreachable"


def test_gap_resets_prevent_false_pair_support():
    row = clip_diagnostic(clip(valid=2), contexts((.9, .9), (0, 30), (2, 6)), {"p": .5})
    assert not row["gate2_attainable"] and row["policies"]["p"]["category"] == "structurally_unreachable"


def test_unsupported_and_empty_clips_remain_in_denominator_not_macro_rankings():
    clips = [clip(), clip("empty", valid=0), clip("no_prefix", event=1)]
    summary, rows = model_diagnostic(clips, {"a": contexts(), "empty": [], "no_prefix": contexts()}, {"p": .5})
    assert summary["clip_count"] == 3 and summary["empty_context_clips"] == 1
    assert summary["support"] == {"both": 1, "no_prefix": 1, "no_target": 0, "neither": 1}
    assert summary["within_clip_macro"]["auroc"] == 1
    assert rows[1]["within_clip"] is None
    assert sum(summary["policy_decomposition"]["p"].values()) == 3


def test_pairing_uses_paths_not_array_order_and_refuses_support_changes():
    a = clip_diagnostic(clip(), contexts(), {"p": .5})
    b = {**a, "path": "b"}
    interval = paired_ranking_interval([a, b], [b, a], "auroc")
    assert interval == {"clips": 2, "difference": 0, "lower_95": 0, "upper_95": 0}
    with pytest.raises(ValueError):
        paired_ranking_interval([a, b], [a], "auroc")


@pytest.mark.parametrize("rows", [contexts(starts=(0, 0, 30, 45)), contexts(times=(2, 1, 6, 8)), contexts(scores=(.1, np.nan, .8, .9))])
def test_invalid_time_grid_or_score_stops_diagnostics(rows):
    with pytest.raises(ValueError):
        clip_diagnostic(clip(), rows, {"p": .5})


def test_freezer_excludes_test_rows_before_scores_are_parsed():
    rows = [{"cohort": "test", "model": "unknown", "path": "b", "probability": "must-not-parse"},
            {"cohort": "calibration", "model": "m", "path": "a", "probability": ".8"}]
    assert calibration_rows(rows, {"m": ["a"]}) == rows[1:]


def test_reader_rejects_test_rows_even_when_their_input_hash_is_updated(tmp_path):
    row = {"cohort": "test", "model": "m", "path": "a", "split": "iid"}
    path = tmp_path / "inventory.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    protocol = {"inputs": {"inventory": {"path": path.name, "rows": 1, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}},
                "model_clip_paths": {"m": ["a"]}, "models": {"m": {"split": "iid"}}}
    with pytest.raises(ValueError, match="Test or unassigned"):
        load_inputs(tmp_path, protocol)
