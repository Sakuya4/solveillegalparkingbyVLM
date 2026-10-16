from copy import deepcopy
import csv
import hashlib
import json
from pathlib import Path
import shutil

import pytest

from scripts.audit_positive_contexts import audit, assert_same, clip_outcome, read_public, summarize
from scripts.export_positive_contexts import export_rows

ROOT = Path(__file__).resolve().parents[1]


def clip(name="a", event=6, valid=3):
    return {"path": name, "annotation_pts_sec": event, "collision_type": "rear-end", "valid_contexts": valid, "rejected_contexts": 0}


def contexts(starts=(0, 15, 30), times=(2, 4, 6), scores=(.9, .9, .9)):
    return [dict(start_slot=start, end_pts_sec=time, probability=score) for start, time, score in zip(starts, times, scores)]


def test_independent_gate_has_one_onset_and_resets_at_missing_hops():
    outcome = clip_outcome(clip(), contexts(), .5)
    assert outcome["detected"] == 0 and outcome["early_episodes"] == 1
    assert outcome["prefix_contexts"] == outcome["prefix_positives"] == 2
    outcome = clip_outcome(clip(), contexts((0, 30, 45)), .5)
    assert outcome["detected"] == 1 and outcome["trigger_pts_sec"] == 6
    assert outcome["early_episodes"] == 0


def test_strict_prefix_and_inclusive_event_boundaries():
    outcome = clip_outcome(clip(event=5), contexts(), .5)
    assert outcome["prefix_contexts"] == 1
    assert outcome["detected"] == 1 and outcome["signed_delay_sec"] == -1


def test_macro_support_never_zero_imputes_unsupported_clips():
    clips = [clip(), clip("b"), clip("empty", valid=0)]
    values = [clip_outcome(clips[0], contexts(), .5), clip_outcome(clips[1], contexts((0,), (2,), (.1,)), .5), clip_outcome(clips[2], [], .5)]
    _, prefix = summarize(clips, values)
    assert prefix["clips_with_support"] == 2 and prefix["clips_without_support"] == 1
    assert prefix["macro_positive_fraction"] == .5
    assert prefix["pooled_positive_fraction"] == 2 / 3
    _, empty = summarize([clips[2]], [values[2]])
    assert empty["macro_positive_fraction"] is None and empty["pooled_positive_fraction"] is None


@pytest.mark.parametrize("rows", [contexts((0, 0)), contexts((1, 15)), contexts(times=(2, 1, 6)), contexts(scores=(.5, float("nan"), .5)), contexts(times=(-.1, 4, 6))])
def test_independent_reader_rejects_invalid_public_contexts(rows):
    with pytest.raises(ValueError):
        clip_outcome(clip(), rows, .5)


def test_export_preserves_empty_clips_without_inventing_contexts():
    protocol = {"models": {"m": {"split": "iid"}}, "cohorts": {"iid": {"test_clip_paths": ["a"]}},
                "annotations": {"a": {"event_time_sec": 6, "collision_type": "rear-end"}}}
    report = {"path": "a", "source_sha256": "source", "window_starts": [], "window_end_times_sec": [], "scores": {"m": []}, "rejected_window_starts": [0]}
    inventory, rows = export_rows(protocol, [report], "test")
    assert len(inventory) == 1 and inventory[0]["valid_contexts"] == 0 and rows == []
    report["scores"]["m"] = [.1]
    with pytest.raises(ValueError):
        export_rows(protocol, [report], "test")


def test_public_scores_reproduce_all_curves_outcomes_and_intervals():
    result = audit(ROOT)
    assert result["calibration_curve_points"] == 816
    assert result["per_clip_outcomes"] == 2000 and result["paired_intervals"] == 8
    assert result["inventory_rows"] == 1508 and result["context_rows"] == 13060


def test_frozen_comparison_keeps_failures_and_original_defaults():
    study = json.loads((ROOT / "docs/results/positive_exposure_results.json").read_text())
    policies = [row for row in study["policies"] if row["policy"] == "positive_context"]
    assert len(policies) == 8 and sum(row["pre_event_cap_exceeded"] for row in policies) == 2
    assert sum(row["episode_budget_exceeded"] for row in policies) == 1
    assert all(row["paired_interval"]["upper_95"] < 0 for row in study["paired_recall_contrasts"])
    changed = deepcopy(study["policies"][0]["pre_event"])
    changed["clips_without_support"] = 0
    with pytest.raises(ValueError):
        assert_same(changed, study["policies"][0]["pre_event"])


@pytest.mark.parametrize("mutation", ["drop_clip", "duplicate_clip", "orphan_context", "changed_source", "changed_anchor", "drop_context"])
def test_public_inventory_rejects_structural_corruption_even_with_rehashed_csv(tmp_path, mutation):
    base = tmp_path / "docs/results"
    base.mkdir(parents=True)
    for name in ("positive_context_trace.json", "positive_exposure_protocol.json", "positive_exposure_selection.json", "positive_exposure_results.json", "positive_context_inventory.csv", "positive_context_scores.csv"):
        shutil.copyfile(ROOT / "docs/results" / name, base / name)
    key = "inventory" if mutation in {"drop_clip", "duplicate_clip", "changed_source", "changed_anchor"} else "contexts"
    manifest_path = base / "positive_context_trace.json"
    manifest = json.loads(manifest_path.read_text())
    target = tmp_path / manifest["files"][key]["path"]
    with target.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    fields = list(rows[0])
    if mutation in {"drop_clip", "drop_context"}:
        rows.pop()
    elif mutation == "duplicate_clip":
        rows.append(rows[0])
    elif mutation == "orphan_context":
        rows[0]["path"] = "not-in-frozen-cohort"
    elif mutation == "changed_source":
        rows[0]["source_sha256"] = "0" * 64
    elif mutation == "changed_anchor":
        rows[0]["annotation_pts_sec"] = "1000"
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    manifest["files"][key]["sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    manifest["inventory_rows" if key == "inventory" else "context_rows"] = len(rows)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        read_public(tmp_path)
