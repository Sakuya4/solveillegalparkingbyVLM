import copy

import numpy as np
import pytest

from illegal_parking.incident_training_support import endpoint_label, fit_source_records, audit_training_source


def source():
    return {"path": "fit.mp4", "supplied_event_frame": 90, "event_time_sec": 6,
            "anchored_windows": [{"label": "incident", "indices": list(range(60, 91, 2)), "end_time_sec": 6}]}


@pytest.mark.parametrize("end,label", [(4.999, "prefix"), (5, "target"), (9, "target"), (9.001, "post")])
def test_endpoint_labels_use_frozen_inclusive_target_boundaries(end, label):
    assert endpoint_label(end, 6) == label


@pytest.mark.parametrize("end", [-1, float("nan"), float("inf")])
def test_invalid_endpoint_cannot_enter_training_labels(end):
    with pytest.raises(ValueError):
        endpoint_label(end, 6)


def test_complete_fit_timeline_counts_censored_post_context_and_gate_support():
    summary, contexts = audit_training_source(source(), np.arange(301) / 15)
    assert summary["valid_contexts"] == 10
    assert (summary["prefix"], summary["target"], summary["post"]) == (2, 2, 6)
    assert summary["both_classes"] and summary["gate2_structural_support"]
    assert summary["two_target_contexts_adjacent"]
    assert summary["anchored_exact_continuous_matches"] == 1
    assert len(contexts) == 10 and contexts[0]["start"] == 0
    assert summary["anchored_endpoint_label_disagreements"] == 0


def test_empty_and_single_context_fit_sources_are_retained():
    item = source()
    item.update(supplied_event_frame=5, event_time_sec=1 / 3)
    summary, contexts = audit_training_source(item, np.arange(10) / 15)
    assert not contexts and summary["valid_contexts"] == 0
    assert not summary["gate1_support"] and not summary["both_classes"]
    item.update(supplied_event_frame=15, event_time_sec=1)
    summary, _ = audit_training_source(item, np.arange(31) / 15)
    assert summary["gate1_support"] and not summary["gate2_structural_support"]


def test_missing_hops_do_not_create_adjacent_target_support():
    times = np.arange(301) / 15
    times[65:] += 1
    item = source()
    item.update(event_time_sec=7)
    summary, contexts = audit_training_source(item, times)
    starts = {row["start"] for row in contexts}
    assert summary["invalid_contexts"] > 0
    assert not {30, 45} <= starts


def test_source_selection_uses_each_split_fit_membership_and_rejects_leakage():
    row = {"path": "a", "supplied_event_frame": 1, "event_time_sec": 1,
           "label": "incident", "fixed_indices": list(range(16)), "fixed_times_sec": list(range(16)),
           "iid_partition": "fit", "geographic_partition": "test"}
    other = {**row, "path": "b", "iid_partition": "calibration"}
    records = fit_source_records([row, other], [{"path": "a", "sha256": "a"}])
    assert len(records) == 1 and records[0]["fit_splits"] == ["iid"]
    bad = copy.deepcopy(row)
    bad["iid_partition"] = "test"
    with pytest.raises(ValueError, match="cross partitions"):
        fit_source_records([row, bad], [{"path": "a"}])


def test_changed_supplied_annotation_time_fails_instead_of_clamping():
    item = source()
    item["event_time_sec"] = 5
    with pytest.raises(ValueError, match="Annotation presentation"):
        audit_training_source(item, np.arange(301) / 15)
