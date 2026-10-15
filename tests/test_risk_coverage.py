from dataclasses import replace

import pytest

from illegal_parking.accidents import AccidentRecord
from illegal_parking.risk_coverage import (
    aggregate_risk_cells, coverage_index, evaluate_camera_selection, select_camera_budget,
)


def test_greedy_avoids_redundant_camera_coverage_at_equal_budget():
    index = {"a": frozenset({"x", "y"}), "b": frozenset({"x", "y"}), "c": frozenset({"z"})}
    weights = {"x": 4.0, "y": 3.0, "z": 2.0}
    assert select_camera_budget(index, weights, 2) == ["a", "c"]
    assert select_camera_budget(index, weights, 2, method="independent_rank") == ["a", "b"]
    assert len(select_camera_budget(index, {}, 3)) == 3


def test_selection_uses_training_weights_and_evaluates_holdout_separately():
    index = {"a": frozenset({"train"}), "b": frozenset({"test"})}
    selected = select_camera_budget(index, {"train": 10, "test": 1}, 1)
    cells = [{"id": "train", "risk_weight": 2, "crash_count": 2}, {"id": "test", "risk_weight": 8, "crash_count": 4}]
    result = evaluate_camera_selection(selected, index, cells)
    assert selected == ["a"]
    assert result["risk_coverage"] == pytest.approx(0.2)
    assert result["crash_coverage"] == pytest.approx(1 / 3)


def test_aggregation_excludes_invalid_coordinates_and_counts_severity():
    record = AccidentRecord("1", "A1", "20240101", "120000", "city", "area", "road", 12, "", 121.5001, 25.0001)
    records = [record, replace(record, category="A2"), replace(record, latitude=float("nan")), replace(record, category="")]
    cells = aggregate_risk_cells(records, a1_weight=5)
    assert len(cells) == 1
    assert cells[0]["risk_weight"] == 6
    assert cells[0]["crash_count"] == 2
    assert cells[0]["latitude"] == 25.0


def test_random_baseline_is_reproducible_and_budget_checks_are_strict():
    index = {str(i): frozenset() for i in range(10)}
    assert select_camera_budget(index, {}, 3, method="random", seed=3) == select_camera_budget(index, {}, 3, method="random", seed=3)
    with pytest.raises(ValueError, match="budget"):
        select_camera_budget(index, {}, 11)
    with pytest.raises(ValueError, match="weights"):
        select_camera_budget(index, {"bad": -1}, 1)


def test_coverage_index_uses_distance_and_rejects_duplicate_sites():
    sites = [{"id": "a", "latitude": 25.0, "longitude": 121.5}]
    cells = [{"id": "near", "latitude": 25.001, "longitude": 121.5}, {"id": "far", "latitude": 25.02, "longitude": 121.5}]
    assert coverage_index(sites, cells, 500)["a"] == frozenset({"near"})
    with pytest.raises(ValueError, match="unique"):
        coverage_index(sites * 2, cells, 500)
