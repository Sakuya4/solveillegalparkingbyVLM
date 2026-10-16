from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any

from .accidents import AccidentRecord
from .deployment_coverage import haversine_m


def aggregate_risk_cells(
    records: list[AccidentRecord], *, a1_weight: float = 5.0, precision: int = 3
) -> list[dict[str, Any]]:
    if not math.isfinite(a1_weight) or a1_weight <= 0:
        raise ValueError("a1_weight must be positive and finite")
    grouped: dict[str, list[AccidentRecord]] = defaultdict(list)
    for record in records:
        lat, lon = record.latitude, record.longitude
        if record.category not in {"A1", "A2"} or lat is None or lon is None:
            continue
        if not (math.isfinite(lat) and math.isfinite(lon) and 21 <= lat <= 26.5 and 118 <= lon <= 123):
            continue
        grouped[f"{lat:.{precision}f},{lon:.{precision}f}"].append(record)
    cells = []
    for grid_id, group in sorted(grouped.items()):
        # Fixed cell centers keep train and holdout geometry identical.
        lat, lon = map(float, grid_id.split(","))
        a1 = sum(record.category == "A1" for record in group)
        a2 = len(group) - a1
        cells.append({
            "id": grid_id, "latitude": lat, "longitude": lon,
            "a1_count": a1, "a2_count": a2, "crash_count": len(group),
            "risk_weight": a1 * a1_weight + a2,
        })
    return cells


def coverage_index(
    camera_sites: list[dict[str, Any]], cells: list[dict[str, Any]], radius_m: float
) -> dict[str, frozenset[str]]:
    if not math.isfinite(radius_m) or radius_m <= 0:
        raise ValueError("radius_m must be positive and finite")
    identifiers = [str(site["id"]) for site in camera_sites]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("camera identifiers must be unique")
    return {
        str(site["id"]): frozenset(
            cell["id"] for cell in cells
            if haversine_m(site["latitude"], site["longitude"], cell["latitude"], cell["longitude"]) <= radius_m
        )
        for site in camera_sites
    }


def select_camera_budget(
    index: dict[str, frozenset[str]], weights: dict[str, float], budget: int,
    *, method: str = "weighted_greedy", seed: int = 42,
) -> list[str]:
    if budget < 0 or budget > len(index):
        raise ValueError("budget must be between zero and the camera count")
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("weights must be nonnegative and finite")
    candidates = sorted(index)
    if method == "random":
        return random.Random(seed).sample(candidates, budget)
    if method == "independent_rank":
        return sorted(candidates, key=lambda key: (-sum(weights.get(cell, 0.0) for cell in index[key]), key))[:budget]
    if method != "weighted_greedy":
        raise ValueError(f"Unknown selection method: {method}")
    selected: list[str] = []
    covered: set[str] = set()
    for _ in range(budget):
        best = min(candidates, key=lambda key: (-sum(weights.get(cell, 0.0) for cell in index[key] - covered), key))
        selected.append(best)
        covered.update(index[best])
        candidates.remove(best)
    return selected


def evaluate_camera_selection(
    selected: list[str], index: dict[str, frozenset[str]], cells: list[dict[str, Any]]
) -> dict[str, float | int]:
    covered = set().union(*(index[key] for key in selected)) if selected else set()
    total_weight = sum(cell["risk_weight"] for cell in cells)
    covered_weight = sum(cell["risk_weight"] for cell in cells if cell["id"] in covered)
    total_crashes = sum(cell["crash_count"] for cell in cells)
    covered_crashes = sum(cell["crash_count"] for cell in cells if cell["id"] in covered)
    return {
        "selected_cameras": len(selected), "covered_cells": len(covered),
        "covered_risk": covered_weight, "total_risk": total_weight,
        "risk_coverage": covered_weight / total_weight if total_weight else 0.0,
        "covered_crashes": covered_crashes, "total_crashes": total_crashes,
        "crash_coverage": covered_crashes / total_crashes if total_crashes else 0.0,
    }
