from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np


def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    if trials <= 0 or not 0 <= successes <= trials or not 0 < confidence < 1:
        raise ValueError("Valid successes, positive trials, and confidence in (0, 1) are required")
    z = NormalDist().inv_cdf((1 + confidence) / 2)
    probability = successes / trials
    denominator = 1 + z * z / trials
    center = (probability + z * z / (2 * trials)) / denominator
    radius = z * math.sqrt(probability * (1 - probability) / trials + z * z / (4 * trials * trials)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def paired_cluster_coverage_interval(
    cluster_totals: np.ndarray, cluster_covered_a: np.ndarray,
    cluster_covered_b: np.ndarray, *, repeats: int = 2000, seed: int = 42,
) -> dict[str, float | int]:
    values = np.column_stack((cluster_totals, cluster_covered_a, cluster_covered_b)).astype(float)
    if len(values) < 2 or repeats <= 0 or not np.all(np.isfinite(values)):
        raise ValueError("At least two finite clusters and positive repeats are required")
    if np.any(values < 0) or np.any(values[:, 1:] > values[:, :1]) or values[:, 0].sum() <= 0:
        raise ValueError("Covered weights must lie between zero and total cluster weight")
    if np.any(values[:, 0] <= 0):
        raise ValueError("Each cluster must contain positive total weight")
    generator = np.random.default_rng(seed)
    differences = []
    for _ in range(repeats):
        sampled = values[generator.integers(0, len(values), len(values))].sum(axis=0)
        differences.append((sampled[1] - sampled[2]) / sampled[0])
    lower, upper = np.quantile(differences, [0.025, 0.975])
    totals = values.sum(axis=0)
    return {"difference": float((totals[1] - totals[2]) / totals[0]), "lower_95": float(lower), "upper_95": float(upper), "clusters": len(values), "bootstrap_repeats": repeats, "seed": seed}
