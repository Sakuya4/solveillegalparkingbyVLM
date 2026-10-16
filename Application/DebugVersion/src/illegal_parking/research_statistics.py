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


def paired_cluster_f1_interval(
    targets: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray,
    clip_ids: np.ndarray, *, threshold_a: float, threshold_b: float,
    repeats: int = 2000, seed: int = 42,
) -> dict[str, float | int]:
    labels = np.asarray(targets).reshape(-1)
    a = np.asarray(scores_a, dtype=float).reshape(-1)
    b = np.asarray(scores_b, dtype=float).reshape(-1)
    clips = np.asarray(clip_ids).reshape(-1)
    if not len(labels) or any(len(array) != len(labels) for array in (a, b, clips)) or not np.all(np.isin(labels, [0, 1])):
        raise ValueError("Aligned nonempty binary targets, scores, and clip identifiers are required")
    if not np.all(np.isfinite([*a, *b, threshold_a, threshold_b])) or np.any((a < 0) | (a > 1)) or np.any((b < 0) | (b > 1)) or not 0 <= threshold_a <= 1 or not 0 <= threshold_b <= 1:
        raise ValueError("Scores and thresholds must be finite and in [0, 1]")
    unique, inverse = np.unique(clips, return_inverse=True)
    if len(unique) < 2 or repeats <= 0:
        raise ValueError("At least two source clips and positive repeats are required")
    columns = []
    for predicted in (a >= threshold_a, b >= threshold_b):
        columns.extend(np.bincount(inverse, weights=condition, minlength=len(unique)) for condition in (predicted & (labels == 1), predicted & (labels == 0), ~predicted & (labels == 1)))
    per_clip = np.column_stack(columns)

    def difference(counts):
        result = []
        for offset in (0, 3):
            tp, fp, fn = counts[offset:offset + 3]
            result.append(2 * tp / max(1.0, 2 * tp + fp + fn))
        return result[0] - result[1]

    generator = np.random.default_rng(seed)
    samples = [difference(per_clip[generator.integers(0, len(unique), len(unique))].sum(axis=0)) for _ in range(repeats)]
    lower, upper = np.quantile(samples, [0.025, 0.975])
    return {"difference": float(difference(per_clip.sum(axis=0))), "lower_95": float(lower), "upper_95": float(upper), "clusters": len(unique), "bootstrap_repeats": repeats, "seed": seed}
