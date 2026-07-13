from __future__ import annotations

from dataclasses import dataclass

import numpy as np


NON_FEATURE_FIELDS = frozenset(
    {
        "path",
        "label",
        "target",
        "collision_type",
        "region",
        "quality",
        "day_time",
        "iid_split",
        "geographic_split",
        "start_frame",
        "end_frame",
        "accident_frame",
        "source_accident_frame",
    }
)


@dataclass(frozen=True)
class StandardizedLogisticModel:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float = 0.5
    feature_names: tuple[str, ...] = ()

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        matrix = _feature_matrix(features)
        standardized = (matrix - self.feature_mean) / self.feature_scale
        return _sigmoid(standardized @ self.weights + self.bias)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return (self.predict_proba(features) >= self.threshold).astype(np.int64)

    def to_dict(self) -> dict:
        return {
            "model_type": "standardized_logistic_regression",
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
            "weights": self.weights.tolist(),
            "bias": self.bias,
            "threshold": self.threshold,
            "feature_names": list(self.feature_names),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "StandardizedLogisticModel":
        return cls(
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float64),
            feature_scale=np.asarray(payload["feature_scale"], dtype=np.float64),
            weights=np.asarray(payload["weights"], dtype=np.float64),
            bias=float(payload["bias"]),
            threshold=float(payload.get("threshold", 0.5)),
            feature_names=tuple(payload.get("feature_names", ())),
        )


def fit_logistic_regression(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int = 600,
    learning_rate: float = 0.05,
    l2: float = 1e-3,
    threshold: float = 0.5,
    feature_names: tuple[str, ...] = (),
) -> StandardizedLogisticModel:
    matrix = _feature_matrix(features)
    labels = np.asarray(targets, dtype=np.float64).reshape(-1)
    if matrix.shape[0] != labels.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    if set(np.unique(labels)) != {0.0, 1.0}:
        raise ValueError("training data must contain both classes")
    if epochs <= 0 or learning_rate <= 0:
        raise ValueError("epochs and learning_rate must be positive")

    feature_mean = np.mean(matrix, axis=0)
    feature_scale = np.std(matrix, axis=0)
    feature_scale = np.where(feature_scale < 1e-9, 1.0, feature_scale)
    standardized = (matrix - feature_mean) / feature_scale

    weights = np.zeros(standardized.shape[1], dtype=np.float64)
    bias = 0.0
    positive_weight = labels.size / (2.0 * np.sum(labels))
    negative_weight = labels.size / (2.0 * np.sum(1.0 - labels))
    sample_weights = np.where(labels == 1.0, positive_weight, negative_weight)

    for _ in range(epochs):
        probabilities = _sigmoid(standardized @ weights + bias)
        errors = (probabilities - labels) * sample_weights
        gradient_weights = standardized.T @ errors / labels.size + l2 * weights
        gradient_bias = float(np.sum(errors) / labels.size)
        weights -= learning_rate * gradient_weights
        bias -= learning_rate * gradient_bias

    return StandardizedLogisticModel(
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        weights=weights,
        bias=bias,
        threshold=threshold,
        feature_names=feature_names,
    )


def binary_classification_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict:
    labels = np.asarray(targets, dtype=np.int64).reshape(-1)
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if labels.shape != scores.shape or labels.size == 0:
        raise ValueError("targets and probabilities must be non-empty and have the same shape")
    predictions = (scores >= threshold).astype(np.int64)
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    precision = _divide(tp, tp + fp)
    recall = _divide(tp, tp + fn)
    return {
        "sample_count": int(labels.size),
        "accuracy": _divide(tp + tn, labels.size),
        "precision": precision,
        "recall": recall,
        "f1": _divide(2.0 * precision * recall, precision + recall),
        "false_positive_rate": _divide(fp, fp + tn),
        "specificity": _divide(tn, tn + fp),
        "threshold": threshold,
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
    }


def select_numeric_feature_names(row: dict[str, str]) -> tuple[str, ...]:
    feature_names: list[str] = []
    for name, value in row.items():
        if name in NON_FEATURE_FIELDS:
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            feature_names.append(name)
    if not feature_names:
        raise ValueError("No numeric feature columns were found")
    return tuple(feature_names)


def _feature_matrix(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("features must be a non-empty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0
