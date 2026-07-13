from __future__ import annotations

import numpy as np
import pytest

from illegal_parking.incident_baseline import (
    binary_classification_metrics,
    fit_logistic_regression,
    select_numeric_feature_names,
)


def test_logistic_regression_learns_separable_motion_features() -> None:
    features = np.asarray(
        [
            [-2.0, -1.0],
            [-1.5, -0.5],
            [-1.0, -2.0],
            [1.0, 1.0],
            [1.5, 0.5],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    targets = np.asarray([0, 0, 0, 1, 1, 1], dtype=np.float64)

    model = fit_logistic_regression(features, targets, epochs=800, learning_rate=0.1)
    predictions = model.predict(features)

    assert predictions.tolist() == targets.astype(int).tolist()
    assert model.predict_proba(features[:1])[0] < 0.5
    assert model.predict_proba(features[-1:])[0] > 0.5


def test_logistic_model_serialization_preserves_predictions() -> None:
    features = np.asarray([[-1.0], [1.0]], dtype=np.float64)
    targets = np.asarray([0, 1], dtype=np.float64)
    model = fit_logistic_regression(features, targets, epochs=400, learning_rate=0.1)

    restored = type(model).from_dict(model.to_dict())

    assert restored.predict_proba(features) == pytest.approx(model.predict_proba(features))


def test_binary_classification_metrics_reports_false_alarm_rate() -> None:
    targets = np.asarray([0, 0, 1, 1], dtype=np.int64)
    probabilities = np.asarray([0.1, 0.8, 0.7, 0.4], dtype=np.float64)

    metrics = binary_classification_metrics(targets, probabilities, threshold=0.5)

    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["precision"] == pytest.approx(0.5)
    assert metrics["recall"] == pytest.approx(0.5)
    assert metrics["false_positive_rate"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 1}


def test_fit_logistic_regression_rejects_single_class_training_data() -> None:
    with pytest.raises(ValueError, match="both classes"):
        fit_logistic_regression(np.ones((3, 2)), np.ones(3))


def test_select_numeric_features_keeps_ttc_without_label_leakage() -> None:
    row = {
        "path": "clip.mp4",
        "target": "1",
        "start_frame": "20",
        "collision_type": "t-bone",
        "min_ttc_sec_min": "0.25",
        "risk_peak_position": "0.8",
    }

    assert select_numeric_feature_names(row) == (
        "min_ttc_sec_min",
        "risk_peak_position",
    )
