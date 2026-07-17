from __future__ import annotations

import numpy as np
import pytest

from illegal_parking.incident_baseline import (
    binary_classification_metrics,
    filter_feature_names,
    fit_logistic_regression,
    grouped_calibration_indices,
    has_oracle_roi_features,
    select_threshold_for_target_fpr,
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


def test_filter_feature_names_supports_deployment_safe_global_motion() -> None:
    names = ("global_flow_mean", "roi_flow_mean", "trajectory_min_ttc")

    assert filter_feature_names(names, ("global_", "trajectory_")) == (
        "global_flow_mean",
        "trajectory_min_ttc",
    )


def test_filter_feature_names_rejects_empty_selection() -> None:
    with pytest.raises(ValueError, match="No feature columns"):
        filter_feature_names(("roi_flow_mean",), ("global_",))


@pytest.mark.parametrize(
    "name",
    ["roi_flow_mean", "motion_roi_flow_mean", "motion_peak_position", "motion_motion_peak_position"],
)
def test_oracle_roi_feature_detection(name: str) -> None:
    assert has_oracle_roi_features((name,)) is True


def test_trajectory_risk_peak_is_not_mistaken_for_oracle_roi() -> None:
    assert has_oracle_roi_features(("trajectory_risk_peak_position",)) is False


def test_threshold_calibration_maximizes_recall_within_train_fpr_budget() -> None:
    targets = np.asarray([0, 0, 0, 0, 1, 1, 1])
    probabilities = np.asarray([0.10, 0.20, 0.70, 0.80, 0.25, 0.75, 0.90])

    threshold = select_threshold_for_target_fpr(targets, probabilities, target_fpr=0.25)
    metrics = binary_classification_metrics(targets, probabilities, threshold)

    assert threshold == pytest.approx(0.75)
    assert metrics["false_positive_rate"] <= 0.25
    assert metrics["recall"] == pytest.approx(2 / 3)


def test_zero_fpr_calibration_can_select_threshold_above_observed_scores() -> None:
    targets = np.asarray([0, 0, 1])
    probabilities = np.asarray([0.9, 1.0, 0.95])

    threshold = select_threshold_for_target_fpr(targets, probabilities, target_fpr=0.0)

    assert threshold > 1.0
    assert binary_classification_metrics(targets, probabilities, threshold)["false_positive_rate"] == 0.0


@pytest.mark.parametrize("target_fpr", [-0.1, 1.1])
def test_threshold_calibration_rejects_invalid_fpr(target_fpr: float) -> None:
    with pytest.raises(ValueError, match="target_fpr"):
        select_threshold_for_target_fpr(
            np.asarray([0, 1]),
            np.asarray([0.2, 0.8]),
            target_fpr=target_fpr,
        )


def test_grouped_calibration_split_is_disjoint_and_deterministic() -> None:
    groups = np.asarray(["a", "a", "b", "b", "c", "c", "d", "d"])

    first_fit, first_calibration = grouped_calibration_indices(groups, fraction=0.25, seed=7)
    second_fit, second_calibration = grouped_calibration_indices(groups, fraction=0.25, seed=7)

    assert np.array_equal(first_fit, second_fit)
    assert np.array_equal(first_calibration, second_calibration)
    assert set(groups[first_fit]).isdisjoint(set(groups[first_calibration]))
    assert len(first_fit) == 6
    assert len(first_calibration) == 2
