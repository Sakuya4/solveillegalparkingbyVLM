import numpy as np
import pytest

from illegal_parking.incident_baseline import binary_classification_metrics
from scripts.publish_full_cohort_results import verify_predictions


def fixture():
    payload = {
        "path": np.array(["train", "negative", "positive"]),
        "label": np.array(["incident", "normal", "incident"]),
        "target": np.array([1, 0, 1]),
        "iid_split": np.array(["train", "test", "test"]),
    }
    predictions = [
        {"path": "positive", "label": "incident", "target": "1", "probability": "0.8"},
        {"path": "negative", "label": "normal", "target": "0", "probability": "0.2"},
    ]
    report = {"decision_threshold": {"effective": 0.5}, "test": binary_classification_metrics(np.array([0, 1]), np.array([0.2, 0.8]))}
    return payload, predictions, report


def test_publication_verifies_test_membership_independently_of_row_order():
    payload, predictions, report = fixture()
    verify_predictions(payload, predictions, "iid", report)


def test_publication_rejects_missing_or_duplicate_test_rows():
    payload, predictions, report = fixture()
    with pytest.raises(ValueError, match="test cohort"):
        verify_predictions(payload, predictions[:1], "iid", report)
    with pytest.raises(ValueError, match="test cohort"):
        verify_predictions(payload, predictions + predictions[:1], "iid", report)


def test_publication_rejects_inconsistent_targets_and_metrics():
    payload, predictions, report = fixture()
    predictions[0]["target"] = "0"
    with pytest.raises(ValueError, match="targets"):
        verify_predictions(payload, predictions, "iid", report)
    predictions[0]["target"] = "1"
    report["test"]["recall"] = 0.5
    with pytest.raises(ValueError, match="metrics"):
        verify_predictions(payload, predictions, "iid", report)
