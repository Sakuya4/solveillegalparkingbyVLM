import numpy as np
import pytest

from scripts.audit_presentation_event_predictions import clock_control, control_results, verify_feature_predictions
from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES
from illegal_parking.incident_presentation_events import evaluate_presentation_event


def test_clock_control_does_not_consult_images_or_model_probabilities():
    report = {"path": "clip", "clock": "normalized_presentation_time", "annotation_free_inference": True,
              "window_starts": [0, 15, 45, 60], "window_end_times_sec": [2, 4, 8, 10],
              "window_probabilities": [float("nan")] * 4, "rejected_window_starts": [30]}
    annotation = {"path": "clip", "event_time_sec": 7, "collision_type": "rear-end"}
    positive = clock_control(report, True)
    assert positive["window_probabilities"] == [1.] * 4
    result = evaluate_presentation_event(positive, annotation, .5)
    assert result.detected and result.trigger_time_sec == 10
    assert result.early_alert_episodes == 1 and result.trigger_delay_sec == 3
    controls = control_results([report], {"clip": annotation})
    assert controls["clock_only_positive"]["event_recall"] == 1
    assert controls["always_negative"]["event_recall"] == 0
    assert controls["always_negative"]["clip_count"] == 1
    assert np.isnan(report["window_probabilities"][0])


def test_saved_feature_prediction_audit_rejects_changed_scores(tmp_path, monkeypatch):
    from scripts import audit_presentation_event_predictions as audit

    class Predictor:
        feature_names = ("candidate_diff_mean",)

        def predict_probabilities(self, features):
            assert features.shape == (2, 15, 1)
            return np.asarray([.1, .9])

    monkeypatch.setattr(audit.IncidentTcnPredictor, "from_checkpoint", lambda *args: Predictor())
    features = tmp_path / "features.npz"
    np.savez_compressed(features, features=np.zeros((2, 15, 14), dtype=np.float32), feature_names=np.asarray(CANDIDATE_MOTION_FEATURE_NAMES))
    batch = {"clips": {"clip": {"features": features.name}}}
    protocol = {"models": {"key": {"path": "fixture.pt"}}}
    reports = [{"path": "clip", "window_starts": [0, 15], "scores": {"key": [.1, .9]}}]
    result = verify_feature_predictions(tmp_path, batch, reports, protocol)
    assert result["verified_probabilities"] == 2 and result["maximum_absolute_error"] == 0
    reports[0]["scores"]["key"][0] = .101
    with pytest.raises(ValueError, match="reproduce"):
        verify_feature_predictions(tmp_path, batch, reports, protocol)
