from __future__ import annotations

import time

import numpy as np

from .incident_ultralytics import reset_ultralytics_trackers, track_vehicle_frame


def warm_camera_models(detector, predictor):
    """Initialize lazy inference backends before the camera observation clock."""
    started = time.perf_counter()
    try:
        track_vehicle_frame(detector, np.zeros((320, 320, 3), dtype=np.uint8), 0)
        predictor.predict_probabilities(np.zeros((1, 15, len(predictor.feature_names)), dtype=np.float32))
    finally:
        # Synthetic initialization must not seed live tracking or temporal state.
        reset_ultralytics_trackers(detector)
    return time.perf_counter() - started
