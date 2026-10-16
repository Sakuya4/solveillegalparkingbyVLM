from types import SimpleNamespace

import numpy as np
import pytest

from illegal_parking.incident_camera_startup import warm_camera_models


@pytest.mark.parametrize("fail", [False, True])
def test_camera_warmup_runs_backends_and_always_resets_synthetic_tracks(fail):
    calls = []

    class Tracker:
        def reset(self):
            calls.append("reset")

    class Detector:
        predictor = SimpleNamespace(trackers=[Tracker()])

        def track(self, **kwargs):
            assert kwargs["source"][0].shape == (320, 320, 3)
            assert not kwargs["source"][0].any()
            assert kwargs["device"] == "cpu" and kwargs["persist"]
            calls.append("detector")
            return [SimpleNamespace(boxes=None)]

    class Predictor:
        feature_names = ["one", "two"]

        def predict_probabilities(self, values):
            assert values.shape == (1, 15, 2)
            assert values.dtype == np.float32 and not values.any()
            calls.append("temporal")
            if fail:
                raise RuntimeError("inference failed")
            return np.zeros(1)

    if fail:
        with pytest.raises(RuntimeError, match="inference failed"):
            warm_camera_models(Detector(), Predictor())
    else:
        assert warm_camera_models(Detector(), Predictor()) >= 0
    assert calls == ["detector", "temporal", "reset"]
