from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from illegal_parking.incident_ultralytics import track_vehicle_frames


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values

    def __len__(self):
        return len(self.values)


@dataclass
class _Boxes:
    xyxy: _Tensor
    id: _Tensor | None
    conf: _Tensor

    def __len__(self):
        return len(self.xyxy)


@dataclass
class _Result:
    boxes: _Boxes
    orig_shape: tuple[int, int]


class _Tracker:
    def __init__(self):
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1


class _Model:
    def __init__(self):
        self.tracker = _Tracker()
        self.predictor = type("Predictor", (), {"trackers": [self.tracker]})()
        self.arguments = None

    def track(self, **kwargs):
        self.arguments = kwargs
        return [
            _Result(
                boxes=_Boxes(
                    xyxy=_Tensor([[10, 20, 30, 50]]),
                    id=_Tensor([7]),
                    conf=_Tensor([0.8]),
                ),
                orig_shape=(100, 200),
            ),
            _Result(
                boxes=_Boxes(xyxy=_Tensor([]).values.reshape(0, 4), id=None, conf=_Tensor([])),
                orig_shape=(100, 200),
            ),
        ]


def test_track_vehicle_frames_preserves_frame_indices_and_resets_tracker() -> None:
    model = _Model()
    frames = [np.zeros((100, 200, 3), dtype=np.uint8) for _ in range(2)]

    tracked = track_vehicle_frames(model, frames, [4, 6], device="cpu", image_size=320)

    assert list(tracked) == [4, 6]
    assert tracked[4][0].track_id == 7
    assert (
        tracked[4][0].x1,
        tracked[4][0].y1,
        tracked[4][0].x2,
        tracked[4][0].y2,
    ) == (0.05, 0.2, 0.15, 0.5)
    assert tracked[6] == []
    assert model.arguments["persist"] is True
    assert model.arguments["classes"] == [2, 3, 5, 7]
    assert model.tracker.reset_count == 1
