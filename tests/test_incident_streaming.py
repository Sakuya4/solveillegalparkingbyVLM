from types import SimpleNamespace
import time

import numpy as np
import pytest

from illegal_parking.incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES, compute_candidate_motion_step, candidate_motion_sequence_matrix
from illegal_parking.incident_streaming import StreamingIncidentProcessor, LatestFrameReader
from illegal_parking.incident_tcn_inference import build_sequence_windows, persistent_positive_flags


class Detector:
    def track(self, **kwargs):
        return [SimpleNamespace(boxes=None, orig_shape=frame.shape[:2]) for frame in kwargs["source"]]


class Predictor:
    feature_names = CANDIDATE_MOTION_FEATURE_NAMES
    threshold = 0.5

    def predict_probabilities(self, windows):
        return np.clip(windows[:, :, 0].mean(axis=1) / 10, 0, 1)


def test_streaming_windows_and_persistence_match_batch_without_future_frames():
    frames = [np.full((24, 32, 3), (index * 9) % 255, dtype=np.uint8) for index in range(80)]
    steps = [compute_candidate_motion_step(a, b, []) for a, b in zip(frames, frames[1:])]
    windows, starts = build_sequence_windows(candidate_motion_sequence_matrix(steps), 15, 15)
    predictor = Predictor()
    expected = predictor.predict_probabilities(windows)
    flags = persistent_positive_flags(expected, predictor.threshold, 2)
    processor = StreamingIncidentProcessor(Detector(), predictor, display_width=32)
    updates = [processor.process(frame, index, index / 7.5) for index, frame in enumerate(frames)]
    decisions = [item for item in updates if item.window_start is not None]
    assert [item.sampled_index for item in decisions] == (starts + 15).tolist()
    np.testing.assert_allclose([item.probability for item in decisions], expected)
    assert [item.review_active for item in decisions] == flags.tolist()
    assert len(processor.features) == 15
    assert all(not update.review_active and update.probability is None for update in updates[:15])
    assert sum(update.new_event for update in updates) == 1
    processor.close()
    assert not processor.features and processor.previous_frame is None


def test_streaming_rejects_nonmonotonic_inputs():
    processor = StreamingIncidentProcessor(Detector(), Predictor())
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    processor.process(frame, 0, 0)
    with pytest.raises(ValueError, match="strictly increase"):
        processor.process(frame, 0, 1)
    with pytest.raises(ValueError, match="strictly increase"):
        processor.process(frame, 1, float("nan"))
    processor.close()


def test_latest_frame_reader_bounds_mailbox_and_closes_source():
    class Source:
        released = False

        def read(self):
            time.sleep(0.002)
            return True, np.zeros((8, 8, 3), dtype=np.uint8)

        def release(self):
            self.released = True

    source = Source()
    reader = LatestFrameReader(source)
    first = reader.read()
    time.sleep(0.03)
    second = reader.read()
    reader.close()
    assert second[0] > first[0] and second[1] > first[1]
    assert reader.overwritten_frames > 0
    assert source.released and not reader.thread.is_alive()


def test_latest_frame_reader_reports_source_failure_without_hanging():
    class Source:
        def read(self):
            return False, None

        def release(self):
            pass

    reader = LatestFrameReader(Source())
    with pytest.raises(ConnectionError, match="read failed"):
        reader.read(timeout_sec=1)
    reader.close()
