from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import threading
import time

import cv2
import numpy as np

from .incident_candidate_roi import CANDIDATE_MOTION_FEATURE_NAMES, compute_candidate_motion_step
from .incident_ultralytics import track_vehicle_frame, reset_ultralytics_trackers


@dataclass(frozen=True)
class StreamingUpdate:
    sampled_index: int
    source_index: int
    source_time_sec: float
    probability: float | None
    review_active: bool
    new_event: bool
    window_start: int | None
    tracks: list
    step: object
    processing_sec: float


class StreamingIncidentProcessor:
    """Causal candidate features and persistence; memory does not grow with duration."""

    def __init__(self, detector, predictor, *, feature_width=None, display_width=640, sequence_steps=15, hop_steps=15, min_positive_windows=2):
        if min(sequence_steps, hop_steps, min_positive_windows, display_width) <= 0 or (feature_width is not None and feature_width <= 0):
            raise ValueError("Widths, windows, and persistence must be positive")
        unknown = set(predictor.feature_names) - set(CANDIDATE_MOTION_FEATURE_NAMES)
        if unknown:
            raise ValueError(f"Unsupported streaming feature names: {sorted(unknown)}")
        self.feature_indices = [CANDIDATE_MOTION_FEATURE_NAMES.index(name) for name in predictor.feature_names]
        self.detector, self.predictor = detector, predictor
        self.feature_width, self.display_width = feature_width, display_width
        self.sequence_steps, self.hop_steps = sequence_steps, hop_steps
        self.min_positive_windows = min_positive_windows
        self.features = deque(maxlen=sequence_steps)
        self.previous_frame = None
        self.sampled_index = -1
        self.previous_source_index = -1
        self.previous_source_time = -1.0
        self.positive_count = 0
        self.probability = None
        self.review_active = False

    def process(self, frame, source_index: int, source_time_sec: float) -> StreamingUpdate:
        if source_index <= self.previous_source_index or not math.isfinite(source_time_sec) or source_time_sec < 0 or source_time_sec <= self.previous_source_time:
            raise ValueError("Streaming source indices and timestamps must strictly increase")
        started = time.perf_counter()
        height, width = frame.shape[:2]
        limit = self.display_width if self.feature_width is None else min(width, self.feature_width)
        feature_frame = cv2.resize(frame, (limit, max(1, round(height * limit / width))), interpolation=cv2.INTER_AREA) if width != limit else frame
        tracks = track_vehicle_frame(self.detector, feature_frame, source_index)
        step = None
        self.sampled_index += 1
        window_start, new_event = None, False
        if self.previous_frame is not None:
            step = compute_candidate_motion_step(self.previous_frame, feature_frame, tracks)
            values = np.asarray([step.features[name] for name in CANDIDATE_MOTION_FEATURE_NAMES], dtype=np.float32)
            self.features.append(values[self.feature_indices])
            if len(self.features) == self.sequence_steps and (self.sampled_index - self.sequence_steps) % self.hop_steps == 0:
                window_start = self.sampled_index - self.sequence_steps
                probability = float(self.predictor.predict_probabilities(np.asarray([self.features], dtype=np.float32))[0])
                if not math.isfinite(probability) or not 0 <= probability <= 1:
                    raise ValueError("Streaming probability must be finite and in [0, 1]")
                self.probability = probability
                self.positive_count = self.positive_count + 1 if probability >= self.predictor.threshold else 0
                active = self.positive_count >= self.min_positive_windows
                new_event = active and not self.review_active
                self.review_active = active
        self.previous_frame = feature_frame
        self.previous_source_index, self.previous_source_time = source_index, source_time_sec
        return StreamingUpdate(self.sampled_index, source_index, source_time_sec, self.probability, self.review_active, new_event, window_start, tracks, step, time.perf_counter() - started)

    def close(self):
        reset_ultralytics_trackers(self.detector)
        self.previous_frame = None
        self.features.clear()


class LatestFrameReader:
    """A one-frame mailbox drains the source independently of inference latency."""

    def __init__(self, source):
        self.source = source
        self.condition = threading.Condition()
        self.latest = None
        self.stopped = threading.Event()
        self.error = None
        self.decoded_frames = self.overwritten_frames = 0
        self.started = time.perf_counter()
        self.thread = threading.Thread(target=self._run, name="camera-reader", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while not self.stopped.is_set():
                ok, frame = self.source.read()
                if not ok or frame is None:
                    raise ConnectionError("Streaming camera read failed")
                observed = time.perf_counter() - self.started
                with self.condition:
                    self.overwritten_frames += int(self.latest is not None)
                    self.latest = (self.decoded_frames, observed, frame)
                    self.decoded_frames += 1
                    self.condition.notify_all()
        except Exception as exc:
            if not self.stopped.is_set():
                self.error = str(exc)
        finally:
            self.stopped.set()
            with self.condition:
                self.condition.notify_all()

    def read(self, timeout_sec=6):
        with self.condition:
            ready = self.condition.wait_for(lambda: self.latest is not None or self.stopped.is_set(), timeout=timeout_sec)
            if not ready:
                raise TimeoutError("No streaming frame became available")
            if self.latest is None:
                raise ConnectionError(self.error or "Camera reader stopped")
            frame, self.latest = self.latest, None
            return frame

    def close(self):
        self.stopped.set()
        self.source.release()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise TimeoutError("Camera reader did not stop after closing the source")
