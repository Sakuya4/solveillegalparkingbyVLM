from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .incident_tcn import SequenceStandardizer, TemporalConvClassifier


@dataclass
class IncidentTcnPredictor:
    model: TemporalConvClassifier
    standardizer: SequenceStandardizer
    feature_names: tuple[str, ...]
    threshold: float
    device: torch.device

    @classmethod
    def from_checkpoint(cls, path: str | Path, device: str = "cpu") -> "IncidentTcnPredictor":
        resolved_device = torch.device(device)
        checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
        model = TemporalConvClassifier(
            input_features=int(checkpoint["input_features"]),
            channels=int(checkpoint["channels"]),
            dropout=float(checkpoint["dropout"]),
        ).to(resolved_device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        standardizer_payload = checkpoint["standardizer"]
        standardizer = SequenceStandardizer(
            feature_mean=np.asarray(standardizer_payload["feature_mean"], dtype=np.float32),
            feature_scale=np.asarray(standardizer_payload["feature_scale"], dtype=np.float32),
        )
        return cls(
            model=model,
            standardizer=standardizer,
            feature_names=tuple(checkpoint["feature_names"]),
            threshold=float(checkpoint["threshold"]),
            device=resolved_device,
        )

    def predict_probabilities(self, features: np.ndarray) -> np.ndarray:
        standardized = self.standardizer.transform(features)
        with torch.no_grad():
            logits = self.model(torch.from_numpy(standardized).to(self.device))
            return torch.sigmoid(logits).cpu().numpy().astype(np.float64)


def build_sequence_windows(
    step_features: np.ndarray,
    sequence_steps: int,
    hop_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(step_features, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] <= 0:
        raise ValueError("step_features must have shape (time, features)")
    if sequence_steps <= 0 or hop_steps <= 0:
        raise ValueError("sequence_steps and hop_steps must be positive")
    starts = np.arange(0, len(matrix) - sequence_steps + 1, hop_steps, dtype=np.int64)
    if not len(starts):
        raise ValueError("step_features does not contain a complete window")
    return np.stack([matrix[start : start + sequence_steps] for start in starts]), starts


def summarize_normal_video_predictions(
    probabilities: np.ndarray,
    threshold: float,
    observed_duration_sec: float,
    min_consecutive_windows: int = 1,
) -> dict[str, float | int]:
    if observed_duration_sec <= 0:
        raise ValueError("observed_duration_sec must be positive")
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not 0.0 <= threshold <= 1.0 or not np.all(np.isfinite(scores)):
        raise ValueError("threshold and probabilities must be finite values in range")
    if min_consecutive_windows <= 0:
        raise ValueError("min_consecutive_windows must be positive")
    positives = scores >= threshold
    alert_episodes = _count_persistent_runs(positives, min_consecutive_windows)
    return {
        "evaluated_windows": len(scores),
        "positive_windows": int(positives.sum()),
        "raw_positive_windows": int(positives.sum()),
        "min_consecutive_windows": min_consecutive_windows,
        "false_alert_episodes": alert_episodes,
        "observed_camera_hours": observed_duration_sec / 3600.0,
        "false_alerts_per_camera_hour": alert_episodes / (observed_duration_sec / 3600.0),
    }


def persistent_positive_flags(
    probabilities: np.ndarray,
    threshold: float,
    min_consecutive_windows: int,
) -> np.ndarray:
    scores = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one")
    if not np.all(np.isfinite(scores)) or np.any((scores < 0.0) | (scores > 1.0)):
        raise ValueError("probabilities must be finite values between zero and one")
    if min_consecutive_windows <= 0:
        raise ValueError("min_consecutive_windows must be positive")
    flags = np.zeros(len(scores), dtype=np.bool_)
    run_length = 0
    for index, positive in enumerate(scores >= threshold):
        run_length = run_length + 1 if positive else 0
        flags[index] = run_length >= min_consecutive_windows
    return flags


def _count_persistent_runs(values: np.ndarray, minimum_length: int) -> int:
    run_length = 0
    count = 0
    for value in values:
        if value:
            run_length += 1
        else:
            count += int(run_length >= minimum_length)
            run_length = 0
    return count + int(run_length >= minimum_length)
