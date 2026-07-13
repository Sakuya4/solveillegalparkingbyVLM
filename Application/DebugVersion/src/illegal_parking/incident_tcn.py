from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional


@dataclass(frozen=True)
class SequenceStandardizer:
    feature_mean: np.ndarray
    feature_scale: np.ndarray

    def transform(self, features: np.ndarray) -> np.ndarray:
        matrix = _sequence_array(features)
        if matrix.shape[2] != len(self.feature_mean):
            raise ValueError("Sequence feature count does not match the standardizer")
        return ((matrix - self.feature_mean) / self.feature_scale).astype(np.float32)

    def to_dict(self) -> dict:
        return {
            "feature_mean": self.feature_mean.tolist(),
            "feature_scale": self.feature_scale.tolist(),
        }


def fit_sequence_standardizer(features: np.ndarray) -> SequenceStandardizer:
    matrix = _sequence_array(features)
    feature_mean = np.mean(matrix, axis=(0, 1))
    feature_scale = np.std(matrix, axis=(0, 1))
    feature_scale = np.where(feature_scale < 1e-9, 1.0, feature_scale)
    return SequenceStandardizer(feature_mean=feature_mean, feature_scale=feature_scale)


class _CausalConv1d(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.left_padding = (kernel_size - 1) * dilation
        self.convolution = nn.Conv1d(
            channels,
            channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.convolution(functional.pad(inputs, (self.left_padding, 0)))


class _TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.first = _CausalConv1d(channels, kernel_size=3, dilation=dilation)
        self.second = _CausalConv1d(channels, kernel_size=3, dilation=dilation)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.dropout(self.activation(self.first(inputs)))
        hidden = self.dropout(self.activation(self.second(hidden)))
        return self.activation(inputs + hidden)


class TemporalConvClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        channels: int = 32,
        dropout: float = 0.1,
        dilations: tuple[int, ...] = (1, 2, 4),
    ) -> None:
        super().__init__()
        if input_features <= 0 or channels <= 0:
            raise ValueError("input_features and channels must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if not dilations or any(dilation <= 0 for dilation in dilations):
            raise ValueError("dilations must be positive")
        self.input_features = input_features
        self.input_projection = nn.Conv1d(input_features, channels, kernel_size=1)
        self.blocks = nn.Sequential(
            *[_TemporalResidualBlock(channels, dilation, dropout) for dilation in dilations]
        )
        self.classifier = nn.Linear(channels, 1)

    def encode_steps(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 3:
            raise ValueError("inputs must have shape (batch, time, features)")
        if inputs.shape[2] != self.input_features:
            raise ValueError(f"Expected {self.input_features} features, got {inputs.shape[2]}")
        projected = self.input_projection(inputs.transpose(1, 2))
        return self.blocks(projected)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.encode_steps(inputs)
        return self.classifier(hidden[:, :, -1]).squeeze(1)


def _sequence_array(features: np.ndarray) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float32)
    if matrix.ndim != 3 or any(size <= 0 for size in matrix.shape):
        raise ValueError("features must have shape (samples, time, features)")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("features must contain only finite values")
    return matrix
