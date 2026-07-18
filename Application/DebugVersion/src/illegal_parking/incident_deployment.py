from __future__ import annotations

import torch
from torch import nn

from .incident_tcn import SequenceStandardizer, TemporalConvClassifier


class StandardizedIncidentTcn(nn.Module):
    def __init__(
        self,
        model: TemporalConvClassifier,
        standardizer: SequenceStandardizer,
    ) -> None:
        super().__init__()
        self.model = model
        self.register_buffer(
            "feature_mean",
            torch.as_tensor(standardizer.feature_mean, dtype=torch.float32).view(1, 1, -1),
        )
        self.register_buffer(
            "feature_scale",
            torch.as_tensor(standardizer.feature_scale, dtype=torch.float32).view(1, 1, -1),
        )

    def forward(self, raw_features: torch.Tensor) -> torch.Tensor:
        standardized = (raw_features - self.feature_mean) / self.feature_scale
        return torch.sigmoid(self.model(standardized))
