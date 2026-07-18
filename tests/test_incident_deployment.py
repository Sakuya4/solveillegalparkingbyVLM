from __future__ import annotations

import numpy as np
import torch

from illegal_parking.incident_deployment import StandardizedIncidentTcn
from illegal_parking.incident_tcn import SequenceStandardizer, TemporalConvClassifier


def test_standardized_incident_tcn_matches_native_preprocessing() -> None:
    torch.manual_seed(7)
    model = TemporalConvClassifier(input_features=3, channels=4, dropout=0.0).eval()
    standardizer = SequenceStandardizer(
        feature_mean=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
        feature_scale=np.asarray([2.0, 4.0, 5.0], dtype=np.float32),
    )
    deployment = StandardizedIncidentTcn(model, standardizer).eval()
    raw = np.arange(30, dtype=np.float32).reshape(2, 5, 3)

    with torch.no_grad():
        expected = torch.sigmoid(model(torch.from_numpy(standardizer.transform(raw))))
        actual = deployment(torch.from_numpy(raw))

    torch.testing.assert_close(actual, expected)
