from __future__ import annotations

import numpy as np
import pytest
import torch

from illegal_parking.incident_tcn import (
    TemporalConvClassifier,
    fit_sequence_standardizer,
)


def test_temporal_conv_classifier_returns_one_logit_per_sequence() -> None:
    model = TemporalConvClassifier(input_features=8, channels=12, dropout=0.0)

    logits = model(torch.randn(4, 15, 8))

    assert logits.shape == (4,)


def test_temporal_encoder_is_causal_for_shared_prefix() -> None:
    torch.manual_seed(3)
    model = TemporalConvClassifier(input_features=2, channels=6, dropout=0.0).eval()
    first = torch.zeros(1, 12, 2)
    second = first.clone()
    second[:, 8:, :] = 10.0

    with torch.no_grad():
        first_steps = model.encode_steps(first)
        second_steps = model.encode_steps(second)

    assert torch.allclose(first_steps[:, :, :8], second_steps[:, :, :8], atol=1e-6)


def test_sequence_standardizer_uses_training_statistics() -> None:
    features = np.asarray(
        [
            [[1.0, 10.0], [3.0, 14.0]],
            [[5.0, 18.0], [7.0, 22.0]],
        ],
        dtype=np.float32,
    )

    standardizer = fit_sequence_standardizer(features)
    transformed = standardizer.transform(features)

    assert transformed.mean(axis=(0, 1)) == pytest.approx([0.0, 0.0], abs=1e-6)
    assert transformed.std(axis=(0, 1)) == pytest.approx([1.0, 1.0], abs=1e-6)


def test_temporal_conv_rejects_wrong_input_shape() -> None:
    model = TemporalConvClassifier(input_features=8)

    with pytest.raises(ValueError, match="batch, time, features"):
        model(torch.randn(3, 8))
