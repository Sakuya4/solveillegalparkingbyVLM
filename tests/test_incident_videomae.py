from __future__ import annotations

import cv2
import numpy as np
import pytest
from torch import nn

from illegal_parking.incident_videomae import (
    configure_videomae_finetuning,
    prepare_videomae_state_dict,
    read_uniform_video_window,
    remap_legacy_attention_biases,
    uniform_frame_indices,
)


def test_uniform_frame_indices_match_left_aligned_tcn_sampling() -> None:
    indices = uniform_frame_indices(start_frame=10, end_frame=42, num_frames=16)

    assert len(indices) == 16
    assert indices[0] == 10
    assert indices[-1] == 40
    assert indices.tolist() == list(range(10, 42, 2))
    assert np.all(np.diff(indices) > 0)


def test_uniform_frame_indices_reject_invalid_window() -> None:
    with pytest.raises(ValueError, match="end_frame"):
        uniform_frame_indices(4, 4, 2)
    with pytest.raises(ValueError, match="num_frames"):
        uniform_frame_indices(0, 10, 0)
    with pytest.raises(ValueError, match="fewer frames"):
        uniform_frame_indices(0, 3, 4)


def test_read_uniform_video_window_returns_rgb_frames(tmp_path) -> None:
    video_path = tmp_path / "sample.avi"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (32, 24),
    )
    assert writer.isOpened()
    for index in range(10):
        frame = np.zeros((24, 32, 3), dtype=np.uint8)
        frame[:, :, 2] = index * 20
        writer.write(frame)
    writer.release()

    frames = read_uniform_video_window(video_path, 1, 10, num_frames=3)

    assert len(frames) == 3
    assert frames[0].shape == (24, 32, 3)
    assert float(frames[0][:, :, 0].mean()) > float(frames[0][:, :, 2].mean())
    assert float(frames[-1][:, :, 0].mean()) > float(frames[0][:, :, 0].mean())


def test_read_uniform_video_window_rejects_unreadable_video(tmp_path) -> None:
    with pytest.raises(ValueError, match="Cannot open video"):
        read_uniform_video_window(tmp_path / "missing.mp4", 0, 16, num_frames=16)


def test_remap_legacy_attention_biases_creates_strict_qkv_biases() -> None:
    prefix = "videomae.encoder.layer.0.attention.attention"
    state = {
        f"{prefix}.q_bias": np.asarray([1.0, 2.0]),
        f"{prefix}.v_bias": np.asarray([3.0, 4.0]),
        f"{prefix}.query.weight": np.ones((2, 2)),
    }

    mapped = remap_legacy_attention_biases(state)

    assert f"{prefix}.q_bias" not in mapped
    assert f"{prefix}.v_bias" not in mapped
    assert mapped[f"{prefix}.query.bias"].tolist() == [1.0, 2.0]
    assert mapped[f"{prefix}.key.bias"].tolist() == [0.0, 0.0]
    assert mapped[f"{prefix}.value.bias"].tolist() == [3.0, 4.0]


def test_prepare_videomae_state_dict_preserves_legacy_transformers_keys() -> None:
    prefix = "videomae.encoder.layer.0.attention.attention"
    state = {
        f"{prefix}.q_bias": np.asarray([1.0]),
        f"{prefix}.v_bias": np.asarray([2.0]),
    }

    prepared = prepare_videomae_state_dict(
        state,
        expected_keys={f"{prefix}.q_bias", f"{prefix}.v_bias"},
    )

    assert set(prepared) == set(state)


def test_prepare_videomae_state_dict_remaps_split_attention_biases() -> None:
    prefix = "videomae.encoder.layer.0.attention.attention"
    state = {
        f"{prefix}.q_bias": np.asarray([1.0]),
        f"{prefix}.v_bias": np.asarray([2.0]),
    }

    prepared = prepare_videomae_state_dict(
        state,
        expected_keys={
            f"{prefix}.query.bias",
            f"{prefix}.key.bias",
            f"{prefix}.value.bias",
        },
    )

    assert f"{prefix}.q_bias" not in prepared
    assert prepared[f"{prefix}.key.bias"].tolist() == [0.0]


class _TinyVideoMaeClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.videomae = nn.Module()
        self.videomae.encoder = nn.Module()
        self.videomae.encoder.layer = nn.ModuleList([nn.Linear(2, 2) for _ in range(3)])
        self.fc_norm = nn.LayerNorm(2)
        self.classifier = nn.Linear(2, 400)
        self.config = type("Config", (), {"hidden_size": 2})()


def test_configure_videomae_finetuning_only_unfreezes_requested_tail_blocks() -> None:
    model = _TinyVideoMaeClassifier()

    configure_videomae_finetuning(model, num_labels=2, trainable_encoder_blocks=1)

    assert model.classifier.out_features == 2
    assert all(not parameter.requires_grad for parameter in model.videomae.encoder.layer[0].parameters())
    assert all(not parameter.requires_grad for parameter in model.videomae.encoder.layer[1].parameters())
    assert all(parameter.requires_grad for parameter in model.videomae.encoder.layer[2].parameters())
    assert all(parameter.requires_grad for parameter in model.fc_norm.parameters())
    assert all(parameter.requires_grad for parameter in model.classifier.parameters())


def test_configure_videomae_finetuning_rejects_too_many_blocks() -> None:
    model = _TinyVideoMaeClassifier()

    with pytest.raises(ValueError, match="trainable_encoder_blocks"):
        configure_videomae_finetuning(model, num_labels=2, trainable_encoder_blocks=4)
