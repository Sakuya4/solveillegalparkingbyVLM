from __future__ import annotations

import cv2
import numpy as np
import pytest

from illegal_parking.incident_videomae import (
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
