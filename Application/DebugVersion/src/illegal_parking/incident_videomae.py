from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def remap_legacy_attention_biases(state_dict: dict[str, Any]) -> dict[str, Any]:
    mapped = dict(state_dict)
    query_bias_keys = [key for key in mapped if key.endswith(".attention.attention.q_bias")]
    for query_bias_key in query_bias_keys:
        prefix = query_bias_key.removesuffix(".q_bias")
        value_bias_key = f"{prefix}.v_bias"
        if value_bias_key not in mapped:
            raise ValueError(f"Missing VideoMAE value bias for {query_bias_key}")
        query_bias = mapped.pop(query_bias_key)
        value_bias = mapped.pop(value_bias_key)
        mapped[f"{prefix}.query.bias"] = query_bias
        mapped[f"{prefix}.key.bias"] = query_bias * 0
        mapped[f"{prefix}.value.bias"] = value_bias
    return mapped


def prepare_videomae_state_dict(
    state_dict: dict[str, Any],
    expected_keys: set[str],
) -> dict[str, Any]:
    has_legacy_biases = any(key.endswith(".attention.attention.q_bias") for key in state_dict)
    if not has_legacy_biases:
        return dict(state_dict)
    if any(key.endswith(".attention.attention.q_bias") for key in expected_keys):
        return dict(state_dict)
    if any(key.endswith(".attention.attention.query.bias") for key in expected_keys):
        return remap_legacy_attention_biases(state_dict)
    raise ValueError("Installed Transformers model has an unsupported VideoMAE attention layout")


def load_compatible_videomae_classifier(model_id: str, device: str):
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import VideoMAEConfig, VideoMAEForVideoClassification

    config = VideoMAEConfig.from_pretrained(model_id)
    model = VideoMAEForVideoClassification(config)
    checkpoint_path = Path(hf_hub_download(model_id, "pytorch_model.bin"))
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    compatible_state = prepare_videomae_state_dict(state_dict, set(model.state_dict()))
    model.load_state_dict(compatible_state, strict=True)
    model.to(torch.device(device)).eval()
    return model, checkpoint_path.parent.name


def uniform_frame_indices(
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> np.ndarray:
    if start_frame < 0 or end_frame <= start_frame:
        raise ValueError("end_frame must be greater than non-negative start_frame")
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    if end_frame - start_frame < num_frames:
        raise ValueError("Video window has fewer frames than num_frames")
    return np.linspace(
        start_frame,
        end_frame,
        num=num_frames,
        endpoint=False,
        dtype=np.int64,
    )


def read_uniform_video_window(
    video_path: str | Path,
    start_frame: int,
    end_frame: int,
    num_frames: int,
) -> list[np.ndarray]:
    indices = uniform_frame_indices(start_frame, end_frame, num_frames)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    capture.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0]))

    requested = set(indices.tolist())
    frames: list[np.ndarray] = []
    current_index = int(indices[0])
    last_index = int(indices[-1])
    try:
        while current_index <= last_index:
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Video ended before requested frame {current_index}: {video_path}"
                )
            if current_index in requested:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            current_index += 1
    finally:
        capture.release()
    if len(frames) != num_frames:
        raise ValueError(f"Expected {num_frames} frames, decoded {len(frames)}: {video_path}")
    return frames
