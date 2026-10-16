from pathlib import Path

import cv2
import numpy as np
import pytest

from illegal_parking.incident_video_pipeline import _read_sampled_video_views, read_sampled_video


@pytest.fixture
def video(tmp_path):
    path = tmp_path / "source.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (480, 240))
    assert writer.isOpened()
    for index in range(20):
        frame = np.zeros((240, 480, 3), dtype=np.uint8)
        cv2.rectangle(frame, (index * 5, 40), (index * 5 + 60, 120), (220, 120, 80), -1)
        writer.write(frame)
    writer.release()
    return path


def test_feature_view_uses_original_decoded_frame_not_resized_display(video):
    display, features, indices, fps = _read_sampled_video_views(video, 10, 640, 320)
    assert fps == 10 and indices == list(range(20))
    assert display[0].shape == (320, 640, 3)
    assert features[0].shape == (160, 320, 3)
    capture = cv2.VideoCapture(str(video))
    ok, original = capture.read()
    capture.release()
    assert ok
    np.testing.assert_array_equal(features[0], cv2.resize(original, (320, 160), interpolation=cv2.INTER_AREA))


def test_feature_view_never_upscales_and_legacy_api_is_unchanged(video):
    display, features, _, _ = _read_sampled_video_views(video, 10, 640, 800)
    assert features[0].shape[1] == 480
    legacy, indices, fps = read_sampled_video(video, 10, 640)
    old_display, old_features, _, _ = _read_sampled_video_views(video, 10, 640)
    assert old_display[0] is old_features[0]
    np.testing.assert_array_equal(legacy[0], display[0])
    assert len(indices) == 20 and fps == 10


@pytest.mark.parametrize("fps,width,features", [(0, 640, 320), (float("nan"), 640, None), (10, 0, None), (10, 640, -1)])
def test_views_reject_invalid_sampling_parameters(video, fps, width, features):
    with pytest.raises(ValueError, match="positive"):
        _read_sampled_video_views(video, fps, width, features)
