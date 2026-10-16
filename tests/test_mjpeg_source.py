import cv2
import numpy as np

from illegal_parking.frame_sources import MJPEGFrameSource


def test_persistent_mjpeg_reader_preserves_multiple_frames_and_one_connection(monkeypatch):
    images = [np.full((4, 6, 3), value, dtype=np.uint8) for value in (20, 180)]
    payloads = [cv2.imencode(".jpg", frame)[1].tobytes() for frame in images]
    data = b"--frame\r\n" + payloads[0] + b"\r\n--frame\r\n" + payloads[1]
    calls = []

    class Response:
        closed = False

        def __init__(self):
            self.chunks = [data[:33], data[33:]]

        def read1(self, size):
            return self.chunks.pop(0) if self.chunks else b""

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: calls.append(args) or response)
    source = MJPEGFrameSource("https://camera.example.test/stream")
    for expected in images:
        ok, actual = source.read()
        assert ok
        assert np.allclose(actual, expected, atol=2)
    assert len(calls) == 1
    assert source.get_metadata()["frame_index"] == 2
    source.release()
    assert response.closed


def test_mjpeg_eof_and_byte_limit_release_connection(monkeypatch):
    class Response:
        closed = False

        def read1(self, size):
            return b"x" * size

        def close(self):
            self.closed = True

    response = Response()
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: response)
    source = MJPEGFrameSource("https://camera.example.test/stream", max_frame_bytes=10)
    ok, frame = source.read()
    assert not ok and frame is None
    assert response.closed
    assert "byte limit" in source.get_metadata()["last_error"]
