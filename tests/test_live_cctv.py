import numpy as np

from illegal_parking.frame_sources import HTTPImageSource
from illegal_parking.live_cctv import extract_media_urls


def test_extract_media_urls_reads_img_and_vlc_param_sources():
    html = """
    <html>
      <body>
        <param name="src" value="/stream-a" />
        <img src="https://example.test/snapshot.jpg">
      </body>
    </html>
    """

    urls = extract_media_urls(html, base_url="https://camera.example.test/page")

    assert urls == [
        "https://camera.example.test/stream-a",
        "https://example.test/snapshot.jpg",
    ]


def test_http_image_source_decodes_jpeg_bytes(monkeypatch):
    import cv2

    frame = np.zeros((4, 6, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok

    class FakeResponse:
        def __init__(self):
            self._done = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size=-1):
            if self._done:
                return b""
            self._done = True
            return encoded.tobytes()

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())

    source = HTTPImageSource("https://example.test/frame.jpg", camera_id="cam_1", max_frames=1)

    ok, actual = source.read()
    done, empty = source.read()

    assert ok
    assert actual.shape == (4, 6, 3)
    assert not done
    assert empty is None
    assert source.get_metadata()["source_type"] == "image_url"


def test_http_image_source_records_fetch_errors(monkeypatch):
    def fail(_request, timeout):
        raise TimeoutError("timed out")

    monkeypatch.setattr("urllib.request.urlopen", fail)
    source = HTTPImageSource("https://example.test/frame.jpg", max_frames=1)

    ok, frame = source.read()

    assert not ok
    assert frame is None
    assert "timed out" in source.get_metadata()["last_error"]


def test_http_image_source_extracts_first_jpeg_from_stream(monkeypatch):
    import cv2

    frame = np.zeros((3, 5, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".jpg", frame)
    assert ok
    chunks = [b"--frame\r\nContent-Type: image/jpeg\r\n\r\n", encoded.tobytes()[:20], encoded.tobytes()[20:] + b"\r\n"]

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def read(self, _size=-1):
            return chunks.pop(0) if chunks else b""

    monkeypatch.setattr("urllib.request.urlopen", lambda request, timeout: FakeResponse())
    source = HTTPImageSource("https://example.test/mjpeg", max_frames=1)

    ok, actual = source.read()

    assert ok
    assert actual.shape == (3, 5, 3)
