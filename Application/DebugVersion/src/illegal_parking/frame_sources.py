from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Protocol

import numpy as np


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]:
        ...

    def get_metadata(self) -> dict:
        ...

    def release(self) -> None:
        ...


@dataclass(frozen=True)
class EdgeProfile:
    name: str
    source_fps: float = 30.0
    target_fps: float = 10.0
    input_size: tuple[int, int] | None = None
    detector_imgsz: int = 640
    vlm_review_enabled: bool = False
    max_events_per_minute: int = 3

    def should_process_frame(self, frame_index: int) -> bool:
        if self.target_fps <= 0:
            return False
        if self.target_fps >= self.source_fps:
            return True
        stride = max(1, round(self.source_fps / self.target_fps))
        return frame_index % stride == 0


class SyntheticFrameSource:
    def __init__(self, frames: list[np.ndarray], camera_id: str = "synthetic_camera"):
        self._frames = frames
        self._index = 0
        self._camera_id = camera_id

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._frames):
            return False, None
        frame = self._frames[self._index]
        self._index += 1
        return True, frame.copy()

    def get_metadata(self) -> dict:
        return {
            "camera_id": self._camera_id,
            "frame_index": self._index,
            "source_type": "synthetic",
            "total_frames": len(self._frames),
        }

    def release(self) -> None:
        return None


class ImageFolderSource:
    def __init__(self, folder_path: str | Path, camera_id: str = "image_folder"):
        self.folder_path = Path(folder_path)
        patterns = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self._paths = sorted(path for pattern in patterns for path in self.folder_path.glob(pattern))
        self._index = 0
        self._camera_id = camera_id

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._index >= len(self._paths):
            return False, None
        import cv2

        path = self._paths[self._index]
        self._index += 1
        frame = cv2.imread(str(path))
        return (frame is not None), frame

    def get_metadata(self) -> dict:
        return {
            "camera_id": self._camera_id,
            "frame_index": self._index,
            "source_type": "image_folder",
            "path": str(self.folder_path),
            "total_frames": len(self._paths),
        }

    def release(self) -> None:
        return None


class HTTPImageSource:
    def __init__(
        self,
        image_url: str,
        camera_id: str = "image_url",
        max_frames: int | None = 1,
        timeout_sec: float = 10.0,
        poll_interval_sec: float = 0.0,
        max_read_bytes: int = 5_000_000,
    ):
        self.image_url = image_url
        self._camera_id = camera_id
        self._max_frames = max_frames
        self._timeout_sec = timeout_sec
        self._poll_interval_sec = poll_interval_sec
        self._max_read_bytes = max_read_bytes
        self._frame_index = 0
        self._last_read_latency_sec: float | None = None
        self._last_error: str | None = None

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self._max_frames is not None and self._frame_index >= self._max_frames:
            return False, None

        import cv2
        import urllib.request

        if self._frame_index > 0 and self._poll_interval_sec > 0:
            time.sleep(self._poll_interval_sec)

        request = urllib.request.Request(
            self.image_url,
            headers={"User-Agent": "solveillegalparkingbyVLM live-cctv/0.1"},
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(request, timeout=self._timeout_sec) as response:
                data = _read_image_payload(response, self._max_read_bytes)
        except Exception as exc:
            self._last_read_latency_sec = time.perf_counter() - start
            self._last_error = str(exc)
            return False, None
        self._last_read_latency_sec = time.perf_counter() - start

        encoded = np.frombuffer(data, dtype=np.uint8)
        frame = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if frame is None:
            self._last_error = "response was not a decodable image"
            return False, None

        self._last_error = None
        self._frame_index += 1
        return True, frame

    def get_metadata(self) -> dict:
        return {
            "camera_id": self._camera_id,
            "frame_index": self._frame_index,
            "source_type": "image_url",
            "source": self.image_url,
            "last_read_latency_sec": self._last_read_latency_sec,
            "last_error": self._last_error,
        }

    def release(self) -> None:
        return None


def _read_image_payload(response, max_read_bytes: int) -> bytes:
    data = bytearray()
    while len(data) < max_read_bytes:
        chunk = response.read(min(8192, max_read_bytes - len(data)))
        if not chunk:
            break
        data.extend(chunk)
        jpeg = _extract_jpeg(bytes(data))
        if jpeg is not None:
            return jpeg
    return bytes(data)


def _extract_jpeg(data: bytes) -> bytes | None:
    start = data.find(b"\xff\xd8")
    if start < 0:
        return None
    end = data.find(b"\xff\xd9", start + 2)
    if end < 0:
        return None
    return data[start : end + 2]


class OpenCVStreamSource:
    def __init__(self, source, source_type: str, camera_id: str):
        import cv2

        self._cv2 = cv2
        self._source = source
        self._source_type = source_type
        self._camera_id = camera_id
        self._frame_index = 0
        self._cap = cv2.VideoCapture(source)

    def read(self) -> tuple[bool, np.ndarray | None]:
        ok, frame = self._cap.read()
        if ok:
            self._frame_index += 1
        return ok, frame if ok else None

    def get_metadata(self) -> dict:
        return {
            "camera_id": self._camera_id,
            "frame_index": self._frame_index,
            "source_type": self._source_type,
            "source": str(self._source),
            "is_opened": bool(self._cap.isOpened()),
        }

    def release(self) -> None:
        self._cap.release()


class VideoFileSource(OpenCVStreamSource):
    def __init__(self, video_path: str | Path, camera_id: str = "video_file"):
        super().__init__(str(video_path), "video_file", camera_id)


class WebcamSource(OpenCVStreamSource):
    def __init__(self, camera_index: int = 0, camera_id: str = "webcam"):
        super().__init__(camera_index, "webcam", camera_id)


class NetworkStreamSource(OpenCVStreamSource):
    def __init__(self, stream_url: str, camera_id: str = "stream_url"):
        super().__init__(stream_url, "stream_url", camera_id)
