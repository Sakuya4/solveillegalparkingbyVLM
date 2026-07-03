from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
