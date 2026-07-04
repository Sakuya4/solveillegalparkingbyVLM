from __future__ import annotations

from dataclasses import asdict, dataclass
import time

from .frame_sources import EdgeProfile, FrameSource


@dataclass(frozen=True)
class SimulationSummary:
    camera_id: str
    source_type: str
    profile_name: str
    total_frames: int
    processed_frames: int
    skipped_frames: int
    elapsed_sec: float
    effective_fps: float
    source_metadata: dict

    def to_dict(self) -> dict:
        return asdict(self)


def run_frame_source(source: FrameSource, profile: EdgeProfile, max_frames: int | None = None) -> SimulationSummary:
    total_frames = 0
    processed_frames = 0
    start = time.perf_counter()

    try:
        while True:
            if max_frames is not None and total_frames >= max_frames:
                break
            ok, frame = source.read()
            if not ok or frame is None:
                break

            if profile.should_process_frame(total_frames):
                processed_frames += 1

            total_frames += 1
    finally:
        source.release()

    elapsed_sec = time.perf_counter() - start
    metadata = source.get_metadata()
    return SimulationSummary(
        camera_id=str(metadata.get("camera_id", "unknown")),
        source_type=str(metadata.get("source_type", "unknown")),
        profile_name=profile.name,
        total_frames=total_frames,
        processed_frames=processed_frames,
        skipped_frames=total_frames - processed_frames,
        elapsed_sec=elapsed_sec,
        effective_fps=total_frames / elapsed_sec if elapsed_sec > 0 else 0.0,
        source_metadata=metadata,
    )
