from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time

from .detectors import Detector
from .evidence_writer import EvidenceWriter
from .frame_sources import EdgeProfile, FrameSource
from .scene import SceneAnalyzer
from .simple_tracker import SimpleIouTracker
from .violation_engine import ViolationEngine


@dataclass(frozen=True)
class PipelineSummary:
    camera_id: str
    source_type: str
    total_frames: int
    processed_frames: int
    skipped_frames: int
    detections_seen: int
    events_written: int
    events_path: str | None
    elapsed_sec: float
    effective_fps: float
    source_metadata: dict

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "source_type": self.source_type,
            "total_frames": self.total_frames,
            "processed_frames": self.processed_frames,
            "skipped_frames": self.skipped_frames,
            "detections_seen": self.detections_seen,
            "events_written": self.events_written,
            "events_path": self.events_path,
            "elapsed_sec": self.elapsed_sec,
            "effective_fps": self.effective_fps,
            "source_metadata": self.source_metadata,
        }


class EventPipeline:
    def __init__(
        self,
        source: FrameSource,
        detector: Detector,
        scene_analyzer: SceneAnalyzer,
        violation_engine: ViolationEngine,
        output_dir: str | Path,
        profile: EdgeProfile,
        tracker: SimpleIouTracker | None = None,
        max_frames: int | None = None,
    ):
        self.source = source
        self.detector = detector
        self.scene_analyzer = scene_analyzer
        self.violation_engine = violation_engine
        self.output_dir = Path(output_dir)
        self.profile = profile
        self.tracker = tracker or SimpleIouTracker()
        self.writer = EvidenceWriter(output_dir)
        self._emitted_track_ids: set[int] = set()
        self.max_frames = max_frames

    def run(self) -> PipelineSummary:
        total_frames = 0
        processed_frames = 0
        detections_seen = 0
        events_written = 0
        start = time.perf_counter()

        try:
            while True:
                if self.max_frames is not None and total_frames >= self.max_frames:
                    break
                ok, frame = self.source.read()
                if not ok or frame is None:
                    break

                if not self.profile.should_process_frame(total_frames):
                    total_frames += 1
                    continue

                timestamp_sec = total_frames / max(1e-6, self.profile.source_fps)
                detections = self.detector.detect(frame)
                detections_seen += len(detections)
                tracks = self.tracker.update(detections, timestamp_sec)

                metadata = self.source.get_metadata()
                camera_id = str(metadata.get("camera_id", "unknown"))
                for track in tracks:
                    scene = self.scene_analyzer.analyze(frame, track.bbox)
                    candidate = self.violation_engine.evaluate(
                        track,
                        redline_overlap_ratio=scene.redline_overlap_ratio,
                        in_no_parking_roi=scene.in_no_parking_roi,
                        mask_evidence=scene.mask_evidence,
                    )
                    if candidate.is_candidate and track.track_id not in self._emitted_track_ids:
                        self.writer.write_event(
                            camera_id=camera_id,
                            frame_index=total_frames,
                            frame_bgr=frame,
                            track=track,
                            candidate=candidate,
                            redline_mask=self.scene_analyzer.get_last_mask(),
                        )
                        self._emitted_track_ids.add(track.track_id)
                        events_written += 1

                processed_frames += 1
                total_frames += 1
        finally:
            self.source.release()

        elapsed_sec = time.perf_counter() - start
        metadata = self.source.get_metadata()
        events_path = self.writer.events_path if self.writer.events_path.exists() else None
        return PipelineSummary(
            camera_id=str(metadata.get("camera_id", "unknown")),
            source_type=str(metadata.get("source_type", "unknown")),
            total_frames=total_frames,
            processed_frames=processed_frames,
            skipped_frames=total_frames - processed_frames,
            detections_seen=detections_seen,
            events_written=events_written,
            events_path=str(events_path) if events_path else None,
            elapsed_sec=elapsed_sec,
            effective_fps=total_frames / elapsed_sec if elapsed_sec > 0 else 0.0,
            source_metadata=metadata,
        )
