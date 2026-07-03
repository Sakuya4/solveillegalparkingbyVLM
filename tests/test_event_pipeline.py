import json
import numpy as np

from illegal_parking.detectors import DetectionRecord, StaticDetector
from illegal_parking.event_models import BBox
from illegal_parking.frame_sources import EdgeProfile, SyntheticFrameSource
from illegal_parking.pipeline import EventPipeline
from illegal_parking.scene import StaticSceneAnalyzer
from illegal_parking.violation_engine import ViolationEngine, ViolationEngineConfig


def test_pipeline_writes_one_candidate_event_with_artifacts(tmp_path):
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(5)]
    source = SyntheticFrameSource(frames, camera_id="sim_cam")
    detector = StaticDetector(
        [
            DetectionRecord(
                bbox=BBox(20, 20, 70, 70),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ]
    )
    scene = StaticSceneAnalyzer(redline_overlap_ratio=0.05, in_no_parking_roi=False)
    engine = ViolationEngine(
        ViolationEngineConfig(
            dwell_threshold_sec=3.0,
            min_stable_frames=1,
            min_redline_overlap_ratio=0.01,
        )
    )

    summary = EventPipeline(
        source=source,
        detector=detector,
        scene_analyzer=scene,
        violation_engine=engine,
        output_dir=tmp_path,
        profile=EdgeProfile(name="test", source_fps=1.0, target_fps=1.0),
    ).run()

    events_path = tmp_path / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text(encoding="utf-8").splitlines()]

    assert summary.total_frames == 5
    assert summary.processed_frames == 5
    assert summary.events_written == 1
    assert len(events) == 1
    assert events[0]["camera_id"] == "sim_cam"
    assert events[0]["track_id"] == 1
    assert events[0]["status"] == "CANDIDATE"
    assert events[0]["rule_evidence"]["dwell_time_sec"] == 3.0
    assert (tmp_path / events[0]["artifacts"]["original_frame"]).exists()
    assert (tmp_path / events[0]["artifacts"]["annotated_frame"]).exists()
    assert (tmp_path / events[0]["artifacts"]["vehicle_crop"]).exists()


def test_pipeline_does_not_write_events_without_detections(tmp_path):
    frames = [np.zeros((80, 120, 3), dtype=np.uint8) for _ in range(3)]
    source = SyntheticFrameSource(frames, camera_id="sim_cam")

    summary = EventPipeline(
        source=source,
        detector=StaticDetector([]),
        scene_analyzer=StaticSceneAnalyzer(redline_overlap_ratio=0.0),
        violation_engine=ViolationEngine(ViolationEngineConfig(dwell_threshold_sec=1.0)),
        output_dir=tmp_path,
        profile=EdgeProfile(name="test", source_fps=1.0, target_fps=1.0),
    ).run()

    assert summary.events_written == 0
    assert not (tmp_path / "events.jsonl").exists()
