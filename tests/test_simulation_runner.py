import numpy as np

from illegal_parking.frame_sources import EdgeProfile, SyntheticFrameSource
from illegal_parking.simulation import run_frame_source


def test_run_frame_source_counts_processed_and_skipped_frames():
    source = SyntheticFrameSource(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(10)],
        camera_id="sim_cam",
    )
    profile = EdgeProfile(name="low_rate", source_fps=10.0, target_fps=2.0)

    summary = run_frame_source(source, profile)

    assert summary.source_type == "synthetic"
    assert summary.camera_id == "sim_cam"
    assert summary.total_frames == 10
    assert summary.processed_frames == 2
    assert summary.skipped_frames == 8


def test_run_frame_source_releases_source():
    class ReleasableSource(SyntheticFrameSource):
        def __init__(self):
            super().__init__([np.zeros((2, 2, 3), dtype=np.uint8)])
            self.released = False

        def release(self):
            self.released = True

    source = ReleasableSource()

    run_frame_source(source, EdgeProfile(name="full_rate"))

    assert source.released


def test_run_frame_source_stops_at_max_frames():
    source = SyntheticFrameSource(
        [np.zeros((2, 2, 3), dtype=np.uint8) for _ in range(10)],
        camera_id="sim_cam",
    )

    summary = run_frame_source(
        source,
        EdgeProfile(name="full_rate", source_fps=10.0, target_fps=10.0),
        max_frames=3,
    )

    assert summary.total_frames == 3
    assert summary.processed_frames == 3
    assert summary.effective_fps >= 0.0
