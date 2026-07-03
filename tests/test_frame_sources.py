import numpy as np

from illegal_parking.frame_sources import EdgeProfile, SyntheticFrameSource


def test_synthetic_frame_source_reads_frames_in_order():
    frames = [
        np.zeros((2, 3, 3), dtype=np.uint8),
        np.ones((2, 3, 3), dtype=np.uint8),
    ]
    source = SyntheticFrameSource(frames, camera_id="sim_cam")

    ok1, frame1 = source.read()
    ok2, frame2 = source.read()
    ok3, frame3 = source.read()

    assert ok1
    assert ok2
    assert not ok3
    assert frame3 is None
    assert int(frame1.sum()) == 0
    assert int(frame2.sum()) == 18


def test_synthetic_frame_source_metadata_tracks_position():
    source = SyntheticFrameSource([np.zeros((2, 3, 3), dtype=np.uint8)], camera_id="sim_cam")

    assert source.get_metadata()["frame_index"] == 0

    source.read()

    metadata = source.get_metadata()
    assert metadata["camera_id"] == "sim_cam"
    assert metadata["frame_index"] == 1
    assert metadata["source_type"] == "synthetic"


def test_edge_profile_can_downsample_frame_rate():
    profile = EdgeProfile(name="tiny_edge", source_fps=30.0, target_fps=5.0)

    kept = [idx for idx in range(12) if profile.should_process_frame(idx)]

    assert kept == [0, 6]


def test_edge_profile_processes_every_frame_when_target_matches_source():
    profile = EdgeProfile(name="full_rate", source_fps=10.0, target_fps=10.0)

    kept = [idx for idx in range(4) if profile.should_process_frame(idx)]

    assert kept == [0, 1, 2, 3]
