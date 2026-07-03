from illegal_parking.detectors import DetectionRecord
from illegal_parking.event_models import BBox
from illegal_parking.simple_tracker import SimpleIouTracker


def detection(x1=10, y1=10, x2=50, y2=50):
    return DetectionRecord(
        bbox=BBox(x1, y1, x2, y2),
        confidence=0.8,
        class_id=2,
        class_name="car",
    )


def test_tracker_keeps_same_id_for_overlapping_detections():
    tracker = SimpleIouTracker(iou_threshold=0.2)

    first = tracker.update([detection()], timestamp_sec=0.0)
    second = tracker.update([detection(12, 10, 52, 50)], timestamp_sec=1.0)

    assert first[0].track_id == second[0].track_id
    assert second[0].stable_frame_count == 2


def test_tracker_reports_speed_from_center_motion():
    tracker = SimpleIouTracker(iou_threshold=0.2)

    tracker.update([detection()], timestamp_sec=0.0)
    second = tracker.update([detection(20, 10, 60, 50)], timestamp_sec=2.0)

    assert second[0].speed_px_per_sec == 5.0
