from illegal_parking.event_models import BBox, TrackSnapshot
from illegal_parking.violation_engine import ViolationEngine, ViolationEngineConfig


def make_track(**overrides):
    values = {
        "track_id": 7,
        "class_name": "car",
        "confidence": 0.86,
        "bbox": BBox(100, 220, 320, 520),
        "first_seen_ts": 0.0,
        "last_seen_ts": 35.0,
        "speed_px_per_sec": 0.03,
        "stable_frame_count": 12,
    }
    values.update(overrides)
    return TrackSnapshot(**values)


def test_candidate_requires_restricted_zone_evidence():
    engine = ViolationEngine()

    candidate = engine.evaluate(
        make_track(),
        redline_overlap_ratio=0.0,
        in_no_parking_roi=False,
    )

    assert candidate.status == "OBSERVE"
    assert not candidate.is_candidate
    assert "no restricted-zone evidence" in candidate.reasons


def test_candidate_is_created_for_stationary_vehicle_on_redline():
    engine = ViolationEngine()

    candidate = engine.evaluate(
        make_track(),
        redline_overlap_ratio=0.04,
        in_no_parking_roi=False,
    )

    assert candidate.status == "CANDIDATE"
    assert candidate.is_candidate
    assert candidate.rule_evidence.dwell_time_sec == 35.0
    assert candidate.rule_evidence.redline_overlap_ratio == 0.04
    assert "redline overlap" in candidate.reasons
    assert "dwell time threshold met" in candidate.reasons


def test_short_dwell_time_stays_observing_even_inside_roi():
    engine = ViolationEngine()

    candidate = engine.evaluate(
        make_track(last_seen_ts=8.0),
        redline_overlap_ratio=0.0,
        in_no_parking_roi=True,
    )

    assert candidate.status == "OBSERVE"
    assert not candidate.is_candidate
    assert candidate.rule_evidence.dwell_time_sec == 8.0


def test_low_confidence_vehicle_is_not_an_event():
    engine = ViolationEngine()

    candidate = engine.evaluate(
        make_track(confidence=0.1),
        redline_overlap_ratio=0.05,
        in_no_parking_roi=True,
    )

    assert candidate.status == "NO_EVENT"
    assert not candidate.is_candidate
    assert "confidence below threshold" in candidate.reasons


def test_unstable_track_is_observed_until_enough_frames():
    engine = ViolationEngine(ViolationEngineConfig(min_stable_frames=5))

    candidate = engine.evaluate(
        make_track(stable_frame_count=2),
        redline_overlap_ratio=0.05,
        in_no_parking_roi=True,
    )

    assert candidate.status == "OBSERVE"
    assert not candidate.is_candidate
    assert "track not stable yet" in candidate.reasons
