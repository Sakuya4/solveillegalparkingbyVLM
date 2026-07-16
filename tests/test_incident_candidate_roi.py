from __future__ import annotations

import numpy as np
import pytest

from illegal_parking.incident_candidate_roi import (
    CandidateRegion,
    compute_candidate_motion_step,
    evaluate_candidate_proposals,
    fuse_candidate_regions,
    propose_motion_regions,
    propose_track_regions,
)
from illegal_parking.incident_trajectory import TrackBox


def _track(track_id: int, x1: float, x2: float, confidence: float = 0.9) -> TrackBox:
    return TrackBox(
        frame_index=1,
        track_id=track_id,
        x1=x1,
        y1=0.4,
        x2=x2,
        y2=0.7,
        confidence=confidence,
    )


def test_nearby_tracks_create_pair_candidate() -> None:
    candidates = propose_track_regions([_track(1, 0.2, 0.4), _track(2, 0.43, 0.63)])

    pair = next(candidate for candidate in candidates if candidate.source == "track_pair")

    assert pair.track_ids == (1, 2)
    assert pair.x1 < 0.2
    assert pair.x2 > 0.63
    assert 0.0 < pair.score <= 1.0


def test_far_tracks_do_not_create_pair_candidate() -> None:
    candidates = propose_track_regions([_track(1, 0.05, 0.15), _track(2, 0.85, 0.95)])

    assert all(candidate.source == "track" for candidate in candidates)


def test_frame_change_creates_normalized_motion_candidate() -> None:
    previous = np.zeros((80, 100, 3), dtype=np.uint8)
    current = previous.copy()
    current[30:55, 40:70] = 255

    candidates = propose_motion_regions(
        previous,
        current,
        change_threshold=20,
        min_area_ratio=0.01,
    )

    assert candidates
    candidate = candidates[0]
    assert candidate.source == "motion"
    assert candidate.x1 <= 0.4 < candidate.x2
    assert candidate.y1 <= 0.375 < candidate.y2


def test_overlapping_track_and_motion_candidates_are_fused() -> None:
    track = CandidateRegion(0.2, 0.2, 0.6, 0.7, score=0.7, source="track", track_ids=(4,))
    motion = CandidateRegion(0.3, 0.3, 0.7, 0.8, score=0.5, source="motion")

    candidates = fuse_candidate_regions([track], [motion], overlap_threshold=0.2)

    fused = candidates[0]
    assert fused.source == "track_motion"
    assert fused.track_ids == (4,)
    assert fused.x1 == pytest.approx(0.2)
    assert fused.x2 == pytest.approx(0.7)
    assert fused.score > track.score


def test_no_candidate_uses_explicit_global_fallback() -> None:
    previous = np.zeros((48, 64, 3), dtype=np.uint8)
    current = previous.copy()

    result = compute_candidate_motion_step(previous, current, [])

    assert result.selected is None
    assert result.features["candidate_available"] == 0.0
    assert result.features["candidate_score"] == 0.0
    assert result.features["candidate_diff_mean"] == result.features["global_diff_mean"]
    assert "ground_truth_iou" not in result.features


def test_ground_truth_box_is_used_only_by_proposal_diagnostics() -> None:
    proposals = [CandidateRegion(0.2, 0.2, 0.6, 0.6, score=0.8, source="track_motion")]

    diagnostics = evaluate_candidate_proposals(proposals, (0.25, 0.25, 0.55, 0.55))

    assert diagnostics.candidate_count == 1
    assert diagnostics.best_iou > 0.5
    assert diagnostics.hit_at_0_3 is True


@pytest.mark.parametrize(
    "coordinates",
    [(-0.1, 0.0, 0.5, 0.5), (0.5, 0.2, 0.5, 0.8), (0.2, 0.9, 0.8, 0.2)],
)
def test_candidate_region_rejects_invalid_coordinates(coordinates: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="normalized"):
        CandidateRegion(*coordinates, score=0.5, source="motion")
