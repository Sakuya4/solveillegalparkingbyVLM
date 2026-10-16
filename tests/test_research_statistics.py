import numpy as np
import pytest

from illegal_parking.research_statistics import paired_cluster_coverage_interval, paired_cluster_f1_interval, wilson_interval


def test_paired_f1_resamples_source_clips_and_preserves_identical_predictions():
    labels = np.array([0, 1, 0, 1])
    scores = np.array([0.1, 0.9, 0.1, 0.9])
    clips = np.array(["a", "a", "b", "b"])
    same = paired_cluster_f1_interval(labels, scores, scores, clips, threshold_a=0.5, threshold_b=0.5, repeats=20)
    assert same["difference"] == same["lower_95"] == same["upper_95"] == 0
    opposite = paired_cluster_f1_interval(labels, scores, 1 - scores, clips, threshold_a=0.5, threshold_b=0.5, repeats=20)
    assert opposite["difference"] == opposite["lower_95"] == opposite["upper_95"] == 1
    assert opposite["clusters"] == 2


def test_wilson_interval_matches_known_binomial_example():
    lower, upper = wilson_interval(75, 100)
    assert lower == pytest.approx(0.656955, abs=1e-5)
    assert upper == pytest.approx(0.824547, abs=1e-5)
    assert wilson_interval(0, 100)[0] == 0
    assert wilson_interval(100, 100)[1] == 1


def test_cluster_bootstrap_uses_paired_coverage_and_is_reproducible():
    totals = np.array([10, 20, 30])
    result = paired_cluster_coverage_interval(totals, totals * 0.5, totals * 0.25, repeats=100)
    assert result["difference"] == pytest.approx(0.25)
    assert result["lower_95"] == pytest.approx(0.25)
    assert result["upper_95"] == pytest.approx(0.25)
    assert result == paired_cluster_coverage_interval(totals, totals * 0.5, totals * 0.25, repeats=100)
    with pytest.raises(ValueError, match="Covered"):
        paired_cluster_coverage_interval(totals, totals + 1, totals)
