import numpy as np
import pytest

from illegal_parking.research_statistics import paired_cluster_coverage_interval, wilson_interval


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
