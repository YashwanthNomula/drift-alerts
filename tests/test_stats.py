"""Tests for PSI / KS statistics."""

from drift_alerts import stats


def test_psi_identical_distributions_near_zero():
    vals = [float(i) for i in range(100)]
    psi, _, _, _ = stats.psi_numeric(vals, vals)
    assert psi < 1e-9


def test_psi_detects_shift():
    ref = [float(i) for i in range(200)]
    cur = [float(i + 100) for i in range(200)]
    psi, _, _, _ = stats.psi_numeric(ref, cur)
    assert psi > stats.PSI_MODERATE


def test_psi_constant_feature_no_crash():
    # Same as driftwatch: the constant value stays as an interior edge and
    # every value lands in a single bin via bisect_right.
    psi, edges, ref_counts, _ = stats.psi_numeric([5.0] * 50, [5.0] * 50)
    assert edges == [float("-inf"), 5.0, float("inf")]
    assert ref_counts == [0, 50]
    assert psi < 1e-9


def test_psi_unseen_category_stays_finite():
    import math
    psi, cats, _, _ = stats.psi_categorical(["a"] * 90 + ["b"] * 10,
                                            ["a"] * 50 + ["c"] * 50)
    assert math.isfinite(psi)
    assert psi > stats.PSI_MODERATE
    assert "c" in cats


def test_ks_identical_is_zero():
    assert stats.ks_2samp([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == 0.0


def test_ks_disjoint_is_one():
    assert stats.ks_2samp([1.0, 2.0], [10.0, 20.0]) == 1.0


def test_ks_partial_shift():
    assert 0.0 < stats.ks_2samp([1.0, 2.0, 3.0, 4.0],
                                [1.0, 2.0, 30.0, 40.0]) < 1.0


def test_ks_empty_returns_zero():
    assert stats.ks_2samp([], [1.0]) == 0.0


def test_verdict_boundaries():
    assert stats.psi_verdict(0.05) == "no significant change"
    assert stats.psi_verdict(0.10) == "moderate shift — watch"
    assert stats.psi_verdict(0.25) == "moderate shift — watch"
    assert stats.psi_verdict(0.26) == "significant drift"
    assert stats.ks_verdict(0.31) == "significant drift"


def test_worst_verdict_picks_most_severe():
    assert stats.worst_verdict("no significant change",
                               "moderate shift — watch") == "moderate shift — watch"
    assert stats.worst_verdict("moderate shift — watch",
                               "significant drift") == "significant drift"
