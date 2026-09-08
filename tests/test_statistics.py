"""Statistical helpers: CI, Wilcoxon, effect size, FDR."""

import numpy as np

from src.statistical_analysis import (bh_fdr, bootstrap_ci, cohens_dz, paired_ttest,
                                      rank_biserial, wilcoxon_paired)


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.2, size=200)
    lo, hi = bootstrap_ci(x, level=0.95, iters=1000)
    assert lo <= np.mean(x) <= hi
    assert lo < hi


def test_wilcoxon_and_paired_ttest_significant():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, size=30)
    b = a + 1.0  # b is shifted up, so paired differences are consistently positive.
    stat, p = wilcoxon_paired(a, b)
    t, pt = paired_ttest(a, b)
    assert p < 0.05
    assert pt < 0.05
    assert np.isfinite(stat)


def test_effect_size_sign_matches_direction():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 0.2, size=30)  # majority degradation
    b = a + 0.5  # minority degradation consistently bigger
    es = rank_biserial(a, b)
    assert es > 0.5
    dz = cohens_dz(a, b)
    assert dz > 0


def test_bh_fdr_monotonic_and_bounded():
    raw = np.array([0.001, 0.01, 0.03, 0.05, 0.20, 0.60])
    adj = bh_fdr(raw)
    assert np.all(adj >= raw)
    assert np.all(adj <= 1.0)
    assert np.all(adj[:-1] <= adj[1:] + 1e-12)


def test_fdr_keeps_smallest_conditionally_significant():
    raw = np.array([0.001, 0.008, 0.5, 0.6])
    adj = bh_fdr(raw)
    assert adj[0] < 0.05
    assert adj[1] < 0.05
    assert adj[2] > 0.05
