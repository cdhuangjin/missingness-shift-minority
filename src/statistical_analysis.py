"""Statistical validation for Phase E1/E2."""

from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    values,
    level: float = 0.95,
    iters: int = 2000,
    seed: int = 0,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Percentile bootstrap CI over the values (bootstrap unit = one value)."""
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    n = arr.shape[0]
    if n == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    resample = arr[rng.integers(0, n, size=(iters, n))]
    if statistic == "mean":
        stat = resample.mean(axis=1)
    elif statistic == "median":
        stat = np.median(resample, axis=1)
    else:
        raise ValueError(statistic)
    alpha = 1.0 - level
    return (float(np.percentile(stat, 100 * alpha / 2)),
            float(np.percentile(stat, 100 * (1 - alpha / 2))))


def wilcoxon_paired(a, b):
    """Wilcoxon signed-rank test on paired a vs b. Returns (stat, p)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 5:
        return (np.nan, np.nan)
    try:
        stat, p = stats.wilcoxon(a, b)
        return (float(stat), float(p))
    except ValueError:
        return (np.nan, np.nan)


def paired_ttest(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    a, b = a[mask], b[mask]
    if len(a) < 3:
        return (np.nan, np.nan)
    t, p = stats.ttest_rel(a, b)
    return (float(t), float(p))


def rank_biserial(a, b):
    """Rank-biserial correlation for paired differences (a vs b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = b - a
    d = d[np.isfinite(d)]
    if len(d) == 0:
        return np.nan
    absd = np.abs(d)
    ranks = stats.rankdata(absd)
    w_plus = float(np.sum(ranks[d > 0]))
    w_minus = float(np.sum(ranks[d < 0]))
    total = w_plus + w_minus
    if total == 0:
        return 0.0
    return (w_plus - w_minus) / total


def cohens_dz(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = (b - a)[np.isfinite(a) & np.isfinite(b)]
    if len(d) < 2:
        return np.nan
    sd = np.std(d, ddof=1)
    if sd == 0:
        return 0.0
    return float(d.mean() / sd)


def effect_size(a, b, method: str = "rank_biserial"):
    if method == "rank_biserial":
        return rank_biserial(a, b)
    if method == "cohens_dz":
        return cohens_dz(a, b)
    raise ValueError(method)


def bh_fdr(pvals, alpha: float = 0.05):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.shape[0]
    if m == 0:
        return np.array([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = ranked * m / np.arange(1, m + 1)
    # enforce monotonicity
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(p)
    out[order] = adjusted
    return out
