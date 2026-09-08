"""Missingness sensitivity slope analysis."""

from __future__ import annotations

import numpy as np


def fit_slope(rates: list[float], metrics: list[float]) -> tuple[float, float]:
    """Fit metrics = a + b * rate over the MCAR environments."""
    if len(rates) != len(metrics) or len(rates) < 2:
        raise ValueError("Need at least two MCAR points for a slope.")
    (b, a) = np.polyfit(np.asarray(rates, dtype=float), np.asarray(metrics, dtype=float), 1)
    return float(b), float(a)


def slope_gap(minority_slope: float, majority_slope: float) -> float:
    """abs(minority_slope) - abs(majority_slope)."""
    return abs(minority_slope) - abs(majority_slope)


def slope_ratio(minority_slope: float, majority_slope: float) -> float:
    """abs(minority_slope) / abs(majority_slope)."""
    return abs(minority_slope) / (abs(majority_slope) + 1e-12)
