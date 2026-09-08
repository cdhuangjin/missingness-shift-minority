"""Feature-specific missingness: each feature gets its own missing rate."""

import numpy as np
import pandas as pd

from src.missingness_generator import apply_mask, generate_feature_specific, measure_missingness


def test_per_feature_rates_match_targets():
    rng = np.random.default_rng(8)
    X = pd.DataFrame(rng.normal(size=(2000, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.where(np.arange(2000) < 300, 1, 0))
    cols = X.columns[:4].tolist()
    rates = [0.40, 0.30, 0.20, 0.10]
    mask = generate_feature_specific(X, cols, rates, rng)
    masked = apply_mask(X, mask)
    meas = measure_missingness(X, masked, y, cols=cols)
    for col, target in zip(cols, rates):
        actual = meas["per_feature_missing_rate"][col]
        assert abs(actual - target) <= 0.04, (col, actual, target)


def test_response_matches_supplied_rates_vector():
    rng = np.random.default_rng(1)
    X = pd.DataFrame(np.zeros((500, 4)), columns=["a", "b", "c", "d"])
    y = pd.Series(np.where(np.arange(500) < 90, 1, 0))
    mask = generate_feature_specific(X, ["a", "b", "c", "d"], [0.5, 0.0, 1.0, 0.0], rng)
    masked = apply_mask(X, mask)
    assert masked["a"].isna().mean() > 0.4
    assert masked["b"].isna().mean() == 0.0
    assert masked["c"].isna().mean() > 0.9
    assert masked["d"].isna().mean() == 0.0
