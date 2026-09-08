"""Verify actual missing rates match targets and ordering, without leakage."""

import numpy as np
import pandas as pd

from src.missingness_generator import (
    apply_mask,
    generate_mcar,
    measure_missingness,
)
from src.preprocessing import stratified_split


def _synthetic():
    rng = np.random.default_rng(0)
    X = pd.DataFrame(
        rng.normal(size=(400, 6)),
        columns=[f"f{i}" for i in range(6)],
    )
    # Imbalanced binary target (about 20% minority).
    y = pd.Series(np.where(np.arange(400) < 80, 1, 0))
    return X, y


def test_mcar_target_rates_and_order():
    X, y = _synthetic()
    cols = X.columns[:4].tolist()
    rates = {}
    for rate in (0.05, 0.10, 0.30):
        mask = generate_mcar(X, cols, rate, np.random.default_rng(7))
        masked = apply_mask(X, mask)
        meas = measure_missingness(X, masked, y, cols=cols)
        rates[rate] = meas["overall_missing_rate"]
        assert 0.03 <= meas["overall_missing_rate"] <= 0.33
    assert rates[0.30] > rates[0.10] > rates[0.05]


def test_sample_count_and_labels_unchanged():
    X, y = _synthetic()
    cols = X.columns[:3].tolist()
    mask = generate_mcar(X, cols, 0.30, np.random.default_rng(11))
    masked = apply_mask(X, mask)
    assert masked.shape[0] == X.shape[0]
    assert list(masked.columns) == list(X.columns)
    assert (masked.index == X.index).all()


def test_stratified_split_preserves_ratio():
    X, y = _synthetic()
    tr, va, te = stratified_split(X, y, {"train": 0.6, "validation": 0.2, "test": 0.2}, 42)
    assert len(tr) + len(va) + len(te) == X.shape[0]
    assert set(tr) & set(va) == set()
    assert set(tr) & set(te) == set()
    for idx, frac in zip((tr, va, te), (0.6, 0.2, 0.2)):
        ratio = y.iloc[idx].mean()
        assert abs(ratio - y.mean()) < 0.06
