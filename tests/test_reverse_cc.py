"""Reverse class-conditional control: exposure flips to the majority."""

import numpy as np
import pandas as pd

from src.missingness_generator import (
    apply_mask,
    generate_class_conditional,
    measure_missingness,
)


def _xy():
    rng = np.random.default_rng(4)
    X = pd.DataFrame(rng.normal(size=(1200, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.where(np.arange(1200) < 180, 1, 0))  # 15% minority
    return X, y


def _rate(minority_rate, majority_rate):
    X, y = _xy()
    cols = X.columns.tolist()
    mask = generate_class_conditional(X, cols, y, minority_rate, majority_rate,
                                      np.random.default_rng(42))
    meas = measure_missingness(X, apply_mask(X, mask), y, cols=cols)
    return meas


def test_normal_exposes_minority_more():
    m = _rate(0.30, 0.10)
    assert m["minority_missing_rate"] > m["majority_missing_rate"]
    assert 0.25 <= m["minority_missing_rate"] <= 0.35
    assert 0.06 <= m["majority_missing_rate"] <= 0.14


def test_reverse_exposes_majority_more():
    m = _rate(0.10, 0.30)
    assert m["majority_missing_rate"] > m["minority_missing_rate"]
    assert 0.25 <= m["majority_missing_rate"] <= 0.35
    assert 0.06 <= m["minority_missing_rate"] <= 0.14
