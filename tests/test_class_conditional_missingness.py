"""Verify class-conditional missingness exposes the minority more."""

import numpy as np
import pandas as pd

from src.missingness_generator import (
    apply_mask,
    generate_class_conditional,
    measure_missingness,
)


def test_minority_exposed_more():
    rng = np.random.default_rng(3)
    X = pd.DataFrame(rng.normal(size=(1000, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series(np.where(np.arange(1000) < 150, 1, 0))  # 15% minority
    mask = generate_class_conditional(X, X.columns.tolist(), y, 0.30, 0.10, rng)
    masked = apply_mask(X, mask)
    meas = measure_missingness(X, masked, y)
    assert meas["minority_missing_rate"] > meas["majority_missing_rate"]
    assert 0.25 <= meas["minority_missing_rate"] <= 0.35
    assert 0.06 <= meas["majority_missing_rate"] <= 0.14


def test_class_conditional_uses_only_predefined_labels():
    # The generator only needs the label to place exposure; it must not alter it.
    X = pd.DataFrame(np.arange(100).reshape(100, 1), columns=["f0"])
    y = pd.Series(np.where(np.arange(100) < 20, 1, 0))
    mask = generate_class_conditional(X, ["f0"], y, 0.30, 0.10, np.random.default_rng(5))
    assert (mask.index == y.index).all()
    assert mask.shape[0] == 100
