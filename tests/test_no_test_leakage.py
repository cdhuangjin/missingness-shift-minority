"""Leakage contract: split disjointness and train-only preprocessing."""

import numpy as np
import pandas as pd

from src.missingness_generator import generate_mcar
from src.preprocessing import MedianModeImputer, rank_numeric_features, stratified_split


def _xy():
    rng = np.random.default_rng(17)
    X = pd.DataFrame(rng.normal(size=(600, 8)), columns=[f"f{i}" for i in range(8)])
    # Force a confounded column so the train median differs from the test median.
    X["f0"] = np.where(np.arange(600) < 300, -5.0, 5.0)
    y = pd.Series(np.where(np.arange(600) < 120, 1, 0))
    return X, y


def test_stratified_split_is_disjoint_and_balanced():
    X, y = _xy()
    tr, va, te = stratified_split(X, y, {"train": 0.6, "validation": 0.2, "test": 0.2}, 42)
    assert set(tr).isdisjoint(va)
    assert set(tr).isdisjoint(te)
    assert set(va).isdisjoint(te)
    assert len(tr) + len(va) + len(te) == X.shape[0]
    assert abs(y.iloc[tr].mean() - y.mean()) < 0.06


def test_imputer_uses_train_median_not_test_median():
    X, y = _xy()
    tr, _, te = stratified_split(X, y, {"train": 0.6, "validation": 0.2, "test": 0.2}, 42)
    X_tr = X.iloc[tr].copy()
    X_te = X.iloc[te].copy()
    # Inject train-side missingness; the imputer learns a per-train-column value.
    mask_tr = generate_mcar(X_tr, X_tr.columns.tolist(), 0.3, np.random.default_rng(5))
    X_tr_masked = X_tr.mask(mask_tr)
    imputer = MedianModeImputer().fit(X_tr_masked)
    # Mask the test set too, so at least one f0 is missing and gets filled.
    mask_te = generate_mcar(X_te, X_te.columns.tolist(), 0.5, np.random.default_rng(6))
    masked_te = X_te.mask(mask_te)
    filled = imputer.transform(masked_te)
    train_median = float(X_tr_masked["f0"].median())
    missing_row = filled.loc[masked_te["f0"].isna(), "f0"]
    assert len(missing_row) > 0
    # The fill for a missing f0 must equal the train median, not the test median.
    assert np.allclose(missing_row.to_numpy(dtype=float), train_median, atol=1e-9)
    assert abs(train_median - float(X_te["f0"].median())) > 1.0


def test_ranking_needs_only_train_labels():
    X, y = _xy()
    tr, _, _ = stratified_split(X, y, {"train": 0.6, "validation": 0.2, "test": 0.2}, 42)
    ranked = rank_numeric_features(X.iloc[tr], y.iloc[tr], X.columns.tolist(), 5)
    assert len(ranked) > 0
    assert all(c in X.columns for c in ranked)


def test_missingness_generators_ignore_labels():
    X, y = _xy()
    # MAR-like generator only uses the driver column, never y.
    from src.missingness_generator import generate_feature_dependent

    mask = generate_feature_dependent(X, X.columns[:3].tolist(), X.columns[-1], 0.2, 2.0,
                                      np.random.default_rng(3))
    assert mask.shape == (X.shape[0], 3)
    assert list(mask.columns) == X.columns[:3].tolist()
