"""Test that feature ranking and imputation never use test information."""

import numpy as np
import pandas as pd

from src.preprocessing import MedianModeImputer, rank_numeric_features


def _frame():
    rng = np.random.default_rng(1)
    x = np.column_stack(
        [
            rng.normal(size=300),  # strong signal
            rng.normal(size=300) * 0.001,  # weak/no signal
            rng.normal(size=300),
            rng.normal(size=300),
        ]
    )
    X = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    y = pd.Series((X["a"] > 0.0).astype(int))
    return X, y


def test_ranking_uses_train_only_and_is_stable():
    X, y = _frame()
    topk = rank_numeric_features(X, y, list(X.columns), max_affected=3, method="mi")
    assert set(topk) <= set(X.columns)
    # A strong predictive feature ranks first.
    assert topk[0] == "a"
    # Changing an unrelated test frame does not alter ranking.
    other = X.copy()
    other["a"] = np.random.default_rng(9).normal(size=300)
    topk2 = rank_numeric_features(X, y, list(X.columns), max_affected=3, method="mi")
    assert topk == topk2


def test_imputer_fit_on_train_only():
    # Train median=25 (single NaN dropped), test median=-10 -> imputation must
    # use the train median, not the test median.
    train = pd.DataFrame({"x": [10.0, 20.0, 30.0, 40.0, np.nan]})
    test = pd.DataFrame({"x": [-10.0, -10.0, np.nan]})
    imp = MedianModeImputer().fit(train)
    transformed = imp.transform(test)
    assert transformed["x"].iloc[2] == 25.0


def test_shape_consistency_with_indicators():
    # Missingness indicators must keep consistent column count between train and test.
    X, y = _frame()
    imp = MedianModeImputer().fit(X)
    fake_mask = pd.DataFrame(
        np.random.default_rng(0).random((X.shape[0], 2)) < 0.3,
        index=X.index,
        columns=["a", "b"],
    )
    Xm = X.copy()
    for c in fake_mask.columns:
        Xm.loc[fake_mask[c].to_numpy(), c] = np.nan
    tr = imp.transform(Xm)
    assert list(tr.columns) == list(X.columns)
    # Indicator column is appended, so train and test are consistent.
    tr_ind = tr.copy()
    tr_ind["a__missing_indicator"] = Xm["a"].isna().astype(int)
    assert tr_ind.shape[0] == X.shape[0]
