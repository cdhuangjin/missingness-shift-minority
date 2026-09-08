"""Exposure score and minority weighting unit tests."""

import numpy as np
import pandas as pd

from src.exposure import (
    audit_exposure,
    base_class_weights,
    compute_exposure_scores,
    compute_sample_weights,
    normalise_exposure_unit_mean,
)


def _xy(n=400, seed=11):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.where(np.arange(n) % 4 == 0, 1, 0))
    return X, y


def test_exposure_shape_and_finite():
    X, y = _xy()
    exp, q, top, _ = compute_exposure_scores(X, y, X.columns.tolist(), 5)
    assert exp.shape == (len(y),)
    assert np.isfinite(exp).all()
    assert len(top) == 5
    assert abs(sum(q.values()) - 1.0) < 1e-6


def test_exposure_bounded_zero_one():
    X, y = _xy()
    exp, _, _, _ = compute_exposure_scores(X, y, X.columns.tolist(), 5)
    assert exp.min() >= 0.0
    assert exp.max() <= 1.0


def test_exposure_has_spread():
    X, y = _xy()
    exp, _, _, _ = compute_exposure_scores(X, y, X.columns.tolist(), 5)
    assert exp.std() > 0.05


def test_minority_weight_greater_than_base_when_exposure_positive():
    X, y = _xy()
    exp, _, _, _ = compute_exposure_scores(X, y, X.columns.tolist(), 5)
    exp_norm = normalise_exposure_unit_mean(exp, y)
    w = compute_sample_weights(y, exp_norm, lambda_=1.0, mode="exposure")
    b = base_class_weights(y)
    minority = (y.to_numpy() == 1)
    assert (w[minority] > b[minority]).all()


def test_majority_weight_unchanged():
    X, y = _xy()
    exp, _, _, _ = compute_exposure_scores(X, y, X.columns.tolist(), 5)
    exp_norm = normalise_exposure_unit_mean(exp, y)
    w = compute_sample_weights(y, exp_norm, lambda_=1.0, mode="exposure")
    b = base_class_weights(y)
    majority = (y.to_numpy() == 0)
    assert np.allclose(w[majority], b[majority])


def test_uniform_minority_weight_constant():
    X, y = _xy()
    w = compute_sample_weights(y, np.ones(len(y)), lambda_=1.0, mode="uniform")
    b = base_class_weights(y)
    minority = (y.to_numpy() == 1)
    # same average extra intensity as exposure mode (e mean == 1 after normalise).
    assert np.allclose(w[minority], b[minority] * 2.0)


def test_exposure_and_uniform_match_intensity():
    X, y = _xy()
    exp, _, _, _ = compute_exposure_scores(X, y, X.columns.tolist(), 5)
    exp_norm = normalise_exposure_unit_mean(exp, y)
    w_exp = compute_sample_weights(y, exp_norm, lambda_=1.0, mode="exposure")
    w_uni = compute_sample_weights(y, exp_norm, lambda_=1.0, mode="uniform")
    minority = (y.to_numpy() == 1)
    # mean of (1 + lambda * e) over minority == 1 + lambda when mean(e)=1.
    assert np.isclose(w_exp[minority].mean(), w_uni[minority].mean(), atol=1e-9)


def test_audit_exposure_reporting():
    X, y = _xy()
    exp, _, _, r = compute_exposure_scores(X, y, X.columns.tolist(), 5)
    aud = audit_exposure(exp, y, r)
    assert aud["n_nan"] == 0
    assert aud["n_inf"] == 0
    assert 0.0 <= aud["exposure_min"] <= aud["exposure_max"] <= 1.0
    assert aud["top10_pct_minority_magnitude_mean"] >= 0.0
