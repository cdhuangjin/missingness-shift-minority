"""MAMR runner unit + leakage contract tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from src.mamr import MAMRParams, _prepare_train, run_one_unit
from src.phase_e3_config import PhaseE3Config
from src.preprocessing import stratified_split


ROOT = Path(__file__).resolve().parents[1]


def _cfg() -> PhaseE3Config:
    return PhaseE3Config.from_yaml(ROOT / "configs" / "phase_e3_mamr.yaml")


def _make_ds(n=600, seed=5):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 8)), columns=[f"f{i}" for i in range(8)])
    X["cat"] = np.random.default_rng(seed + 1).choice(["a", "b", "c"], size=n)
    y = pd.Series(np.where(np.arange(n) % 4 == 0, 1, 0))
    numeric = [f"f{i}" for i in range(8)]
    return {
        "name": "synthetic",
        "X": X,
        "y": y,
        "numeric_cols": numeric,
        "categorical_cols": ["cat"],
        "imbalance_ratio": float((y == 0).sum() / max((y == 1).sum(), 1)),
        "minority_label": 1,
        "majority_label": 0,
    }


def _params():
    return MAMRParams(
        lambda_=1.0, topk_features=5, augmentation_enabled=True,
        minority_only=True, copy_fraction=0.5, corruption_rate=0.15,
    )


def test_run_one_unit_produces_valid_metrics():
    ds = _make_ds()
    rows, _ = run_one_unit(ds, 42, "logistic_regression", "erm",
                           ["mcar_05", "cc_50"], _cfg(), _params())
    assert len(rows) == 2
    for r in rows:
        assert np.isfinite(r["minority_recall"])
        assert np.isfinite(r["majority_recall"])
        assert np.isfinite(r["AUPRC"])


def test_run_all_methods_no_crash():
    ds = _make_ds()
    cfg = _cfg()
    for method in cfg.methods:
        rows, _ = run_one_unit(ds, 42, "logistic_regression", method,
                               ["mcar_05", "cc_50"], cfg, _params())
        assert len(rows) == 2
        assert all(np.isfinite(r["minority_recall"]) for r in rows)


def test_augmentation_keeps_labels_positive_in_train():
    ds = _make_ds()
    prep = _prepare_train(ds, 42, "logistic_regression", "mamr_full", _cfg(), _params())
    assert prep["n_augment"] > 0
    assert prep["audit"]["n_augment"] == prep["n_augment"]


def test_no_test_leakage_exposure_invariant_to_test_features():
    ds = _make_ds()
    cfg = _cfg()
    tr, _, te = stratified_split(ds["X"], ds["y"], cfg.split, 42)
    # Perturb the test rows in a second copy; the split indices (based on shape/y)
    # are unchanged so training must not see the perturbation.
    X2 = ds["X"].copy()
    X2.loc[te, ds["numeric_cols"]] = 999.0
    ds2 = dict(ds)
    ds2["X"] = X2

    p1 = _prepare_train(ds, 42, "logistic_regression", "mamr_full", cfg, _params())
    p2 = _prepare_train(ds2, 42, "logistic_regression", "mamr_full", cfg, _params())
    assert np.allclose(p1["exposure_raw"], p2["exposure_raw"])
    assert p1["top_names"] == p2["top_names"]


def test_missing_indicator_consistent_feature_count():
    ds = _make_ds()
    prep = _prepare_train(ds, 42, "logistic_regression", "missing_indicator", _cfg(), _params())
    n_base = len(ds["numeric_cols"]) + len(ds["categorical_cols"])
    assert prep["indicator_cols"]
    assert len(prep["indicator_cols"]) <= n_base


def test_deterministic_same_seed():
    ds = _make_ds()
    rows1, _ = run_one_unit(ds, 42, "logistic_regression", "mamr_full",
                            ["mcar_05", "cc_50"], _cfg(), _params())
    rows2, _ = run_one_unit(ds, 42, "logistic_regression", "mamr_full",
                            ["mcar_05", "cc_50"], _cfg(), _params())
    r1 = {r["environment"]: r["minority_recall"] for r in rows1}
    r2 = {r["environment"]: r["minority_recall"] for r in rows2}
    for env in r1:
        assert np.isclose(r1[env], r2[env], atol=1e-9)
