"""Phase E1/E2 experimental runner (checkpointed, tier-based)."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from .config import Config
from .metrics import compute_metrics, threshold_05
from .missingness_generator import (
    apply_mask,
    generate_block,
    generate_class_conditional,
    generate_feature_dependent,
    generate_feature_specific,
    generate_mcar,
    measure_missingness,
    pick_driver_col,
)
from .models import make_model
from .preprocessing import (
    MedianModeImputer,
    apply_categorical_encoders,
    fit_categorical_encoders,
    rank_numeric_features,
    stratified_split,
)
from .resume_manager import ResumeManager


def rank_fsets(X, y, numeric_cols, cfg) -> dict[str, list[str]]:
    if not numeric_cols:
        return {"top": [], "random": [], "low": []}
    all_ranked = rank_numeric_features(
        X, y, numeric_cols, len(numeric_cols), cfg.features["rank_method"]
    )
    k = min(cfg.features["max_affected"], max(2, len(numeric_cols) // 4))
    top = all_ranked[:k]
    rng = np.random.default_rng(0)
    random_k = rng.choice(numeric_cols, size=min(k, len(numeric_cols)), replace=False).tolist()
    low = all_ranked[-k:] if k <= len(all_ranked) else all_ranked
    return {"top": top, "random": random_k, "low": low}


def build_environments(X_test, y_test, feat_cols, driver, seed, cfg) -> dict[str, pd.DataFrame]:
    env_masks: dict[str, pd.DataFrame] = {}

    def rng(env: str) -> np.random.Generator:
        h = int(hashlib.md5(env.encode()).hexdigest(), 16)
        return np.random.default_rng(seed * 100_003 + (h % 10_007))

    def cc_env(cname: str) -> str:
        # config key 'cc30' -> canonical 'cc_30'
        return f"cc_{cname[2:]}" if cname.startswith("cc") else cname

    for rate in cfg.mcar_rates:
        name = f"mcar_{int(round(rate * 100)):02d}"
        env_masks[name] = generate_mcar(X_test, feat_cols, float(rate), rng(name))

    for mname, rate in cfg.mar.items():
        env_masks[f"mar_{mname}"] = generate_feature_dependent(
            X_test, feat_cols, driver, float(rate), 2.0, rng(mname)
        )

    cc_rates = {}
    for cname, cc in cfg.class_conditional.items():
        env_name = cc_env(cname)
        mask = generate_class_conditional(
            X_test, feat_cols, y_test, float(cc["minority"]), float(cc["majority"]), rng(cname)
        )
        env_masks[env_name] = mask
        meas = measure_missingness(X_test, apply_mask(X_test, mask), y_test, cols=feat_cols)
        cc_rates[env_name] = meas["overall_missing_rate"]

    for cc_env_name, rate in cc_rates.items():
        env_masks[f"matched_mcar_{cc_env_name.replace('_', '')}"] = generate_mcar(
            X_test, feat_cols, float(rate), rng(f"matched_{cc_env_name}")
        )

    for rname, rcc in cfg.reverse.items():
        rev_env = f"reverse_cc_{rname.replace('reverse', '')}"
        env_masks[rev_env] = generate_class_conditional(
            X_test, feat_cols, y_test, float(rcc["minority"]), float(rcc["majority"]), rng(rname)
        )

    for bname, brate in cfg.block.items():
        env_masks[f"block_{bname}"] = generate_block(X_test, feat_cols, float(brate), rng(bname))

    env_masks["feature_specific"] = generate_feature_specific(
        X_test, feat_cols, cfg.feature_specific["rates"], rng("feature_specific")
    )
    return env_masks


def _encode(X, encoders):
    if encoders:
        return apply_categorical_encoders(X, encoders)
    return X


def prepare_imputer(X_train_src, categorical_cols, imputer_name):
    encoders = fit_categorical_encoders(X_train_src, categorical_cols)
    if imputer_name == "median":
        imputer = MedianModeImputer().fit(X_train_src)
    elif imputer_name in ("knn", "iterative"):
        try:
            from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        except ImportError:  # pragma: no cover
            pass
        from sklearn.impute import IterativeImputer, KNNImputer

        imputer = KNNImputer(n_neighbors=5) if imputer_name == "knn" else IterativeImputer(max_iter=10)
        # Fit on the same (encoded) numeric frame that transform_features will use.
        train_enc = apply_categorical_encoders(X_train_src, encoders).astype(float)
        imputer.fit(train_enc.to_numpy())
    else:
        imputer = None
    return encoders, imputer


def transform_features(X_masked, encoders, imputer, imputer_name, indicator_cols, pipeline):
    if imputer_name == "median" and pipeline != "native_missing":
        out = imputer.transform(X_masked)
        out = _encode(out, encoders)
    elif imputer_name in ("knn", "iterative"):
        base = _encode(X_masked, encoders).astype(float)
        out = pd.DataFrame(
            imputer.transform(base.to_numpy()), columns=base.columns, index=base.index
        )
    else:
        out = _encode(X_masked, encoders)
    if pipeline == "p1":
        for col in indicator_cols:
            out[f"{col}__missing_indicator"] = X_masked[col].isna().astype(int)
    return out


def fit_scaler(matrix: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    s = StandardScaler()
    num = matrix.select_dtypes(include=[np.number]).columns
    s.fit(matrix[num].to_numpy())
    return s


def apply_scaler(matrix: pd.DataFrame, scaler):
    num = matrix.select_dtypes(include=[np.number]).columns
    out = matrix.copy()
    out[num] = scaler.transform(out[num].to_numpy())
    return out


def run_one_unit(
    ds,
    seed,
    model,
    pipeline,
    imputer_name,
    fset,
    feat_cols,
    env_names,
    cfg,
) -> tuple[list[dict], list[dict]]:
    X, y = ds["X"], ds["y"]
    numeric_cols = ds["numeric_cols"]
    categorical_cols = ds["categorical_cols"]
    train_idx, _, test_idx = stratified_split(X, y, cfg.split, seed)
    X_train_raw = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    X_test_raw = X.iloc[test_idx]
    y_test = y.iloc[test_idx]
    driver = pick_driver_col(X_test_raw, feat_cols, numeric_cols)

    src_rng = np.random.default_rng(seed * 100_003 + 1)
    train_mask = generate_mcar(X_train_raw, feat_cols, cfg.source_missingness["rate"], src_rng)
    X_train_src = apply_mask(X_train_raw, train_mask)

    encoders, imputer = prepare_imputer(X_train_src, categorical_cols, imputer_name)
    X_tr = transform_features(
        X_train_src, encoders, imputer, imputer_name, feat_cols, pipeline
    )
    scaler = None
    if model == "logistic_regression" and pipeline != "native_missing":
        scaler = fit_scaler(X_tr)
        X_tr = apply_scaler(X_tr, scaler)

    estimator = make_model(model, seed)
    estimator.fit(X_tr, y_train)

    env_masks = build_environments(X_test_raw, y_test, feat_cols, driver, seed, cfg)
    raw_rows, missing_rows = [], []
    for env_name in env_names:
        if env_name not in env_masks:
            continue
        X_test_masked = apply_mask(X_test_raw, env_masks[env_name])
        X_te = transform_features(
            X_test_masked, encoders, imputer, imputer_name, feat_cols, pipeline
        )
        if scaler is not None:
            X_te = apply_scaler(X_te, scaler)
        proba = estimator.predict_proba(X_te)[:, 1]
        pred = threshold_05(proba)
        metrics = compute_metrics(y_test, proba, pred)
        meas = measure_missingness(X_test_raw, X_test_masked, y_test, cols=feat_cols)
        raw_rows.append(
            {
                "dataset": ds["name"],
                "seed": seed,
                "model": model,
                "pipeline": pipeline,
                "imputer": imputer_name,
                "feature_set": fset,
                "environment": env_name,
                "missing_rate": meas["overall_missing_rate"],
                **metrics,
            }
        )
        missing_rows.append(
            {
                "dataset": ds["name"],
                "seed": seed,
                "environment": env_name,
                "feature_set": fset,
                "overall_missing_rate": meas["overall_missing_rate"],
                "minority_missing_rate": meas["minority_missing_rate"],
                "majority_missing_rate": meas["majority_missing_rate"],
                "top_features_affected": ",".join(feat_cols),
            }
        )
    return raw_rows, missing_rows
