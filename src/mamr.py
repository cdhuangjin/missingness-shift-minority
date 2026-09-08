"""MAMR method orchestration and the Phase E3 single-unit runner.

This is the model-agnostic MAMR plugin used by the Method Gate. Every method
shares the exact same stratified split, source missingness (MCAR 5%), imputer
(median for numeric, mode for categorical, fitted on source train only) and
evaluation protocol; the methods differ only in the training-side strategy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .augmentation import augment_minority
from .exposure import (
    audit_exposure,
    base_class_weights,
    compute_exposure_scores,
    compute_sample_weights,
    normalise_exposure_unit_mean,
)
from .metrics import compute_metrics, threshold_05
from .missingness_generator import (
    apply_mask,
    generate_block,
    generate_class_conditional,
    generate_feature_specific,
    generate_mcar,
    measure_missingness,
)
from .models import make_model
from .phase_e12_runner import apply_scaler, fit_scaler, rank_fsets
from .preprocessing import (
    MedianModeImputer,
    apply_categorical_encoders,
    fit_categorical_encoders,
    stratified_split,
)


@dataclass
class MAMRParams:
    lambda_: float = 1.0
    topk_features: int = 5
    augmentation_enabled: bool = True
    minority_only: bool = True
    copy_fraction: float = 0.5
    corruption_rate: float = 0.15
    majority_corruption_rate: float = 0.05
    mechanism_mix: dict[str, float] = field(
        default_factory=lambda: {"mcar": 0.4, "feature_specific": 0.3, "block": 0.3}
    )


def _env_rng(env: str, seed: int) -> np.random.Generator:
    h = int(hashlib.md5(env.encode()).hexdigest(), 16)
    return np.random.default_rng(seed * 100_003 + (h % 10_007))


def build_environments(
    X_test: pd.DataFrame,
    feat_cols: list[str],
    y_test: pd.Series,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Build the 8 frozen Phase E3 test environments."""
    env_masks: dict[str, pd.DataFrame] = {}
    env_masks["mcar_05"] = generate_mcar(X_test, feat_cols, 0.05, _env_rng("mcar_05", seed))
    env_masks["mcar_30"] = generate_mcar(X_test, feat_cols, 0.30, _env_rng("mcar_30", seed))
    env_masks["mcar_40"] = generate_mcar(X_test, feat_cols, 0.40, _env_rng("mcar_40", seed))
    env_masks["feature_specific"] = generate_feature_specific(
        X_test, feat_cols, [0.40, 0.30, 0.20, 0.10], _env_rng("feature_specific", seed)
    )
    env_masks["block_severe"] = generate_block(
        X_test, feat_cols, 0.30, _env_rng("block_severe", seed)
    )
    env_masks["cc_40"] = generate_class_conditional(
        X_test, feat_cols, y_test, 0.40, 0.10, _env_rng("cc_40", seed)
    )
    env_masks["cc_50"] = generate_class_conditional(
        X_test, feat_cols, y_test, 0.50, 0.10, _env_rng("cc_50", seed)
    )
    env_masks["reverse_cc_40"] = generate_class_conditional(
        X_test, feat_cols, y_test, 0.10, 0.40, _env_rng("reverse_cc_40", seed)
    )
    return env_masks


def _encode(X: pd.DataFrame, encoders: dict[str, dict]) -> pd.DataFrame:
    return apply_categorical_encoders(X, encoders) if encoders else X.copy()


def _transform(X_masked: pd.DataFrame, prep: dict[str, Any]) -> pd.DataFrame:
    out = prep["imputer"].transform(X_masked)
    out = _encode(out, prep["encoders"])
    if prep["indicator_cols"]:
        for col in prep["indicator_cols"]:
            if col in X_masked.columns:
                out[f"{col}__missing_indicator"] = X_masked[col].isna().astype(int)
    if prep["scaler"] is not None:
        out = apply_scaler(out, prep["scaler"])
    return out


def _weight_expand(sw: np.ndarray, aug_pos: np.ndarray | None) -> np.ndarray:
    if aug_pos is None or aug_pos.size == 0:
        return sw.astype(float)
    return np.concatenate([sw.astype(float), sw[aug_pos].astype(float)])


def _prepare_train(
    ds: dict[str, Any],
    seed: int,
    model: str,
    method: str,
    cfg: Any,
    params: MAMRParams,
    fit_model: bool = True,
) -> dict[str, Any]:
    X, y = ds["X"], ds["y"]
    numeric_cols = ds["numeric_cols"]
    categorical_cols = ds["categorical_cols"]
    train_idx, _, _ = stratified_split(X, y, cfg.split, seed)
    X_train_raw = X.iloc[train_idx]
    y_train = y.iloc[train_idx]
    feat_cols = rank_fsets(X_train_raw, y_train, numeric_cols, cfg)["top"]

    src_rng = np.random.default_rng(seed * 100_003 + 1)
    train_mask = generate_mcar(
        X_train_raw, feat_cols, float(cfg.source_missingness["rate"]), src_rng
    )
    X_train_src = apply_mask(X_train_raw, train_mask)

    encoders = fit_categorical_encoders(X_train_src, categorical_cols)
    imputer = MedianModeImputer().fit(X_train_src)

    use_aug = method in ("augmentation_only", "mamr_full", "safe_mamr") and params.augmentation_enabled
    use_oversample = method == "random_oversampling"
    indicator_cols = list(feat_cols) if method == "missing_indicator" else []

    aug_pos: np.ndarray | None = None
    aug_raw: pd.DataFrame | None = None
    aug_y: pd.Series | None = None
    if use_aug:
        aug_rng = np.random.default_rng(seed * 100_003 + 7)
        aug_raw, aug_y, aug_pos = augment_minority(
            X_train_src,
            y_train,
            list(feat_cols),
            params.copy_fraction,
            params.corruption_rate,
            aug_rng,
            mechanism_mix=params.mechanism_mix,
            majority_corruption_rate=params.majority_corruption_rate,
            minority_only=params.minority_only,
        )

    ovs_raw: pd.DataFrame | None = None
    ovs_y: pd.Series | None = None
    if use_oversample:
        y_np = y_train.to_numpy()
        is_min = (y_np == 1)
        n_min = int(is_min.sum())
        n_maj = int((~is_min).sum())
        minority_idx = np.where(is_min)[0]
        if n_maj > n_min:
            n_extra = n_maj - n_min
            chosen = np.random.default_rng(seed * 100_003 + 11).choice(
                minority_idx, size=n_extra, replace=True
            )
            ovs_raw = X_train_src.iloc[chosen]
            ovs_y = y_train.iloc[chosen]

    # Impute / encode / scale the matrix actually passed to the model.
    parts = [X_train_src]
    y_parts = [y_train]
    if aug_raw is not None and aug_raw.shape[0]:
        parts.append(aug_raw)
        y_parts.append(aug_y)
    if ovs_raw is not None and ovs_raw.shape[0]:
        parts.append(ovs_raw)
        y_parts.append(ovs_y)
    X_full_raw = pd.concat(parts, axis=0, ignore_index=True)
    y_fit = pd.concat(y_parts, axis=0, ignore_index=True)

    # Original (non-augmented) encoded + scaled matrix, used for exposure and for
    # the base scaler. Everything is source-train only.
    X_tr_orig = _encode(imputer.transform(X_train_src), encoders)
    if indicator_cols:
        for col in indicator_cols:
            if col in X_train_src.columns:
                X_tr_orig[f"{col}__missing_indicator"] = X_train_src[col].isna().astype(int)
    scaler = None
    if model == "logistic_regression":
        scaler = fit_scaler(X_tr_orig)
        X_tr_orig_scaled = apply_scaler(X_tr_orig, scaler)
    else:
        X_tr_orig_scaled = X_tr_orig

    # Exposure proxy on the original source train.
    exposure_raw, q_dict, top_names, r_norm = compute_exposure_scores(
        X_tr_orig_scaled,
        y_train,
        numeric_cols,
        len(feat_cols),
        cfg.features["rank_method"],
        feature_set=list(feat_cols),
    )
    exposure_norm = normalise_exposure_unit_mean(exposure_raw, y_train)

    # Sample weights over the original source train, per method.
    base = base_class_weights(y_train, balanced=True)
    if method == "erm":
        sw_orig = np.ones(y_train.shape[0], dtype=float)
    elif method == "class_weight":
        sw_orig = base
    elif method in ("augmentation_only",):
        sw_orig = base
    elif method == "exposure_weighting_only":
        sw_orig = compute_sample_weights(y_train, exposure_norm, params.lambda_, "exposure")
    elif method == "uniform_minority_weight":
        sw_orig = compute_sample_weights(y_train, exposure_norm, params.lambda_, "uniform")
    elif method in ("mamr_full", "safe_mamr"):
        sw_orig = compute_sample_weights(y_train, exposure_norm, params.lambda_, "exposure")
    elif method in ("random_oversampling", "missing_indicator"):
        sw_orig = np.ones(y_train.shape[0], dtype=float)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Build the final fit matrix.
    encoded_full = _encode(imputer.transform(X_full_raw), encoders)
    if indicator_cols:
        for col in indicator_cols:
            if col in X_full_raw.columns:
                encoded_full[f"{col}__missing_indicator"] = X_full_raw[col].isna().astype(int)
    if scaler is not None:
        X_fit = apply_scaler(encoded_full, scaler)
    else:
        X_fit = encoded_full

    if use_aug and aug_pos is not None:
        sw_fit = _weight_expand(sw_orig, aug_pos)
    elif use_oversample and ovs_raw is not None and ovs_raw.shape[0]:
        sw_fit = np.ones(y_fit.shape[0], dtype=float)
    else:
        sw_fit = sw_orig

    estimator = None
    if fit_model:
        estimator = make_model(model, seed)
        estimator.fit(X_fit.to_numpy(), y_fit.to_numpy(), sample_weight=sw_fit)

    audit = audit_exposure(exposure_raw, y_train, r_norm)
    audit.update(
        {
            "method": method,
            "model": model,
            "seed": seed,
            "dataset": ds["name"],
            "top_features": ",".join(feat_cols),
            "n_source_train": int(y_train.shape[0]),
            "n_augment": int(aug_raw.shape[0]) if aug_raw is not None else 0,
            "lambda": float(params.lambda_),
        }
    )
    return {
        "estimator": estimator,
        "imputer": imputer,
        "encoders": encoders,
        "scaler": scaler,
        "indicator_cols": indicator_cols,
        "feat_cols": list(feat_cols),
        "exposure_raw": exposure_raw,
        "exposure_norm": exposure_norm,
        "audit": audit,
        "q_dict": q_dict,
        "top_names": top_names,
        "y_train": y_train,
        "n_augment": int(aug_raw.shape[0]) if aug_raw is not None else 0,
    }


def run_one_unit(
    ds: dict[str, Any],
    seed: int,
    model: str,
    method: str,
    env_names: list[str],
    cfg: Any,
    params: MAMRParams,
    eval_mode: str = "test",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run one (dataset, seed, model, method) unit.

    ``eval_mode='test'`` evaluates the frozen test environments (returns one row
    per environment). ``eval_mode='val'`` evaluates the source-domain validation
    split (one row, environment ``validation``) for the hyperparameter gate.
    """
    X, y = ds["X"], ds["y"]
    train_idx, val_idx, test_idx = stratified_split(X, y, cfg.split, seed)
    X_train_raw = X.iloc[train_idx]
    y_train = y.iloc[train_idx]

    prep = _prepare_train(ds, seed, model, method, cfg, params)
    feat_cols = prep["feat_cols"]

    rows: list[dict[str, Any]] = []
    if eval_mode == "val":
        X_val_raw = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        val_rng = np.random.default_rng(seed * 100_003 + 3)
        mask = generate_mcar(
            X_val_raw, feat_cols, float(cfg.source_missingness["rate"]), val_rng
        )
        X_val_masked = apply_mask(X_val_raw, mask)
        X_te = _transform(X_val_masked, prep)
        proba = prep["estimator"].predict_proba(X_te.to_numpy())[:, 1]
        pred = threshold_05(proba)
        metrics = compute_metrics(y_val, proba, pred)
        meas = measure_missingness(X_val_raw, X_val_masked, y_val, cols=feat_cols)
        rows.append(
            {
                "environment": "validation",
                "missing_rate": meas["overall_missing_rate"],
                **metrics,
            }
        )
    else:
        X_test_raw = X.iloc[test_idx]
        y_test = y.iloc[test_idx]
        env_masks = build_environments(X_test_raw, feat_cols, y_test, seed)
        for env_name in env_names:
            if env_name not in env_masks:
                continue
            X_test_masked = apply_mask(X_test_raw, env_masks[env_name])
            X_te = _transform(X_test_masked, prep)
            proba = prep["estimator"].predict_proba(X_te.to_numpy())[:, 1]
            pred = threshold_05(proba)
            metrics = compute_metrics(y_test, proba, pred)
            meas = measure_missingness(X_test_raw, X_test_masked, y_test, cols=feat_cols)
            rows.append(
                {
                    "environment": env_name,
                    "missing_rate": meas["overall_missing_rate"],
                    **metrics,
                }
            )

    return rows, prep["audit"]
