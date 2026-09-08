"""Exposure-aware minority weighting (MAMR component A).

The exposure score is a *training-side missingness exposure proxy*. It is
computed entirely from the source train so it can never depend on target test
labels, statistics or missingness profile. It must never be described as a
causal quantity (§27, §41).

Definition (prompt §5.2):

    e_i = sum_{j in S} q_j * r_ij / (sum_{j in S} q_j + eps)

where:
- ``S`` is the set of high-importance predictive features (top-k);
- ``q_j`` is the normalised mutual-information importance;
- ``r_ij`` is the clipped, normalised standardised feature magnitude.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif


EPS = 1e-12


def importance_weights(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    topk_features: int,
    rank_method: str = "mutual_information",
) -> tuple[list[str], dict[str, float], np.ndarray, np.ndarray]:
    """Return (topk_names, q_dict, q_array, raw_scores) over numeric features.

    ``q_j`` is the normalised importance restricted to the top-k set, so the
    weights sum to one over that set.
    """
    if not numeric_cols:
        return [], {}, np.array([], dtype=float), np.array([], dtype=float)

    sub = X[numeric_cols].astype(float)
    if sub.isna().any().any():
        # Complete-case view only; no test labels are involved.
        mask = sub.notna().all(axis=1).to_numpy()
        sub = sub.loc[mask]
        y_rank = y.loc[mask]
    else:
        y_rank = y

    if sub.shape[0] < 2 or y_rank.nunique() < 2:
        fallback = numeric_cols[: int(np.clip(topk_features, 1, len(numeric_cols)))]
        scores = np.ones(len(fallback))
        names = fallback
    elif rank_method in ("mutual_information", "mi"):
        scores = mutual_info_classif(sub.to_numpy(), y_rank.to_numpy(), random_state=0)
        names = list(numeric_cols)
    elif rank_method in ("xgboost_gain", "gain"):
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=60, max_depth=4, tree_method="hist",
            eval_metric="logloss", random_state=0,
        )
        model.fit(sub.to_numpy(), y_rank.to_numpy())
        scores = model.feature_importances_
        names = list(numeric_cols)
    else:
        raise ValueError(f"Unknown rank method: {rank_method}")

    order = np.argsort(-scores)
    k = int(min(topk_features, len(names)))
    top_names = [names[i] for i in order[:k]]
    top_scores = np.asarray([scores[i] for i in order[:k]], dtype=float)
    denom = float(top_scores.sum() + EPS)
    q = top_scores / denom
    q_dict = {name: float(q[i]) for i, name in enumerate(top_names)}
    return top_names, q_dict, q, top_scores


def standardise_feature_magnitude(X: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Robust z-score of ``cols`` computed on the given (train) frame only."""
    out = X[cols].astype(float).copy()
    for col in cols:
        s = out[col]
        mu = float(np.nanmean(s.to_numpy())) if np.isnan(s.to_numpy()).any() else float(s.mean())
        sd = float(np.nanstd(s.to_numpy())) if np.isnan(s.to_numpy()).any() else float(s.std())
        if not np.isfinite(sd) or sd < 1e-8:
            sd = 1.0
        out[col] = (s - mu) / sd
    return out


def compute_exposure_scores(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    topk_features: int,
    rank_method: str = "mutual_information",
    feature_set: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, float], list[str], np.ndarray]:
    """Compute the raw exposure score for every train sample.

    ``X`` is the model-ready (imputed + encoded, and for linear models scaled)
    source-train matrix. Returns (exposure, q_dict, top_names, r_norm).
    """
    if feature_set is not None and feature_set:
        top_names = list(feature_set)
        sub = X[top_names].astype(float)
        if sub.isna().any().any():
            mask = sub.notna().all(axis=1).to_numpy()
            sub2 = sub.loc[mask]
            y_rank = y.loc[mask]
        else:
            sub2 = sub
            y_rank = y
        if sub2.shape[0] < 2 or y_rank.nunique() < 2:
            scores = np.ones(len(top_names))
        elif rank_method in ("mutual_information", "mi"):
            scores = mutual_info_classif(sub2.to_numpy(), y_rank.to_numpy(), random_state=0)
        else:
            import xgboost as xgb

            model = xgb.XGBClassifier(
                n_estimators=60, max_depth=4, tree_method="hist",
                eval_metric="logloss", random_state=0,
            )
            model.fit(sub2.to_numpy(), y_rank.to_numpy())
            scores = model.feature_importances_
        denom = float(np.asarray(scores, dtype=float).sum() + EPS)
        q_array = np.asarray(scores, dtype=float) / denom
        q_dict = {name: float(q_array[i]) for i, name in enumerate(top_names)}
    else:
        top_names, q_dict, q_array, _ = importance_weights(
            X, y, numeric_cols, topk_features, rank_method
        )
    if not top_names:
        zeros = np.zeros(X.shape[0], dtype=float)
        return zeros, {}, [], zeros

    Z = standardise_feature_magnitude(X, top_names)
    # clip |z| to [0,3] then normalise to [0,1].
    r = np.clip(np.abs(Z.to_numpy()), 0.0, 3.0) / 3.0
    weighted = r @ q_array
    denom = float(q_array.sum() + EPS)
    exposure = weighted / denom
    return exposure, q_dict, top_names, r


def normalise_exposure_unit_mean(
    exposure: np.ndarray, y: pd.Series
) -> np.ndarray:
    """Scale the exposure proxy so the minority mean is exactly 1.

    This makes the mean weight intensity of ``exposure_weighting_only`` match
    ``uniform_minority_weight`` (whose ``e_i = 1``), so the ablation isolates the
    exposure *decomposition* rather than the overall minority uplift.
    """
    e = np.asarray(exposure, dtype=float)
    is_min = (np.asarray(y) == 1)
    minority = e[is_min]
    m = float(minority.mean()) if minority.size else 0.0
    if not np.isfinite(m) or m <= EPS:
        return np.ones_like(e)  # constant exposure -> abort signal handled upstream.
    out = e.copy()
    out[is_min] = e[is_min] / m
    return out


def base_class_weights(y: pd.Series, balanced: bool = True) -> np.ndarray:
    """Return the per-sample base class weight."""
    y_np = np.asarray(y)
    n = y_np.shape[0]
    n1 = float((y_np == 1).sum())
    n0 = float((y_np == 0).sum())
    if not balanced:
        return np.ones(n, dtype=float)
    w1 = n / (2.0 * n1) if n1 > 0 else 1.0
    w0 = n / (2.0 * n0) if n0 > 0 else 1.0
    out = np.where(y_np == 1, w1, w0).astype(float)
    return out


def compute_sample_weights(
    y: pd.Series,
    exposure_norm: np.ndarray,
    lambda_: float,
    mode: str,
    balanced_base: bool = True,
) -> np.ndarray:
    """Final sample weights for a given MAMR weighting mode.

    Modes:
    - ``none``        : balanced base only (class_weight-like).
    - ``exposure``    : base * (1 + lambda * e_i) for minority.
    - ``uniform``     : base * (1 + lambda) for minority (e_i = 1).
    """
    base = base_class_weights(y, balanced=balanced_base)
    y_np = np.asarray(y)
    is_min = (y_np == 1)
    if mode == "none":
        return base
    if mode == "exposure":
        mult = np.ones(y_np.shape[0], dtype=float)
        mult[is_min] = 1.0 + lambda_ * np.asarray(exposure_norm)[is_min]
        return base * mult
    if mode == "uniform":
        mult = np.ones(y_np.shape[0], dtype=float)
        mult[is_min] = 1.0 + lambda_
        return base * mult
    raise ValueError(f"Unknown weighting mode: {mode}")


def audit_exposure(
    exposure: np.ndarray,
    y: pd.Series,
    feature_magnitudes: np.ndarray,
) -> dict[str, float]:
    """Produce the §27 exposure sanity audit."""
    e = np.asarray(exposure, dtype=float)
    is_min = (np.asarray(y) == 1)
    minority = e[is_min]
    majority = e[~is_min]
    top10_min = minority[np.argsort(-minority)[: max(1, int(0.10 * minority.size))]] if minority.size else np.array([])
    # Correlation of exposure with a corruption-sensitivity proxy (the mean
    # standardised magnitude over the affected features).
    mag = np.asarray(feature_magnitudes, dtype=float)
    proxy = mag.mean(axis=1) if mag.ndim == 2 and mag.shape[1] else np.zeros_like(e)
    corr = np.corrcoef(e, proxy)[0, 1] if np.isfinite(proxy).all() and np.std(proxy) > 0 else np.nan
    return {
        "exposure_mean": float(np.nanmean(e)),
        "exposure_std": float(np.nanstd(e)),
        "exposure_min": float(np.nanmin(e)),
        "exposure_max": float(np.nanmax(e)),
        "minority_exposure_mean": float(np.nanmean(minority)) if minority.size else np.nan,
        "majority_exposure_mean": float(np.nanmean(majority)) if majority.size else np.nan,
        "top10_pct_minority_magnitude_mean": float(np.nanmean(top10_min)) if top10_min.size else np.nan,
        "exposure_corr_rho": corr,
        "n_nan": int(np.isnan(e).sum()),
        "n_inf": int(np.isinf(e).sum()),
    }
