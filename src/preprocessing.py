"""Splits, feature ranking and source-only imputation.

Leakage contract:
- ranking uses only the source train (labels from train only);
- the imputer is fit only on the source train (after source missingness);
- the same fitted imputer is reused for every target environment.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import train_test_split


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    fractions: dict[str, float],
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train/val/test index arrays using a stratified 60/20/20 split."""
    train_frac = fractions["train"]
    val_frac = fractions["validation"]
    test_frac = fractions["test"]
    if abs(train_frac + val_frac + test_frac - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.")

    idx = np.arange(X.shape[0])
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=val_frac + test_frac, random_state=seed, stratify=y
    )
    # Split the held-out part into val / test with the same ratio as val:test.
    test_share = test_frac / (val_frac + test_frac)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx, y_temp, test_size=test_share, random_state=seed, stratify=y_temp
    )
    return train_idx, val_idx, test_idx


def rank_numeric_features(
    X: pd.DataFrame,
    y: pd.Series,
    numeric_cols: list[str],
    max_affected: int,
    method: str = "mutual_information",
) -> list[str]:
    """Rank numeric features and return the top-k missingness feature set.

    k = min(max_affected, max(2, n_numeric // 4)).
    """
    if not numeric_cols:
        return []
    sub = X[numeric_cols].astype(float)
    if sub.isna().any().any():
        # Rank on a complete-case view to avoid mutual_info errors; no test
        # labels are involved here.
        sub = sub.dropna(axis=0)
        y_rank = y.loc[sub.index]
    else:
        y_rank = y
    if sub.shape[0] < 2 or y_rank.nunique() < 2:
        return numeric_cols[: max(2, len(numeric_cols) // 4)]

    if method in ("mutual_information", "mi"):
        scores = mutual_info_classif(sub.values, y_rank.values, random_state=0)
    elif method in ("xgboost_gain", "gain"):
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=60,
            max_depth=4,
            tree_method="hist",
            eval_metric="logloss",
            random_state=0,
        )
        model.fit(sub.values, y_rank.values)
        scores = model.feature_importances_
    else:
        raise ValueError(f"Unknown rank method: {method}")

    order = sub.columns[np.argsort(-scores)].tolist()
    k = min(max_affected, max(2, len(numeric_cols) // 4))
    return order[:k]


class MedianModeImputer:
    """Median for numeric, most-frequent for categorical, fit on train only."""

    def __init__(self) -> None:
        self.values: dict[str, Any] = {}
        self.fitted: bool = False

    def fit(self, X: pd.DataFrame) -> "MedianModeImputer":
        for col in X.columns:
            series = X[col]
            if series.dtype.kind in ("f", "i", "u"):
                valid = pd.to_numeric(series, errors="coerce").dropna()
                self.values[col] = float(valid.median()) if len(valid) else 0.0
            else:
                counts = series.dropna()
                if counts.empty:
                    self.values[col] = "0"
                else:
                    self.values[col] = counts.mode().iloc[0]
        self.fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted:
            raise RuntimeError("Imputer not fitted.")
        out = X.copy()
        missing_cols = [c for c in out.columns if c not in self.values]
        if missing_cols:
            # Never seen during training; fill with a neutral default and warn once
            # at the caller level. Here we use 0 for numeric / "missing" for object.
            for col in missing_cols:
                if out[col].dtype.kind in ("f", "i", "u"):
                    out[col] = out[col].fillna(0.0)
                else:
                    out[col] = out[col].fillna("missing")
        for col, val in self.values.items():
            if col in out.columns:
                out[col] = out[col].fillna(val)
        return out


def fit_categorical_encoders(
    X: pd.DataFrame, categorical_cols: list[str]
) -> dict[str, dict]:
    """Fit ordinal label maps for categorical columns on source train only."""
    encoders: dict[str, dict] = {}
    for col in categorical_cols:
        classes = X[col].dropna().astype(str).unique().tolist()
        encoders[col] = {label: i for i, label in enumerate(sorted(classes))}
    return encoders


def apply_categorical_encoders(
    X: pd.DataFrame, encoders: dict[str, dict]
) -> pd.DataFrame:
    """Map categorical values to integer codes; unseen labels get -1."""
    out = X.copy()
    for col, mapping in encoders.items():
        if col not in out.columns:
            continue
        codes = out[col].astype(str).map(mapping).fillna(-1).astype(int)
        out[col] = codes
    return out
