"""Missingness generators and profilers.

E0/E1/E2  : MCAR at 5%/10%/30%.
E3        : MAR-like feature-dependent missingness (never depends on the label).
E4        : class-conditional missingness stress test (label used only to define
            the stress generator, never for training or hyper-parameter choice).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _rng(seed: int | np.random.Generator) -> np.random.Generator:
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)


def generate_mcar(
    df: pd.DataFrame,
    cols: list[str],
    rate: float,
    rng: int | np.random.Generator,
) -> pd.DataFrame:
    """Independent Bernoulli MCAR mask over the selected columns."""
    gen = _rng(rng)
    mask = gen.random((df.shape[0], len(cols))) < rate
    return pd.DataFrame(mask, index=df.index, columns=cols)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_b(z: np.ndarray, target: float, scale: float) -> float:
    lo, hi = -60.0, 60.0
    f_lo = float(np.mean(_sigmoid(scale * z + lo)) - target)
    f_hi = float(np.mean(_sigmoid(scale * z + hi)) - target)
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        f_mid = float(np.mean(_sigmoid(scale * z + mid)) - target)
        if abs(f_mid) < 1e-9:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def generate_feature_dependent(
    df: pd.DataFrame,
    cols: list[str],
    driver_col: str,
    target_rate: float,
    scale: float,
    rng: int | np.random.Generator,
) -> pd.DataFrame:
    """MAR-like missingness P(M=1 | X_driver) with a calibrated overall rate."""
    gen = _rng(rng)
    driver = pd.to_numeric(df[driver_col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(driver).any():
        driver = np.nan_to_num(driver, nan=float(np.nanmean(driver)))
    z = (driver - driver.mean()) / (driver.std() + 1e-12)
    b = _calibrate_b(z, target_rate, scale)
    prob = _sigmoid(scale * z + b)
    uniform = gen.random((df.shape[0], len(cols)))
    mask = uniform < prob[:, None]
    return pd.DataFrame(mask, index=df.index, columns=cols)


def generate_class_conditional(
    df: pd.DataFrame,
    cols: list[str],
    y: pd.Series,
    minority_rate: float,
    majority_rate: float,
    rng: int | np.random.Generator,
) -> pd.DataFrame:
    """Class-conditional missingness stress test (minority exposed more)."""
    gen = _rng(rng)
    is_minority = (y == 1).to_numpy()
    row_prob = np.where(is_minority, minority_rate, majority_rate)
    mask = np.zeros((df.shape[0], len(cols)), dtype=bool)
    for j in range(len(cols)):
        mask[:, j] = gen.random(df.shape[0]) < row_prob
    return pd.DataFrame(mask, index=df.index, columns=cols)


def generate_block(
    df: pd.DataFrame,
    cols: list[str],
    target_rate: float,
    rng: int | np.random.Generator,
    block_size: int | None = None,
) -> pd.DataFrame:
    """Block missingness: a subset of features is jointly missing on a subset
    of rows. overall_missing = p * block_size / n_cols, calibrated to target."""
    gen = _rng(rng)
    n_cols = len(cols)
    if block_size is None:
        block_size = max(2, n_cols // 2)
    block_size = min(block_size, n_cols)
    p = min(1.0, max(0.0, target_rate * n_cols / block_size))
    selected = gen.random(df.shape[0]) < p
    mask = np.zeros((df.shape[0], n_cols), dtype=bool)
    mask[:, :block_size] = selected[:, None]
    return pd.DataFrame(mask, index=df.index, columns=cols)


def generate_feature_specific(
    df: pd.DataFrame,
    cols: list[str],
    rates: list[float],
    rng: int | np.random.Generator,
) -> pd.DataFrame:
    """Feature-specific missingness: each feature has its own missing rate."""
    gen = _rng(rng)
    mask = np.zeros((df.shape[0], len(cols)), dtype=bool)
    for j in range(len(cols)):
        rate = rates[j % len(rates)]
        mask[:, j] = gen.random(df.shape[0]) < rate
    return pd.DataFrame(mask, index=df.index, columns=cols)


def apply_mask(df: pd.DataFrame, mask: pd.DataFrame) -> pd.DataFrame:
    """Return a copy where masked cells become NaN."""
    out = df.copy()
    for col in mask.columns:
        out.loc[mask[col].to_numpy(), col] = np.nan
    return out


def measure_missingness(
    df_original: pd.DataFrame,
    df_masked: pd.DataFrame,
    y: pd.Series,
    cols: list[str] | None = None,
) -> dict[str, Any]:
    """Report actual overall / minority / majority / per-feature rates."""
    if cols is None:
        cols = list(df_original.columns)
    total_cells = df_original.shape[0] * len(cols)
    overall = float(df_masked[cols].isna().sum().sum() / max(total_cells, 1))

    is_minority = (y == 1).to_numpy()
    n_min = max(int(is_minority.sum()), 1)
    n_maj = max(int((~is_minority).sum()), 1)
    n_cells_min = max(n_min * len(cols), 1)
    n_cells_maj = max(n_maj * len(cols), 1)

    minority_rate = float(
        df_masked.loc[is_minority, cols].isna().sum().sum() / n_cells_min
    )
    majority_rate = float(
        df_masked.loc[~is_minority, cols].isna().sum().sum() / n_cells_maj
    )
    per_feature = {
        col: float(df_masked[col].isna().mean()) for col in cols
    }
    return {
        "overall_missing_rate": overall,
        "minority_missing_rate": minority_rate,
        "majority_missing_rate": majority_rate,
        "per_feature_missing_rate": per_feature,
    }


def pick_driver_col(df: pd.DataFrame, topk: list[str], all_numeric: list[str]) -> str:
    """Choose an observed numeric driver not in the top-k set, if possible."""
    candidates = [c for c in all_numeric if c not in set(topk)]
    if candidates:
        return candidates[0]
    return topk[0]
