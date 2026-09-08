"""Minority-focused mild missingness augmentation (MAMR component B).

The augmentation duplicates a corrupted copy of the source train and
concatenates it. The original source train is always 100% retained. Corruption
only touches the top predictive feature set and is deliberately milder than the
severe test shifts (§6).

Mechanism mixture (per duplicated row):
- ``mcar``           : independent Bernoulli at corruption_rate.
- ``feature_specific``: per-feature rates derived from corruption_rate.
- ``block``          : a block of affected features is jointly missing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _mechanism_for(row: int, mix: dict[str, float], rng: np.random.Generator) -> str:
    keys = ["mcar", "feature_specific", "block"]
    probs = [float(mix.get(k, 0.0)) for k in keys]
    total = sum(probs)
    if total <= 0:
        return "mcar"
    chosen = rng.choice(keys, p=np.asarray(probs) / total)
    return str(chosen)


def generate_row_mask(
    n_features: int,
    mechanism: str,
    corruption_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a single-row boolean missingness mask over ``n_features``."""
    if n_features == 0:
        return np.zeros(0, dtype=bool)
    if mechanism == "mcar":
        return rng.random(n_features) < corruption_rate
    if mechanism == "feature_specific":
        # Mild per-feature variation around the corruption rate.
        factors = np.linspace(0.7, 1.3, n_features)
        return rng.random(n_features) < (corruption_rate * factors)
    if mechanism == "block":
        block_size = int(max(1, np.clip(n_features // 2, 1, n_features)))
        start = int(rng.integers(0, max(1, n_features - block_size + 1)))
        mask = np.zeros(n_features, dtype=bool)
        mask[start : start + block_size] = True
        return mask
    raise ValueError(f"Unknown mechanism: {mechanism}")


def generate_augmentation_masks(
    n_rows: int,
    n_features: int,
    mechanism_mix: dict[str, float],
    corruption_rate: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return an (n_rows, n_features) boolean mask for the duplicated rows."""
    if n_rows == 0 or n_features == 0:
        return np.zeros((n_rows, n_features), dtype=bool)
    rows = []
    for i in range(n_rows):
        mech = _mechanism_for(i, mechanism_mix, rng)
        rows.append(generate_row_mask(n_features, mech, corruption_rate, rng))
    return np.vstack(rows)


def augment_minority(
    X: pd.DataFrame,
    y: pd.Series,
    affected_cols: list[str],
    copy_fraction: float,
    corruption_rate: float,
    rng: np.random.Generator,
    mechanism_mix: dict[str, float] | None = None,
    majority_corruption_rate: float = 0.05,
    minority_only: bool = True,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Return duplicated (corrupted) rows and their labels.

    ``minority_only=True`` duplicates only the minority (positive) class. When
    ``False`` a small majority subset is also corrupted, for mechanism control.
    The third return value is the integer position (into the original ``X``
    frame) of each duplicated row, used to inherit sample weights.
    """
    mix = mechanism_mix or {"mcar": 0.4, "feature_specific": 0.3, "block": 0.3}
    y_np = np.asarray(y)
    is_min = (y_np == 1)
    minority_idx = np.where(is_min)[0]
    n_min = int(minority_idx.size)
    pick: list[int] = []
    labels: list[int] = []

    if n_min > 0:
        n_copy = int(round(n_min * float(copy_fraction)))
        if n_copy > 0:
            chosen = rng.choice(minority_idx, size=min(n_copy, n_min), replace=False)
            pick.extend(chosen.tolist())
            labels.extend([1] * chosen.size)

    if not minority_only:
        majority_idx = np.where(~is_min)[0]
        n_maj = int(majority_idx.size)
        n_copy_maj = int(round(n_maj * float(copy_fraction) * 0.2))
        if n_maj > 0 and n_copy_maj > 0:
            chosen_maj = rng.choice(majority_idx, size=min(n_copy_maj, n_maj), replace=False)
            pick.extend(chosen_maj.tolist())
            labels.extend([0] * chosen_maj.size)

    if not pick:
        return (
            X.iloc[0:0].copy(),
            y.iloc[0:0].copy(),
            np.asarray(pick, dtype=int),
        )

    aug = X.iloc[pick].copy()
    aug_y = pd.Series(labels, index=aug.index)
    masks = generate_augmentation_masks(
        len(pick), len(affected_cols), mix, corruption_rate, rng
    )
    for j, col in enumerate(affected_cols):
        if col not in aug.columns:
            continue
        vals = aug[col].astype(float).to_numpy().copy()
        vals[masks[:, j]] = np.nan
        aug[col] = vals

    # If a majority subset was included, its corruption rate should be milder.
    if not minority_only and majority_corruption_rate != corruption_rate:
        maj_mask = (aug_y.to_numpy() == 0)
        if maj_mask.any():
            m = generate_augmentation_masks(
                int(maj_mask.sum()), len(affected_cols), mix,
                majority_corruption_rate, rng,
            )
            pos = np.where(maj_mask)[0]
            for j, col in enumerate(affected_cols):
                if col not in aug.columns:
                    continue
                vals = aug[col].astype(float).to_numpy().copy()
                vals[pos[m[:, j]]] = np.nan
                aug[col] = vals

    return (
        aug.reset_index(drop=True),
        aug_y.reset_index(drop=True),
        np.asarray(pick, dtype=int),
    )
