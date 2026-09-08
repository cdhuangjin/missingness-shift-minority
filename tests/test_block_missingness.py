"""Correlated/block missingness: a feature block disappears jointly."""

import numpy as np
import pandas as pd

from src.missingness_generator import apply_mask, generate_block, measure_missingness


def test_block_structure_and_rate():
    rng = np.random.default_rng(2)
    X = pd.DataFrame(rng.normal(size=(1000, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.where(np.arange(1000) < 150, 1, 0))
    cols = X.columns[:6].tolist()
    block_size = 3
    mask = generate_block(X, cols, 0.30, rng, block_size=block_size)
    masked = apply_mask(X, mask)
    rate = measure_missingness(X, masked, y, cols=cols)["overall_missing_rate"]
    assert 0.24 <= rate <= 0.36  # overall ~= p * block_size / n_cols

    # Rows selected for the block must have all block features missing jointly.
    selected = masked[cols[:block_size]].isna().all(axis=1)
    n_selected = int(selected.sum())
    assert n_selected > 0
    # For those selected rows, the features outside the block must be observed.
    outside = masked.loc[selected, cols[block_size:]].isna()
    assert not outside.any().any()
