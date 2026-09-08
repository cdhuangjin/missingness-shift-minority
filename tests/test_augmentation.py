"""Augmentation mechanics tests."""

import numpy as np
import pandas as pd

from src.augmentation import (
    augment_minority,
    generate_augmentation_masks,
    generate_row_mask,
)


def _xy(n=300, seed=3):
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 6)), columns=[f"f{i}" for i in range(6)])
    y = pd.Series(np.where(np.arange(n) % 5 == 0, 1, 0))
    return X, y


def test_augmentation_preserves_labels():
    X, y = _xy()
    aug, aug_y, pos = augment_minority(X, y, ["f0", "f1", "f2"], 0.5, 0.15,
                                       np.random.default_rng(0))
    assert (aug_y.to_numpy() == 1).all()
    assert set(pos) <= set(np.where((y.to_numpy() == 1))[0])


def test_augmentation_only_modifies_allowed_features():
    X, y = _xy()
    affected = ["f0", "f1", "f2", "f3"]
    aug, _, _ = augment_minority(X, y, affected, 0.5, 0.3,
                                 np.random.default_rng(4))
    allowed = set(affected)
    for col in X.columns:
        if col in allowed:
            assert aug[col].isna().any(), f"expected corruption in {col}"
        else:
            assert not aug[col].isna().any(), f"unexpected corruption in {col}"


def test_augmentation_reproducible_by_seed():
    X, y = _xy()
    a1, _, _ = augment_minority(X, y, ["f0", "f1", "f2"], 0.5, 0.15,
                                np.random.default_rng(99))
    a2, _, _ = augment_minority(X, y, ["f0", "f1", "f2"], 0.5, 0.15,
                                np.random.default_rng(99))
    pd.testing.assert_frame_equal(a1, a2)


def test_generate_row_mask_shapes():
    mask = generate_row_mask(5, "mcar", 0.2, np.random.default_rng(0))
    assert mask.shape == (5,)
    assert mask.dtype == bool


def test_generate_augmentation_masks_shape_and_mechanisms():
    mask = generate_augmentation_masks(20, 6, {"mcar": 0.4, "feature_specific": 0.3, "block": 0.3},
                                       0.15, np.random.default_rng(0))
    assert mask.shape == (20, 6)
    assert mask.dtype == bool


def test_block_mask_is_contiguous():
    mask = generate_row_mask(8, "block", 0.2, np.random.default_rng(5))
    if mask.any():
        idx = np.where(mask)[0]
        assert np.all(np.diff(idx) == 1), "block mask must be contiguous"
