"""Matched-MCAR comparator tracks the class-conditional overall rate."""

import numpy as np
import pandas as pd

from src.config import Config
from src.missingness_generator import apply_mask, measure_missingness
from src.phase_e12_runner import build_environments


def _cfg():
    cfg = Config()
    cfg.mcar_rates = [0.05]
    cfg.mar = {"medium": 0.20}
    cfg.class_conditional = {"cc30": {"minority": 0.30, "majority": 0.10}}
    cfg.reverse = {"reverse30": {"minority": 0.10, "majority": 0.30}}
    cfg.block = {"severe": 0.30}
    cfg.feature_specific = {"rates": [0.40, 0.30, 0.20, 0.10]}
    return cfg


def test_matched_rate_within_tolerance():
    rng = np.random.default_rng(9)
    X = pd.DataFrame(rng.normal(size=(2000, 8)), columns=[f"f{i}" for i in range(8)])
    y = pd.Series(np.where(np.arange(2000) < 300, 1, 0))
    cols = X.columns[:4].tolist()
    driver = X.columns[-1]
    envs = build_environments(X, y, cols, driver, 42, _cfg())

    # Env names should be canonical even though config keys are raw.
    assert "cc_30" in envs and "matched_mcar_cc30" in envs
    assert "mar_medium" in envs and "reverse_cc_30" in envs
    assert "block_severe" in envs

    cc_mask = envs["cc_30"]
    match_mask = envs["matched_mcar_cc30"]
    cc_rate = measure_missingness(X, apply_mask(X, cc_mask), y, cols=cols)["overall_missing_rate"]
    match_rate = measure_missingness(X, apply_mask(X, match_mask), y, cols=cols)["overall_missing_rate"]
    assert abs(cc_rate - match_rate) <= 0.01
