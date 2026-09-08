"""Phase E3R: MAMR conservative safety configuration support.

This module adds the pre-registered validation stress suite, the safety utility
(``U_safe``), the global conservative parameter selection and the E3R metrics
``MGR`` / ``MHR`` / Pareto non-dominance. It never selects parameters using test
labels: every selection decision is made on the source-domain validation split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mamr import (
    _env_rng,
    _prepare_train,
    _transform,
    stratified_split,
)
from .metrics import compute_metrics, threshold_05
from .missingness_generator import (
    apply_mask,
    generate_block,
    generate_class_conditional,
    generate_feature_specific,
    generate_mcar,
)

EPS = 1e-12


def build_validation_stress_envs(
    X_val: pd.DataFrame,
    feat_cols: list[str],
    y_val: pd.Series,
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Frozen source-validation stress suite used only for parameter search."""
    envs: dict[str, pd.DataFrame] = {}
    envs["vmcar_20"] = generate_mcar(X_val, feat_cols, 0.20, _env_rng("vmcar_20", seed))
    envs["vmcar_30"] = generate_mcar(X_val, feat_cols, 0.30, _env_rng("vmcar_30", seed))
    envs["vfeature_mild"] = generate_feature_specific(
        X_val, feat_cols, [0.25, 0.20, 0.15, 0.10], _env_rng("vfeature_mild", seed)
    )
    envs["vblock_mild"] = generate_block(X_val, feat_cols, 0.15, _env_rng("vblock_mild", seed))
    envs["vcc_mild"] = generate_class_conditional(
        X_val, feat_cols, y_val, 0.30, 0.10, _env_rng("vcc_mild", seed)
    )
    return envs


def run_stress_cell(
    ds: dict[str, Any],
    seed: int,
    model: str,
    method: str,
    cfg: Any,
    params: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Train ``method`` on the source train and evaluate the validation stress suite."""
    X, y = ds["X"], ds["y"]
    train_idx, val_idx, _ = stratified_split(X, y, cfg.split, seed)
    X_val_raw = X.iloc[val_idx]
    y_val = y.iloc[val_idx]
    prep = _prepare_train(ds, seed, model, method, cfg, params)
    feat_cols = prep["feat_cols"]
    masks = build_validation_stress_envs(X_val_raw, feat_cols, y_val, seed)
    rows: list[dict[str, Any]] = []
    for env_name, mask in masks.items():
        X_val_masked = apply_mask(X_val_raw, mask)
        X_te = _transform(X_val_masked, prep)
        proba = prep["estimator"].predict_proba(X_te.to_numpy())[:, 1]
        pred = threshold_05(proba)
        metrics = compute_metrics(y_val, proba, pred)
        rows.append({"environment": env_name, **metrics})
    return rows, prep["audit"]


def safe_utility(
    minority: float,
    majority: float,
    auprc: float,
    ref_minority: float,
    ref_majority: float,
    ref_auprc: float,
    alpha: float,
    beta: float,
) -> float:
    """Pre-registered safety utility ``U_safe`` (§6)."""
    return (
        minority
        - alpha * max(0.0, ref_majority - majority)
        - beta * max(0.0, ref_auprc - auprc)
    )


def safety_feasible(
    majority_delta: float,
    auprc_delta: float,
    hard_majority_delta: float,
    hard_auprc_delta: float,
) -> bool:
    """Validation hard constraints (§6): both deltas must be above the floor."""
    return majority_delta >= hard_majority_delta and auprc_delta >= hard_auprc_delta


def build_candidate_grid_step_a(cfg: Any) -> list[dict[str, float]]:
    ss = cfg.raw["safety_search"]
    step_a = ss["step_a"]
    return [
        {
            "step": "A",
            "lambda_": float(lam),
            "copy_fraction": float(step_a["copy_fraction"]),
            "corruption_rate": float(step_a["corruption_rate"]),
        }
        for lam in step_a["lambda_candidates"]
    ]


def build_candidate_grid_step_b(cfg: Any, lambda_: float) -> list[dict[str, float]]:
    ss = cfg.raw["safety_search"]
    step_b = ss["step_b"]
    return [
        {
            "step": "B",
            "lambda_": float(lambda_),
            "copy_fraction": float(cf),
            "corruption_rate": float(cr),
        }
        for cf in step_b["copy_fraction_candidates"]
        for cr in step_b["corruption_rate_candidates"]
    ]


def compute_mgr(
    gain_safe: float,
    gain_original: float,
    eps: float = EPS,
) -> float:
    """Minority Gain Retention (relative to best_standard)."""
    return gain_safe / (gain_original + eps)


def compute_mhr(
    majority_delta_safe: float,
    majority_delta_original: float,
    eps: float = EPS,
) -> float:
    """Majority Harm Reduction: 1 - H_safe/(H_original + eps)."""
    h_safe = max(0.0, -majority_delta_safe)
    h_orig = max(0.0, -majority_delta_original)
    return 1.0 - h_safe / (h_orig + eps)


def pareto_non_dominated(
    frame: pd.DataFrame,
    target: str = "safe_mamr",
) -> bool:
    """Whether ``target`` is Pareto non-dominated among the candidates in ``frame``.

    ``frame`` must have columns ``method``, ``minority_recall``, ``majority_recall``,
    ``AUPRC``. A candidate is dominated if some other candidate is no worse on all
    three and strictly better on at least one.
    """
    if frame.empty or target not in set(frame["method"]):
        return False
    cand = frame[frame["method"] == target]
    cand_r = float(cand["minority_recall"].mean())
    cand_a = float(cand["majority_recall"].mean())
    cand_p = float(cand["AUPRC"].mean())
    for _, other in frame.iterrows():
        if other["method"] == target:
            continue
        o_r = float(other["minority_recall"])
        o_a = float(other["majority_recall"])
        o_p = float(other["AUPRC"])
        # Is ``other`` dominating target?
        ge_all = (o_r >= cand_r - EPS) and (o_a >= cand_a - EPS) and (o_p >= cand_p - EPS)
        strictly_better = (o_r > cand_r + EPS) or (o_a > cand_a + EPS) or (o_p > cand_p + EPS)
        if ge_all and strictly_better:
            return False
    return True


def select_global_safe_config(
    frame: pd.DataFrame,
    *,
    feasible_fraction_min: float = 0.80,
    eps: float = EPS,
) -> dict[str, Any]:
    """Choose the global conservative config (§7).

    Prefers the candidate with the highest median ``U_safe`` among those whose
    validation feasibility fraction is >= ``feasible_fraction_min``. If none meet
    the fraction, picks the candidate with the smallest majority sacrifice and
    marks ``validation_safety_feasible=False``.
    """
    if frame.empty:
        raise ValueError("Cannot select a safe config from an empty search frame.")
    cols = ["lambda_", "copy_fraction", "corruption_rate"]
    keys = ["step", "lambda_", "copy_fraction", "corruption_rate"]
    grouped = frame.groupby(keys)[["u_safe", "majority_delta", "feasible"]].agg(
        {
            "u_safe": ["median", "mean"],
            "majority_delta": "mean",
            "feasible": "mean",
        }
    )
    grouped.columns = ["u_safe_median", "u_safe_mean", "majority_delta_mean", "feasible_fraction"]
    grouped = grouped.reset_index()
    feasible = grouped[grouped["feasible_fraction"] >= feasible_fraction_min - eps]
    if not feasible.empty:
        chosen = feasible.sort_values(
            ["u_safe_median", "u_safe_mean"], ascending=False
        ).iloc[0]
        ok = True
    else:
        # Fallback: least majority sacrifice (largest / least-negative delta).
        chosen = grouped.sort_values("majority_delta_mean", ascending=False).iloc[0]
        ok = False
    return {
        "lambda_": float(chosen["lambda_"]),
        "copy_fraction": float(chosen["copy_fraction"]),
        "corruption_rate": float(chosen["corruption_rate"]),
        "step": str(chosen["step"]),
        "u_safe_median": float(chosen["u_safe_median"]),
        "u_safe_mean": float(chosen["u_safe_mean"]),
        "feasible_fraction": float(chosen["feasible_fraction"]),
        "majority_delta_mean": float(chosen["majority_delta_mean"]),
        "validation_safety_feasible": bool(ok),
    }
