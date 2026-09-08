"""Phase E3R Safety Gate: pre-registered R1-R10 conditions and terminal verdict.

The Safety Gate is the last repair gate for MAMR. It only ever returns
``STRONG-GO`` / ``GO`` / ``HOLD-STOP`` / ``STOP``. ``R3`` (majority safety) is
mandatory: a GO is impossible while ``R3`` is False. The gate never broadens the
original E3 GO-C floor.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .mamr_safety import compute_mgr, compute_mhr, pareto_non_dominated
from .method_gate import (
    compute_best_standard,
    compute_big,
    compute_iid_cost,
    compute_lrr,
    compute_mas,
)
from .statistical_analysis import (
    bootstrap_ci,
    cohens_dz,
    paired_ttest,
    rank_biserial,
    wilcoxon_paired,
)

EPS = 1e-12


def _ok(raw: pd.DataFrame) -> pd.DataFrame:
    if "status" in raw.columns:
        return raw[raw["status"].fillna("ok") == "ok"].copy()
    return raw.copy()


def _merge_target(raw: pd.DataFrame, best_standard: pd.DataFrame, target: str, hr: list[str]) -> pd.DataFrame:
    cells = raw[raw["method"] == target]
    merged = cells.merge(
        best_standard,
        on=["dataset", "model", "seed", "environment"],
        how="left",
    )
    merged = merged[merged["environment"].isin(hr)]
    merged["minority_recall_gain"] = merged["minority_recall"] - merged["best_standard_minority_recall"]
    merged["majority_recall_delta"] = merged["majority_recall"] - merged["best_standard_majority_recall"]
    merged["auprc_delta"] = merged["AUPRC"] - merged["best_standard_auprc"]
    return merged


def _cell_metrics(raw: pd.DataFrame, target: str, hr: list[str]) -> dict[str, float]:
    """Per-cell (dataset x model) high-risk-averaged metrics for one method."""
    sub = _ok(raw)
    sub = sub[(sub["method"] == target) & (sub["environment"].isin(hr))]
    if sub.empty:
        return {}
    return {
        "minority_recall": float(sub.groupby(["dataset", "model"])["minority_recall"].mean().mean()),
        "majority_recall": float(sub.groupby(["dataset", "model"])["majority_recall"].mean().mean()),
        "AUPRC": float(sub.groupby(["dataset", "model"])["AUPRC"].mean().mean()),
    }


def _pareto_cells(raw: pd.DataFrame, hr: list[str], candidates: list[str]) -> int:
    """Number of dataset x model cells where ``safe_mamr`` is Pareto non-dominated."""
    sub = _ok(raw)
    sub = sub[sub["environment"].isin(hr) & sub["method"].isin(candidates + ["safe_mamr"])]
    count = 0
    total = 0
    for (ds, model), grp in sub.groupby(["dataset", "model"]):
        total += 1
        frame = grp.groupby("method")[["minority_recall", "majority_recall", "AUPRC"]].mean().reset_index()
        keep = ["method", "minority_recall", "majority_recall", "AUPRC"]
        if pareto_non_dominated(frame[keep].copy()):
            count += 1
    return count, total


def evaluate_safety_gate(raw: pd.DataFrame, cfg: Any) -> dict[str, Any]:
    """Evaluate R1-R10 and return the Safety Gate verdict payload."""
    raw = _ok(raw)
    gate = cfg.raw.get("gate", {})
    hr = list(cfg.high_risk_mechanisms)
    control = list(cfg.control_mechanisms)
    shift_envs = [e for e in cfg.environments if e != "mcar_05"]
    candidates = cfg.raw["safety_search"]["pareto_candidates"]
    orig = cfg.raw["original_mamr"]

    best_standard = compute_best_standard(raw)
    merged = _merge_target(raw, best_standard, "safe_mamr", hr)
    merged_orig = _merge_target(raw, best_standard, "original_mamr", hr)

    # Minority / majority / AUPRC deltas vs best_standard (high-risk).
    mr_safe = float(merged["minority_recall_gain"].mean()) if not merged.empty else np.nan
    mr_orig = float(merged_orig["minority_recall_gain"].mean()) if not merged_orig.empty else np.nan
    maj_safe = float(merged["majority_recall_delta"].mean()) if not merged.empty else np.nan
    maj_worst = float(merged.groupby("dataset")["majority_recall_delta"].mean().min()) if not merged.empty else np.nan
    maj_orig = float(merged_orig["majority_recall_delta"].mean()) if not merged_orig.empty else np.nan
    maj_orig_worst = float(merged_orig.groupby("dataset")["majority_recall_delta"].mean().min()) if not merged_orig.empty else np.nan
    ap_safe = float(merged["auprc_delta"].mean()) if not merged.empty else np.nan
    ap_worst = float(merged.groupby("dataset")["auprc_delta"].mean().min()) if not merged.empty else np.nan

    mgr = compute_mgr(mr_safe, mr_orig)
    mhr = compute_mhr(maj_safe, maj_orig)

    # IID safety vs ERM.
    iid = compute_iid_cost(raw, mamr_method="safe_mamr")
    iid_mr = float(iid["iid_minority_recall_delta"].mean()) if not iid.empty else np.nan
    iid_ap = float(iid["iid_auprc_delta"].mean()) if not iid.empty else np.nan

    # BIG vs best_standard.
    big = compute_big(raw, best_standard, "safe_mamr", shift_envs)
    big_mean = float(big["big"].mean()) if not big.empty else np.nan
    big_pos = float((big["big"] > 0).mean()) if not big.empty else np.nan
    if not big.empty:
        ds_big = big.groupby("dataset")["big"].mean()
        n_ds_big = int((ds_big >= float(gate["r1_min_big_dataset"])).sum())
    else:
        n_ds_big = 0

    # MAS.
    mas = compute_mas(raw, hr, control, mamr_method="safe_mamr")
    mas_val = float(mas["mas"]) if np.isfinite(mas["mas"]) else np.nan
    hr_gain = float(mas["high_risk_gain"]) if np.isfinite(mas["high_risk_gain"]) else np.nan
    ct_gain = float(mas["control_gain"]) if np.isfinite(mas["control_gain"]) else np.nan

    # LRR on collapse cases.
    lrr = compute_lrr(raw, pd.Series(["safe_mamr"]), shift_envs, include_filter=False)
    collapse = lrr[(lrr["method"] == "safe_mamr") & (lrr["erm_recall_drop"] >= float(gate["r6_min_recall_drop"]))]
    lrr_vals = collapse["lrr"].dropna()
    lrr_med = float(lrr_vals.median()) if len(lrr_vals) else np.nan
    lrr_pos = float((lrr_vals > 0).mean()) if len(lrr_vals) else np.nan

    # Cross-model consistency.
    lr_big = float(big[big["model"] == "logistic_regression"]["big"].mean()) if (not big.empty and (big["model"] == "logistic_regression").any()) else np.nan
    xgb_big = float(big[big["model"] == "xgboost"]["big"].mean()) if (not big.empty and (big["model"] == "xgboost").any()) else np.nan

    # Pareto.
    pareto_count, pareto_total = _pareto_cells(raw, hr, candidates)

    # Statistics (paired bootstrap -> CI on high-risk).
    stats = _run_statistics(raw, cfg, best_standard, hr)

    conditions = {
        "R1": (
            np.isfinite(big_mean) and big_mean >= float(gate["r1_min_big_mean"])
            and np.isfinite(big_pos) and big_pos >= float(gate["r1_min_big_frac"])
            and n_ds_big >= int(gate["r1_min_datasets"])
        ),
        "R2": mgr >= float(gate["r2_min_mgr"]),
        "R3": (
            np.isfinite(maj_safe) and maj_safe >= float(gate["r3_min_majority_mean"])
            and np.isfinite(maj_worst) and maj_worst >= float(gate["r3_min_majority_worst"])
        ),
        "R4": (
            np.isfinite(ap_safe) and ap_safe >= float(gate["r4_min_auprc_mean"])
            and np.isfinite(ap_worst) and ap_worst >= float(gate["r4_min_auprc_worst"])
        ),
        "R5": (
            np.isfinite(iid_mr) and iid_mr >= float(gate["r5_min_iid_recall"])
            and np.isfinite(iid_ap) and iid_ap >= float(gate["r5_min_iid_auprc"])
        ),
        "R6": (
            np.isfinite(lrr_med) and lrr_med >= float(gate["r6_min_lrr"])
            and np.isfinite(lrr_pos) and lrr_pos >= float(gate["r6_min_lrr_pos"])
        ),
        "R7": np.isfinite(lr_big) and lr_big > 0 and np.isfinite(xgb_big) and xgb_big > 0,
        "R8": (
            np.isfinite(mas_val) and mas_val >= float(gate["r8_min_mas"])
            and np.isfinite(hr_gain) and np.isfinite(ct_gain) and hr_gain > ct_gain
        ),
        "R9": (
            np.isfinite(stats["minority_boot_low"]) and stats["minority_boot_low"] > 0
            and np.isfinite(maj_safe) and np.isfinite(maj_orig) and maj_safe > maj_orig
        ),
        "R10": pareto_count >= int(gate["r10_min_cells"]),
    }
    n_pass = int(sum(1 for v in conditions.values() if v))

    # Terminal STOP triggers.
    benefit_significant = (np.isfinite(big_mean) and big_mean > 0) and mgr >= float(gate["stop_mgr_min"]) and n_ds_big >= 1
    stop_auprc = (np.isfinite(ap_safe) and ap_safe < float(gate["stop_auprc_mean"])) or (
        np.isfinite(ap_worst) and ap_worst < float(gate["stop_auprc_worst"])
    )
    stop_iid = (np.isfinite(iid_mr) and iid_mr < float(gate["stop_iid_recall"])) or (
        np.isfinite(iid_ap) and iid_ap < float(gate["stop_iid_auprc"])
    )
    stop_benefit = (np.isfinite(big_mean) and big_mean <= 0) or (not np.isfinite(mgr) or mgr < float(gate["stop_mgr_min"])) or n_ds_big < 1

    if stop_benefit or stop_auprc or stop_iid:
        verdict = "STOP"
    elif not conditions["R3"]:
        verdict = "HOLD-STOP" if benefit_significant else "STOP"
    elif all(conditions.values()):
        verdict = "STRONG-GO"
    elif n_pass >= int(gate["go_min_pass"]) and conditions["R1"] and conditions["R2"] and conditions["R4"] and conditions["R6"] and conditions["R7"]:
        verdict = "GO"
    else:
        verdict = "STOP"

    recommended = {
        "STRONG-GO": "Phase E4 full validation planned (do not auto-enter)",
        "GO": "Phase E4 full validation planned (do not auto-enter)",
        "HOLD-STOP": "Terminal: unresolved real Pareto trade-off; no further MAMR repair",
        "STOP": "Terminal: MAMR method claim halted; retain E1/E2 phenomenon conclusions",
    }[verdict]

    payload = {
        "gate": verdict,
        "conditions": {k: bool(v) for k, v in conditions.items()},
        "counts": {
            "n_pass": n_pass,
            "pareto_cells": pareto_count,
            "pareto_total": pareto_total,
            "n_datasets_big_ge_005": n_ds_big,
            "median_lrr": lrr_med,
            "positive_lrr_fraction": lrr_pos,
            "mas": mas_val,
            "high_risk_gain": hr_gain,
            "control_gain": ct_gain,
        },
        "metrics": {
            "minority_gain_safe": mr_safe,
            "minority_gain_original": mr_orig,
            "mgr": mgr,
            "majority_delta_safe": maj_safe,
            "majority_delta_original": maj_orig,
            "majority_harm_reduction": mhr,
            "worst_dataset_majority_delta_safe": maj_worst,
            "worst_dataset_majority_delta_original": maj_orig_worst,
            "auprc_delta_safe": ap_safe,
            "worst_dataset_auprc_delta_safe": ap_worst,
            "iid_minority_recall_delta": iid_mr,
            "iid_auprc_delta": iid_ap,
            "big_mean": big_mean,
            "big_positive_fraction": big_pos,
            "lr_mean_big": lr_big,
            "xgb_mean_big": xgb_big,
        },
        "statistics": stats,
        "recommended_next_phase": recommended,
    }
    return payload


def _run_statistics(raw: pd.DataFrame, cfg: Any, best_standard: pd.DataFrame, hr: list[str]) -> dict[str, Any]:
    """Paired Safe-MAMR vs best_standard statistics on high-risk shifts."""
    sub = raw[(raw["method"] == "safe_mamr") & (raw["environment"].isin(hr))]
    sub = sub.merge(best_standard, on=["dataset", "model", "seed", "environment"], how="left")
    rows: dict[str, Any] = {}
    iters = int(cfg.raw["statistics"]["bootstrap_iterations"])
    for metric, (base_col, safe_col) in {
        "minority_recall": ("best_standard_minority_recall", "minority_recall"),
        "majority_recall": ("best_standard_majority_recall", "majority_recall"),
        "auprc": ("best_standard_auprc", "AUPRC"),
    }.items():
        a = pd.to_numeric(sub[base_col], errors="coerce").to_numpy()
        b = pd.to_numeric(sub[safe_col], errors="coerce").to_numpy()
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        diff = b - a
        lo, hi = bootstrap_ci(diff, level=float(cfg.raw["statistics"]["confidence_level"]), iters=iters)
        stat, p = wilcoxon_paired(a, b)
        t, pt = paired_ttest(a, b)
        rows[metric] = {
            "n": int(len(diff)),
            "mean_delta": float(diff.mean()) if diff.size else np.nan,
            "median_delta": float(np.median(diff)) if diff.size else np.nan,
            "boot_ci_low": float(lo),
            "boot_ci_high": float(hi),
            "wilcoxon_stat": stat,
            "p_wilcoxon": p,
            "t_stat": t,
            "p_paired_t": pt,
            "rank_biserial": rank_biserial(a, b),
            "cohens_dz": cohens_dz(a, b),
        }
    return {
        "minority_boot_low": rows["minority_recall"]["boot_ci_low"],
        "minority_boot_high": rows["minority_recall"]["boot_ci_high"],
        "majority_boot_low": rows["majority_recall"]["boot_ci_low"],
        "majority_boot_high": rows["majority_recall"]["boot_ci_high"],
        "rows": rows,
    }
