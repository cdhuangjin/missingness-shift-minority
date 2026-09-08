"""Paper Gate: STRONG-GO / GO / HOLD / STOP-PIVOT."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _count(df: pd.DataFrame, col: str, threshold: float, op: str = ">") -> int:
    if df.empty or col not in df.columns:
        return 0
    vals = pd.to_numeric(df[col], errors="coerce")
    if op == ">":
        return int((vals > threshold).sum())
    if op == ">=":
        return int((vals >= threshold).sum())
    return int((vals < threshold).sum())


def _abs_count(df: pd.DataFrame, minor: str, major: str) -> int:
    if df.empty or minor not in df.columns or major not in df.columns or "dataset" not in df.columns:
        return 0
    m = pd.to_numeric(df[minor], errors="coerce").abs()
    j = pd.to_numeric(df[major], errors="coerce").abs()
    return int(df.loc[(m > j).to_numpy(), "dataset"].nunique())


def paper_gate(
    dataset_consistency: pd.DataFrame,
    model_consistency: pd.DataFrame,
    slopes: pd.DataFrame,
    cc_summary: pd.DataFrame,
    hidden_summary: pd.DataFrame,
    reverse_summary: pd.DataFrame,
    stats_row: dict,
    n_datasets: int,
    effect_vs_seed_noise_ok: bool,
) -> tuple[str, dict]:
    n = max(n_datasets, 1)
    conditions: dict[str, bool] = {}
    counts: dict[str, int] = {}

    counts["c1_pos_mvg"] = _count(dataset_consistency, "mean_MVG", 0.0, ">")
    conditions["c1_5of8_datasets_mvg_pos"] = counts["c1_pos_mvg"] >= max(1, int(round(0.625 * n)))

    counts["c2_mvg_ge_010"] = _count(dataset_consistency, "mean_MVG", 0.10, ">=")
    conditions["c2_4of8_datasets_mvg_ge_010"] = counts["c2_mvg_ge_010"] >= max(1, int(round(0.5 * n)))

    counts["c3_models_pos"] = _count(model_consistency, "mean_MVG", 0.0, ">")
    conditions["c3_3models_pos"] = counts["c3_models_pos"] >= 3

    fdr_p = float(stats_row.get("fdr_p", stats_row.get("p_fdr", np.nan)))
    conditions["c4_wilcoxon_fdr_sig"] = (not np.isnan(fdr_p)) and fdr_p < 0.05
    counts["c4_fdr_p"] = fdr_p

    es = float(stats_row.get(
        "effect_size",
        stats_row.get("rank_biserial", stats_row.get("cohens_dz", np.nan)),
    ))
    conditions["c5_effect_moderate"] = (not np.isnan(es)) and abs(es) >= 0.3
    counts["c5_effect_size"] = es

    counts["c6_minority_steeper"] = _abs_count(slopes, "minority_slope", "majority_slope")
    conditions["c6_5of8_steeper"] = counts["c6_minority_steeper"] >= max(1, int(round(0.625 * n)))

    cc_pos = cc_summary[cc_summary["CCEP_recall"] >= 0.10] if not cc_summary.empty else cc_summary
    counts["c7_ccep_010"] = int(cc_pos["dataset"].nunique()) if not cc_pos.empty else 0
    conditions["c7_3datasets_ccep"] = counts["c7_ccep_010"] >= 3

    hid_pos = hidden_summary[hidden_summary["n_hidden"] > 0] if not hidden_summary.empty else hidden_summary
    counts["c8_hidden"] = int(hid_pos["dataset"].nunique()) if not hid_pos.empty else 0
    conditions["c8_3datasets_hidden"] = counts["c8_hidden"] >= 3

    counts["c9_reverse"] = _count(reverse_summary, "delta_MVG_reverse", 0.05, ">=")
    conditions["c9_4datasets_reverse"] = counts["c9_reverse"] >= max(1, int(round(0.5 * n)))

    conditions["c10_effect_gt_seed_noise"] = bool(effect_vs_seed_noise_ok)

    hit = sum(1 for v in conditions.values() if v)
    counts["conditions_hit"] = hit
    counts["n_datasets"] = n

    if hit >= 7:
        verdict = "STRONG-GO"
    else:
        stable_mvg = conditions["c1_5of8_datasets_mvg_pos"] and (
            conditions["c6_5of8_steeper"] or conditions["c4_wilcoxon_fdr_sig"]
        )
        partly_cc_hidden = (counts["c7_ccep_010"] < 3) or (counts["c8_hidden"] < 3)
        if stable_mvg and not conditions["c3_3models_pos"]:
            verdict = "HOLD"
        elif stable_mvg:
            verdict = "GO"
        elif counts["c1_pos_mvg"] <= 2 and counts["c2_mvg_ge_010"] <= 1:
            verdict = "STOP-PIVOT"
        else:
            verdict = "HOLD"

    return verdict, {"conditions": conditions, "counts": counts, "n_hit": hit}
