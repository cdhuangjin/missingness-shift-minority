"""Mechanism-level, consistency and interaction analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def mechanism_summary(vuln: pd.DataFrame, cols=("MVG_recall",)):
    rows = []
    for env, grp in vuln.groupby("environment"):
        for col in cols:
            vals = grp[col].dropna()
            rows.append(
                {
                    "environment": env,
                    "metric": col,
                    "mean": vals.mean(),
                    "median": vals.median(),
                    "positive_fraction": float((vals > 0).mean()),
                    "n": int(vals.shape[0]),
                }
            )
    return pd.DataFrame(rows)


def dataset_consistency(vuln: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ds, grp in vuln.groupby("dataset"):
        vals = grp["MVG_recall"].dropna()
        rows.append(
            {
                "dataset": ds,
                "mean_MVG": vals.mean(),
                "median_MVG": vals.median(),
                "positive_fraction": float((vals > 0).mean()),
                "n": int(vals.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def model_consistency(vuln: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, grp in vuln.groupby("model"):
        vals = grp["MVG_recall"].dropna()
        rows.append(
            {
                "model": model,
                "mean_MVG": vals.mean(),
                "median_MVG": vals.median(),
                "positive_fraction": float((vals > 0).mean()),
                "n": int(vals.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def imbalance_correlation(
    profiles: pd.DataFrame,
    consistency: pd.DataFrame,
    slopes: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Spearman rank correlation between log imbalance and vulnerability."""
    merged = consistency.merge(profiles[["dataset", "imbalance_ratio"]], on="dataset")
    merged["log_imbalance"] = np.log10(merged["imbalance_ratio"])
    if slopes is not None and not slopes.empty and "dataset" in slopes.columns:
        slope_mean = slopes.groupby("dataset")["minority_slope"].mean().reset_index()
        merged = merged.merge(slope_mean, on="dataset", how="left")

    out = []
    for col in ("mean_MVG", "minority_slope"):
        if col not in merged.columns:
            continue
        valid = merged[["log_imbalance", col]].dropna()
        if valid.shape[0] >= 3:
            rho, p = stats.spearmanr(valid["log_imbalance"], valid[col])
            out.append({"variable": col, "spearman_rho": float(rho), "p": float(p),
                        "n": int(valid.shape[0])})
        else:
            out.append({"variable": col, "spearman_rho": np.nan, "p": np.nan,
                        "n": int(valid.shape[0])})
    return pd.DataFrame(out)


def interaction_ols(df: pd.DataFrame) -> dict:
    """MVG ~ missing_rate + log_imbalance + missing_rate*log_imbalance."""
    import statsmodels.formula.api as smf

    sub = df[["MVG_recall", "missing_rate", "log_imbalance"]].dropna()
    if sub.shape[0] < 10:
        return {"n": int(sub.shape[0]), "model": None}
    model = smf.ols(
        "MVG_recall ~ missing_rate + log_imbalance + missing_rate:log_imbalance",
        data=sub,
    ).fit()
    params = {k: float(v) for k, v in model.params.items()}
    pvals = {k: float(v) for k, v in model.pvalues.items()}
    return {
        "n": int(sub.shape[0]),
        "params": params,
        "pvalues": pvals,
        "r_squared": float(model.rsquared),
        "interaction_est": params.get("missing_rate:log_imbalance", np.nan),
        "interaction_p": pvals.get("missing_rate:log_imbalance", np.nan),
    }


def ambg_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Aggregate-Metric Blindness Gap (descriptive)."""
    rows = []
    p0 = raw[raw["pipeline"] == "p0"]
    for (ds, model, seed), grp in p0.groupby(["dataset", "model", "seed"]):
        iid = grp[grp["environment"] == "mcar_05"]
        if iid.empty:
            continue
        a0 = float(iid["AUROC"].iloc[0])
        ap0 = float(iid["AUPRC"].iloc[0])
        r0 = float(iid["minority_recall"].iloc[0])
        shifted = grp[grp["environment"] != "mcar_05"]
        for _, row in shifted.iterrows():
            recall_drop = r0 - float(row["minority_recall"])
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "seed": seed,
                    "environment": row["environment"],
                    "recall_drop": recall_drop,
                    "auroc_drop": a0 - float(row["AUROC"]),
                    "auprc_drop": ap0 - float(row["AUPRC"]),
                    "AMBG": recall_drop - (a0 - float(row["AUROC"])),
                    "AMBG_auprc": recall_drop - (ap0 - float(row["AUPRC"])),
                }
            )
    return pd.DataFrame(rows)
