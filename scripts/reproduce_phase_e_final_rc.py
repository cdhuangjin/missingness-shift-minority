"""One-shot Reviewer-Proof supplementary validation (Phase E-Final-RC).

Purely re-aggregates existing Phase E1/E2 / E3 / E3R artifacts. No model is
trained; the frozen E1/E2 definitions of MiRD / MaRD / MVG / MRR / CCEP /
reverse-exposure effect / hidden failure / slope are used unchanged.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.reverse_control import reverse_exposure_effect  # noqa: E402
from src.slope_analysis import fit_slope  # noqa: E402
from src.statistical_analysis import (  # noqa: E402
    bootstrap_ci,
    cohens_dz,
    rank_biserial,
    wilcoxon_paired,
)
from src.vulnerability import (  # noqa: E402
    mvg,
    relative_degradation,
)

EPS = 1e-12
MCAR_LABEL = {0.05: "mcar_05", 0.10: "mcar_10", 0.20: "mcar_20", 0.30: "mcar_30", 0.40: "mcar_40"}
AGGREGATE = ["AUPRC", "AUROC", "balanced_accuracy", "MCC", "Gmean"]
NATURAL_MISSING = {"bankruptcy", "ozone"}


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _raw() -> pd.DataFrame:
    return pd.read_csv(PROJECT_ROOT / "results" / "phase_e12" / "raw_results.csv")


def _primary(raw: pd.DataFrame) -> pd.DataFrame:
    return raw[
        (raw["pipeline"] == "p0")
        & (raw["imputer"] == "median")
        & (raw["feature_set"] == "top")
    ].copy()


def compute_metric_sensitivity(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per shifted cell: class-specific MiRD/MaRD/MVG + aggregate drops."""
    p0 = _primary(raw)
    rows = []
    for (ds, seed, model), grp in p0.groupby(["dataset", "seed", "model"]):
        iid = grp[grp["environment"] == "mcar_05"]
        if iid.empty:
            continue
        b = iid.iloc[0]
        for _, r in grp[grp["environment"] != "mcar_05"].iterrows():
            miRD_r = relative_degradation(float(b["minority_recall"]), float(r["minority_recall"]), EPS)
            maRD_r = relative_degradation(float(b["majority_recall"]), float(r["majority_recall"]), EPS)
            miRD_f = relative_degradation(float(b["minority_f1"]), float(r["minority_f1"]), EPS)
            maRD_f = relative_degradation(float(b["majority_f1"]), float(r["majority_f1"]), EPS)
            row = {
                "dataset": ds,
                "seed": seed,
                "model": model,
                "environment": r["environment"],
                "missing_rate": r["missing_rate"],
                "MiRD_recall": miRD_r,
                "MaRD_recall": maRD_r,
                "MVG_recall": mvg(miRD_r, maRD_r),
                "MiRD_f1": miRD_f,
                "MaRD_f1": maRD_f,
                "MVG_f1": mvg(miRD_f, maRD_f),
            }
            for m in AGGREGATE:
                base = float(b[m])
                shift = float(r[m])
                row[f"{m}_drop"] = base - shift
                row[f"{m}_rel_drop"] = relative_degradation(base, shift, EPS)
                row[f"{m}_iid"] = base
            rows.append(row)
    detail = pd.DataFrame(rows)
    if detail.empty:
        return detail, pd.DataFrame()
    sig = (np.sign(detail["MiRD_recall"]) == np.sign(detail["MiRD_f1"])).mean()
    corr_rf = float(detail["MiRD_recall"].corr(detail["MiRD_f1"]))
    rel_recall = detail["MiRD_recall"].mean()
    rel_drops = {
        k: float(detail[f"{k}_rel_drop"].mean())
        for k in ["AUROC", "AUPRC", "balanced_accuracy", "MCC", "Gmean"]
    }
    summary = pd.DataFrame(
        [
            {"question": "Q1_minority_recall_f1_same_direction", "value": float(sig), "raw": corr_rf},
            {"question": "Q1_minority_recall_mean", "value": rel_recall, "raw": float(detail.shape[0])},
            {"question": "Q1_minority_f1_mean", "value": detail["MiRD_f1"].mean(), "raw": float(detail.shape[0])},
            {"question": "Q2_balanced_accuracy_rel_drop", "value": rel_drops["balanced_accuracy"], "raw": float(detail.shape[0])},
            {"question": "Q2_gmean_rel_drop", "value": rel_drops["Gmean"], "raw": float(detail.shape[0])},
            {"question": "Q3_auroc_rel_drop", "value": rel_drops["AUROC"], "raw": float(detail.shape[0])},
            {"question": "Q3_minority_recall_rel_drop", "value": rel_recall, "raw": float(detail.shape[0])},
            {"question": "Q4_auprc_rel_drop", "value": rel_drops["AUPRC"], "raw": float(detail.shape[0])},
            {"question": "Q4_auroc_rel_drop", "value": rel_drops["AUROC"], "raw": float(detail.shape[0])},
        ]
    )
    return detail, summary


def _vuln_pool(raw: pd.DataFrame) -> pd.DataFrame:
    p0 = _primary(raw)
    rows = []
    for (ds, seed, model), grp in p0.groupby(["dataset", "seed", "model"]):
        iid = grp[grp["environment"] == "mcar_05"]
        if iid.empty:
            continue
        b = iid.iloc[0]
        for _, r in grp[grp["environment"] != "mcar_05"].iterrows():
            miR = relative_degradation(float(b["minority_recall"]), float(r["minority_recall"]), EPS)
            maR = relative_degradation(float(b["majority_recall"]), float(r["majority_recall"]), EPS)
            miF = relative_degradation(float(b["minority_f1"]), float(r["minority_f1"]), EPS)
            maF = relative_degradation(float(b["majority_f1"]), float(r["majority_f1"]), EPS)
            rows.append(
                {
                    "dataset": ds, "seed": seed, "model": model,
                    "environment": r["environment"],
                    "MVG_recall": mvg(miR, maR), "MVG_f1": mvg(miF, maF),
                    "MiRD_recall": miR, "MaRD_recall": maR,
                }
            )
    return pd.DataFrame(rows)


def _slopes(raw: pd.DataFrame, rates: list[float] | None = None) -> pd.DataFrame:
    p0 = _primary(raw)
    rates = rates or sorted(MCAR_LABEL.keys())
    envs = [MCAR_LABEL[r] for r in rates]
    mcar = p0[p0["environment"].isin(envs)]
    rows = []
    for (ds, model), grp in mcar.groupby(["dataset", "model"]):
        xs, yr, ym = [], [], []
        for rate in rates:
            env = MCAR_LABEL[rate]
            sub = grp[grp["environment"] == env]
            if sub.empty:
                continue
            xs.append(rate)
            yr.append(float(sub["minority_recall"].mean()))
            ym.append(float(sub["majority_recall"].mean()))
        if len(xs) < 2:
            continue
        ms, _ = fit_slope(xs, yr)
        js, _ = fit_slope(xs, ym)
        rows.append(
            {
                "dataset": ds, "model": model,
                "minority_slope": ms, "majority_slope": js,
                "slope_gap": abs(ms) - abs(js),
            }
        )
    return pd.DataFrame(rows)


def _ccep_cc30(raw: pd.DataFrame) -> pd.DataFrame:
    p0 = _primary(raw)
    rows = []
    for (ds, seed, model), grp in p0.groupby(["dataset", "seed", "model"]):
        c = grp[grp["environment"] == "cc_30"]
        m = grp[grp["environment"] == "matched_mcar_cc30"]
        if c.empty or m.empty:
            continue
        rows.append(
            {
                "dataset": ds, "seed": seed, "model": model, "cc_level": "cc_30",
                "matched_mcar_recall": float(m["minority_recall"].iloc[0]),
                "cc_recall": float(c["minority_recall"].iloc[0]),
                "CCEP_recall": float(m["minority_recall"].iloc[0] - c["minority_recall"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _hidden(raw: pd.DataFrame, auroc_th: float = 0.03, recall_th: float = 0.10, auprc_th: float | None = None) -> pd.DataFrame:
    p0 = _primary(raw)
    rows = []
    for (ds, seed, model), grp in p0.groupby(["dataset", "seed", "model"]):
        iid = grp[grp["environment"] == "mcar_05"]
        if iid.empty:
            continue
        a0, ap0, r0 = (float(iid["AUROC"].iloc[0]), float(iid["AUPRC"].iloc[0]), float(iid["minority_recall"].iloc[0]))
        for _, r in grp[grp["environment"] != "mcar_05"].iterrows():
            auroc_drop = a0 - float(r["AUROC"])
            auprc_drop = ap0 - float(r["AUPRC"])
            recall_drop = r0 - float(r["minority_recall"])
            if auprc_th is None:
                flag = (auroc_drop <= auroc_th) and (recall_drop >= recall_th)
            else:
                flag = (auprc_drop <= auprc_th) and (recall_drop >= recall_th)
            rows.append(
                {
                    "dataset": ds, "seed": seed, "model": model, "environment": r["environment"],
                    "AUROC_drop": auroc_drop, "AUPRC_drop": auprc_drop,
                    "minority_recall_drop": recall_drop, "hidden": bool(flag),
                }
            )
    return pd.DataFrame(rows)


def _pool_stats(sel: pd.DataFrame) -> dict:
    mir = sel["MiRD_recall"].dropna().to_numpy()
    mar = sel["MaRD_recall"].dropna().to_numpy()
    mvgv = sel["MVG_recall"].dropna().to_numpy()
    if mir.size < 3:
        return {}
    _, p = wilcoxon_paired(mar, mir)
    rbs = rank_biserial(mar, mir)
    dz = cohens_dz(mar, mir)
    lo, hi = bootstrap_ci(mvgv, level=0.95, iters=2000)
    return {
        "wilcoxon_p": float(p), "rank_biserial": float(rbs), "cohens_dz": float(dz),
        "mvg_ci_low": float(lo), "mvg_ci_high": float(hi), "n": int(mir.size),
    }


def rc_lodo(raw: pd.DataFrame) -> pd.DataFrame:
    vuln = _vuln_pool(raw)
    slopes = _slopes(raw)
    ccep = _ccep_cc30(raw)
    hidden = _hidden(raw)
    rows = []
    for ex in sorted(vuln["dataset"].unique()):
        v = vuln[vuln["dataset"] != ex]
        s = slopes[slopes["dataset"] != ex]
        c = ccep[ccep["dataset"] != ex]
        h = hidden[hidden["dataset"] != ex]
        st = _pool_stats(v)
        rows.append(
            {
                "excluded_dataset": ex, "pooled_mvg_recall": float(v["MVG_recall"].mean()),
                "pooled_mvg_f1": float(v["MVG_f1"].mean()),
                "minority_slope": float(s["minority_slope"].mean()),
                "majority_slope": float(s["majority_slope"].mean()),
                "slope_gap": float(s["slope_gap"].mean()),
                "mean_ccep": float(c["CCEP_recall"].mean()) if not c.empty else np.nan,
                "hidden_failure_count": int(h["hidden"].sum()),
                "hidden_failure_prevalence": float(h["hidden"].mean()) if not h.empty else np.nan,
                **st,
            }
        )
    return pd.DataFrame(rows)


def rc_lomo(raw: pd.DataFrame) -> pd.DataFrame:
    vuln = _vuln_pool(raw)
    slopes = _slopes(raw)
    ccep = _ccep_cc30(raw)
    hidden = _hidden(raw)
    rows = []
    for ex in sorted(vuln["model"].unique()):
        v = vuln[vuln["model"] != ex]
        s = slopes[slopes["model"] != ex]
        c = ccep[ccep["model"] != ex]
        h = hidden[hidden["model"] != ex]
        st = _pool_stats(v)
        rows.append(
            {
                "excluded_model": ex, "pooled_mvg_recall": float(v["MVG_recall"].mean()),
                "slope_gap": float(s["slope_gap"].mean()),
                "mean_ccep": float(c["CCEP_recall"].mean()) if not c.empty else np.nan,
                "hidden_failure_count": int(h["hidden"].sum()),
                "hidden_failure_prevalence": float(h["hidden"].mean()) if not h.empty else np.nan,
                **st,
            }
        )
    return pd.DataFrame(rows)


def rc_seed(raw: pd.DataFrame) -> pd.DataFrame:
    vuln = _vuln_pool(raw)
    ccep = _ccep_cc30(raw)
    hidden = _hidden(raw)
    seeds = sorted(vuln["seed"].unique())
    combos = {
        "seed_42": [42], "seed_52": [52], "seed_62": [62],
        "42+52": [42, 52], "42+62": [42, 62], "52+62": [52, 62], "all": seeds,
    }
    rows = []
    for name, seed_list in combos.items():
        v = vuln[vuln["seed"].isin(seed_list)]
        s = _slopes(raw[raw["seed"].isin(seed_list)])
        c = ccep[ccep["seed"].isin(seed_list)]
        h = hidden[hidden["seed"].isin(seed_list)]
        rows.append(
            {
                "combo": name, "mean_mvg": float(v["MVG_recall"].mean()),
                "mean_ccep": float(c["CCEP_recall"].mean()) if not c.empty else np.nan,
                "minority_slope": float(s["minority_slope"].mean()),
                "majority_slope": float(s["majority_slope"].mean()),
                "slope_gap": float(s["slope_gap"].mean()),
                "hidden_failure_count": int(h["hidden"].sum()),
            }
        )
    return pd.DataFrame(rows)


def rc_severity(raw: pd.DataFrame) -> pd.DataFrame:
    scenarios = {
        "full_5_40": [0.05, 0.10, 0.20, 0.30, 0.40],
        "moderate_5_30": [0.05, 0.10, 0.20, 0.30],
        "low_to_medium_5_20": [0.05, 0.10, 0.20],
    }
    rows = []
    for name, rates in scenarios.items():
        s = _slopes(raw, rates)
        rows.append(
            {
                "scenario": name, "minority_slope": float(s["minority_slope"].mean()),
                "majority_slope": float(s["majority_slope"].mean()),
                "slope_gap": float(s["slope_gap"].mean()), "n_dataset_model": int(s.shape[0]),
            }
        )
    return pd.DataFrame(rows)


def rc_ccep(raw: pd.DataFrame) -> pd.DataFrame:
    ccep = _ccep_cc30(raw)
    vals = ccep["CCEP_recall"].dropna().to_numpy()
    lo, hi = bootstrap_ci(vals, level=0.95, iters=5000)
    ds_pos = ccep.groupby("dataset")["CCEP_recall"].mean()
    return pd.DataFrame(
        [
            {"metric": "CCEP_recall_cc30", "mean": float(vals.mean()), "median": float(np.median(vals)),
             "ci_low": float(lo), "ci_high": float(hi), "positive_fraction": float((vals > 0).mean()),
             "n_cells": int(vals.size), "n_datasets_positive": int((ds_pos > 0).sum()),
             "n_datasets": int(len(ds_pos))}
        ]
    )


def rc_reverse(raw: pd.DataFrame) -> pd.DataFrame:
    p0 = _primary(raw)
    rows = []
    for (ds, seed, model), grp in p0.groupby(["dataset", "seed", "model"]):
        iid = grp[grp["environment"] == "mcar_05"]
        if iid.empty:
            continue
        b = iid.iloc[0]

        def cell_mvg(env):
            sub = grp[grp["environment"] == env]
            if sub.empty:
                return np.nan
            r = sub.iloc[0]
            mi = relative_degradation(float(b["minority_recall"]), float(r["minority_recall"]), EPS)
            ma = relative_degradation(float(b["majority_recall"]), float(r["majority_recall"]), EPS)
            return mvg(mi, ma)

        normal = cell_mvg("cc_30")
        reverse = cell_mvg("reverse_cc_30")
        if not (np.isfinite(normal) and np.isfinite(reverse)):
            continue
        rows.append(
            {
                "dataset": ds, "seed": seed, "model": model,
                "normal_cc_MVG": normal, "reverse_cc_MVG": reverse,
                "reverse_exposure_effect": reverse_exposure_effect(normal, reverse),
            }
        )
    return pd.DataFrame(rows)


def rc_natural(raw: pd.DataFrame) -> pd.DataFrame:
    vuln = _vuln_pool(raw)
    slopes = _slopes(raw)
    ccep = _ccep_cc30(raw)
    hidden = _hidden(raw)
    rows = []
    for label, group in [("zero_natural_missing", None), ("natural_missing", None)]:
        ds_set = (set(vuln["dataset"].unique()) - NATURAL_MISSING) if label == "zero_natural_missing" else NATURAL_MISSING
        v = vuln[vuln["dataset"].isin(ds_set)]
        s = slopes[slopes["dataset"].isin(ds_set)]
        c = ccep[ccep["dataset"].isin(ds_set)]
        h = hidden[hidden["dataset"].isin(ds_set)]
        rows.append(
            {
                "group": label, "n_datasets": len(ds_set),
                "mean_mvg_recall": float(v["MVG_recall"].mean()),
                "mean_ccep": float(c["CCEP_recall"].mean()) if not c.empty else np.nan,
                "slope_gap": float(s["slope_gap"].mean()),
                "hidden_failure_prevalence": float(h["hidden"].mean()) if not h.empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def rc_imputation(raw: pd.DataFrame) -> pd.DataFrame:
    configs = {
        "P0_median": (raw["pipeline"] == "p0") & (raw["imputer"] == "median") & (raw["feature_set"] == "top"),
        "P1_indicator": (raw["pipeline"] == "p1") & (raw["imputer"] == "median") & (raw["feature_set"] == "top"),
        "native_missing": (raw["pipeline"] == "native_missing") & (raw["imputer"] == "median") & (raw["feature_set"] == "top"),
        "KNN": (raw["pipeline"] == "p0") & (raw["imputer"] == "knn") & (raw["feature_set"] == "top"),
        "Iterative": (raw["pipeline"] == "p0") & (raw["imputer"] == "iterative") & (raw["feature_set"] == "top"),
    }
    keys = ["dataset", "seed", "model", "environment"]
    subsets = {name: raw[mask].copy() for name, mask in configs.items()}
    subset_keys = {name: {tuple(x) for x in sub[keys].drop_duplicates().itertuples(index=False)} for name, sub in subsets.items()}
    common = set.intersection(*subset_keys.values()) if subset_keys else set()
    rows = []
    for name, sub in subsets.items():
        sub = sub[sub.apply(lambda r: tuple(r[keys]) in common, axis=1)]
        vals = []
        for (ds, seed, model), grp in sub.groupby(["dataset", "seed", "model"]):
            iid = grp[grp["environment"] == "mcar_05"]
            if iid.empty:
                continue
            b = iid.iloc[0]
            for _, r in grp[grp["environment"] != "mcar_05"].iterrows():
                if tuple(r[keys]) not in common:
                    continue
                mi = relative_degradation(float(b["minority_recall"]), float(r["minority_recall"]), EPS)
                ma = relative_degradation(float(b["majority_recall"]), float(r["majority_recall"]), EPS)
                vals.append(mvg(mi, ma))
        rows.append(
            {
                "config": name, "n_common_cells": int(len(common)),
                "mean_MVG_recall": float(np.mean(vals)) if vals else np.nan,
                "n_cells_used": int(len(vals)),
            }
        )
    return pd.DataFrame(rows)


def rc_masking(raw: pd.DataFrame) -> pd.DataFrame:
    hidden = _hidden(raw)
    from scipy.stats import spearmanr

    x = hidden["minority_recall_drop"].to_numpy()
    ya = hidden["AUROC_drop"].to_numpy()
    yp = hidden["AUPRC_drop"].to_numpy()
    r_ra = spearmanr(x, ya)[0]
    r_rp = spearmanr(x, yp)[0]
    f10 = float(((x >= 0.10) & (ya <= 0.03)).mean())
    f20 = float(((x >= 0.20) & (ya <= 0.03)).mean())
    return pd.DataFrame(
        [
            {"statistic": "spearman_recall_auroc", "value": float(r_ra), "n": int(len(x))},
            {"statistic": "spearman_recall_auprc", "value": float(r_rp), "n": int(len(x))},
            {"statistic": "fraction_recall_ge_010_auroc_le_003", "value": f10, "n": int(len(x))},
            {"statistic": "fraction_recall_ge_020_auroc_le_003", "value": f20, "n": int(len(x))},
        ]
    )


def rc_hidden_thresholds(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    defs = {
        "Primary": dict(auroc_th=0.03, recall_th=0.10),
        "S1": dict(auroc_th=0.01, recall_th=0.10),
        "S2": dict(auroc_th=0.02, recall_th=0.10),
        "S3": dict(auroc_th=0.03, recall_th=0.20),
        "S4": dict(auroc_th=0.02, recall_th=0.20),
        "S5": dict(auprc_th=0.03, recall_th=0.10),
        "S6": dict(auprc_th=0.02, recall_th=0.20),
    }
    rows = []
    for name, kw in defs.items():
        h = _hidden(raw, **kw)
        pos = h[h["hidden"]]
        rows.append(
            {
                "definition": name, "n_cases": int(pos.shape[0]),
                "n_datasets": int(pos["dataset"].nunique()),
                "n_models": int(pos["model"].nunique()),
                "n_mechanisms": int(pos["environment"].nunique()),
                "prevalence": float(h["hidden"].mean()) if not h.empty else np.nan,
            }
        )
    ex = _hidden(raw)
    ex = ex[ex["hidden"]].copy()
    return pd.DataFrame(rows), ex


def rc_mamr_negative() -> pd.DataFrame:
    e3_path = PROJECT_ROOT / "results" / "phase_e3_mamr" / "method_gate.json"
    e3r_path = PROJECT_ROOT / "results" / "phase_e3r_mamr_safety" / "safety_gate.json"
    if not (e3_path.exists() and e3r_path.exists()):
        return pd.DataFrame(
            [{"method": "N/A (MAMR artifacts absent)", "minority_gain": np.nan,
              "majority_cost": np.nan, "auprc_delta": np.nan, "big": np.nan,
              "mgr": np.nan, "mhr": np.nan, "gate_decision": "N/A"}]
        )
    e3 = json.loads(e3_path.read_text())
    e3r = json.loads(e3r_path.read_text())
    em, e3m = e3["metrics"], e3r["metrics"]
    return pd.DataFrame(
        [
            {"method": "best_standard_imbalance_baseline", "minority_gain": 0.0, "majority_cost": 0.0,
             "auprc_delta": 0.0, "big": np.nan, "mgr": np.nan, "mhr": np.nan, "gate_decision": "reference"},
            {"method": "Original_MAMR (E3)", "minority_gain": em["mean_minority_recall_gain_vs_best_standard"],
             "majority_cost": em["mean_majority_recall_delta"], "auprc_delta": em["mean_auprc_delta"],
             "big": em["mean_big"], "mgr": 1.0, "mhr": 0.0, "gate_decision": e3["gate"]},
            {"method": "Safe-MAMR (E3R)", "minority_gain": e3m["minority_gain_safe"],
             "majority_cost": e3m["majority_delta_safe"], "auprc_delta": e3m["auprc_delta_safe"],
             "big": e3m["big_mean"], "mgr": e3m["mgr"], "mhr": e3m["majority_harm_reduction"],
             "gate_decision": e3r["gate"]},
        ]
    )


def _figures(ms_detail, lodo, sev, ccep, reverse, outdir) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    out = []

    def save(fig, name):
        p = outdir / name
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out.append(p)

    fig, ax = plt.subplots()
    labels = ["minority_recall", "AUROC", "AUPRC", "balanced_accuracy", "Gmean"]
    vals = [ms_detail["MiRD_recall"].mean()]
    for m in ["AUROC", "AUPRC", "balanced_accuracy", "Gmean"]:
        vals.append(ms_detail[f"{m}_rel_drop"].mean())
    ax.bar(labels, vals, color=["#d62728", "#7f7f7f", "#1f77b4", "#2ca02c", "#9467bd"])
    ax.set_ylabel("mean relative drop")
    save(fig, "fig_s_rc1_metric_sensitivity.png")

    fig, ax = plt.subplots()
    ax.bar(lodo["excluded_dataset"], lodo["pooled_mvg_recall"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(len(lodo)))
    ax.set_xticklabels(lodo["excluded_dataset"], rotation=45, ha="right")
    ax.set_ylabel("pooled MVG_recall")
    save(fig, "fig_s_rc2_lodo_mvg.png")

    fig, ax = plt.subplots()
    ax.bar(sev["scenario"], sev["slope_gap"])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_ylabel("slope gap")
    save(fig, "fig_s_rc3_severity_slope_gap.png")

    fig, ax = plt.subplots()
    grp = ccep.groupby("dataset")[["matched_mcar_recall", "cc_recall"]].mean()
    grp = grp.sort_values("matched_mcar_recall")
    x = np.arange(len(grp))
    ax.bar(x - 0.18, grp["matched_mcar_recall"], 0.34, label="matched MCAR")
    ax.bar(x + 0.18, grp["cc_recall"], 0.34, label="class-conditional (CC)")
    ax.set_xticks(x)
    ax.set_xticklabels(grp.index, rotation=45, ha="right")
    ax.set_ylabel("minority recall")
    ax.legend()
    save(fig, "fig_s_rc4_matched_mcar_vs_cc.png")

    fig, ax = plt.subplots()
    grp = reverse.groupby("dataset")[["normal_cc_MVG", "reverse_cc_MVG"]].mean().reset_index()
    x = np.arange(len(grp))
    ax.bar(x - 0.18, grp["normal_cc_MVG"], 0.34, label="normal CC")
    ax.bar(x + 0.18, grp["reverse_cc_MVG"], 0.34, label="reverse CC")
    ax.set_xticks(x)
    ax.set_xticklabels(grp["dataset"], rotation=45, ha="right")
    ax.set_ylabel("MVG_recall")
    ax.legend()
    save(fig, "fig_s_rc5_normal_vs_reverse_cc.png")

    return out


def _build_report(v, figures, outdir) -> str:
    lines = [
        "# Project E Final RC — Reviewer-Proof Supplementary Validation",
        "",
        "**Experiment status: `FROZEN`**",
        "",
        "## Reviewer checks",
        "",
    ]
    for name, val in v.items():
        lines.append(f"- **{name}**: {val}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    for p in sorted(outdir.glob("*.csv")):
        lines.append(f"- {p.name}")
    for p in figures:
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("## Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/reproduce_phase_e_final_rc.py")
    lines.append("```")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Project E Final RC reviewer-proof supplementary validation")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()
    outdir = Path(args.results) if args.results else (PROJECT_ROOT / "results" / "phase_e_final_rc")
    outdir.mkdir(parents=True, exist_ok=True)
    raw = _raw()

    ms_detail, ms_summary = compute_metric_sensitivity(raw)
    ms_detail.to_csv(outdir / "metric_sensitivity.csv", index=False)
    ms_summary.to_csv(outdir / "metric_sensitivity_summary.csv", index=False)
    ht, hex = rc_hidden_thresholds(raw)
    ht.to_csv(outdir / "hidden_failure_threshold_sensitivity.csv", index=False)
    hex.to_csv(outdir / "hidden_failure_examples_rc.csv", index=False)
    lodo = rc_lodo(raw)
    lodo.to_csv(outdir / "lodo_dataset_influence.csv", index=False)
    lomo = rc_lomo(raw)
    lomo.to_csv(outdir / "lomo_model_influence.csv", index=False)
    seed = rc_seed(raw)
    seed.to_csv(outdir / "seed_sensitivity.csv", index=False)
    sev = rc_severity(raw)
    sev.to_csv(outdir / "severity_exclusion.csv", index=False)
    ccep = rc_ccep(raw)
    ccep.to_csv(outdir / "ccep_reviewer_check.csv", index=False)
    ccep_detail = _ccep_cc30(raw)
    reverse = rc_reverse(raw)
    reverse.to_csv(outdir / "reverse_cc_reviewer_check.csv", index=False)
    nat = rc_natural(raw)
    nat.to_csv(outdir / "natural_missingness_sensitivity.csv", index=False)
    imp = rc_imputation(raw)
    imp.to_csv(outdir / "imputation_sensitivity.csv", index=False)
    mask = rc_masking(raw)
    mask.to_csv(outdir / "aggregate_masking_quantification.csv", index=False)
    neg = rc_mamr_negative()
    neg.to_csv(outdir / "mamr_negative_result_summary.csv", index=False)

    v = {}
    apu = ms_summary.loc[ms_summary["question"] == "Q4_auprc_rel_drop", "value"].iloc[0]
    arc = ms_summary.loc[ms_summary["question"] == "Q3_auroc_rel_drop", "value"].iloc[0]
    v["RC1_metric_sensitivity"] = "PASS" if apu > arc else "REVIEW"
    v["RC2_hidden_failure_threshold"] = "PASS" if (ht["n_datasets"] >= 2).sum() >= 3 else "SENSITIVE"
    v["RC3_lodo"] = "PASS" if (lodo["pooled_mvg_recall"] > 0).all() and (lodo["slope_gap"] > 0).all() else "SENSITIVE"
    v["RC4_lomo"] = "PASS" if (lomo["pooled_mvg_recall"] > 0).all() and (lomo["slope_gap"] > 0).all() else "SENSITIVE"
    v["RC5_seed"] = "PASS" if (seed[seed["combo"].str.startswith("seed_")]["mean_mvg"] > 0).all() else "SENSITIVE"
    v["RC6_severity"] = "PASS" if float(sev[sev["scenario"] == "moderate_5_30"]["slope_gap"].iloc[0]) > 0 else "SENSITIVE"
    v["RC7_ccep"] = "PASS" if float(ccep["positive_fraction"].iloc[0]) >= 0.6 and float(ccep["mean"].iloc[0]) > 0 else "REVIEW"
    v["RC8_reverse_cc"] = "PASS" if float((reverse["reverse_exposure_effect"] > 0).mean()) >= 0.6 else "REVIEW"
    v["RC9_natural_missingness"] = "DESCRIPTIVE"
    v["RC10_imputation"] = "PASS" if imp.loc[imp["config"] != "P0_median", "mean_MVG_recall"].notna().all() and (imp.loc[imp["config"] != "P0_median", "mean_MVG_recall"] > 0).all() else "REVIEW"
    v["RC11_aggregate_masking"] = "QUANTIFIED"
    mamr_artifacts_present = (
        (PROJECT_ROOT / "results" / "phase_e3_mamr" / "method_gate.json").exists()
        and (PROJECT_ROOT / "results" / "phase_e3r_mamr_safety" / "safety_gate.json").exists()
    )
    v["RC12_mamr_negative"] = "NEGATIVE_RESULT" if mamr_artifacts_present else "N/A"

    figures = _figures(ms_detail, lodo, sev, ccep_detail, reverse, outdir / "figures")
    (outdir / "phase_e_final_rc_report.md").write_text(_build_report(v, figures, outdir), encoding="utf-8")
    manifest = {
        "timestamp": datetime.now().isoformat(), "python_version": platform.python_version(),
        "experiment_status": "FROZEN", "datasets": sorted(raw["dataset"].unique()),
        "models": sorted(raw["model"].unique()), "seeds": sorted(int(s) for s in raw["seed"].unique()),
        "verdicts": v,
    }
    (outdir / "manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    (outdir / "progress.json").write_text(json.dumps({"completed": True, "status": "FROZEN", "timestamp": manifest["timestamp"]}, indent=2), encoding="utf-8")
    (outdir / "failed_runs.csv").write_text("run_key,error,timestamp\n", encoding="utf-8")

    print("\n# Project E Final RC — Reviewer-Proof Supplementary Validation")
    print("Experiment status: FROZEN")
    for k, val in v.items():
        print(f"{k}: {val}")
    print("Authoritative artifacts: results/phase_e_final_rc/")
    print("Reproduce: python scripts/reproduce_phase_e_final_rc.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
