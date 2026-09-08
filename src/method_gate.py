"""Phase E3 Method Gate: derived metrics and GO / HOLD / STOP decision."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .statistical_analysis import (
    bh_fdr,
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


def _iid(grp: pd.DataFrame) -> pd.Series | None:
    sub = grp[grp["environment"] == "mcar_05"]
    if sub.empty:
        return None
    return sub.iloc[0]


def compute_lrr(
    raw: pd.DataFrame,
    methods: pd.Series,
    shift_envs: list[str],
    include_filter: bool = True,
) -> pd.DataFrame:
    """Lost Recall Recovery of each method relative to ERM."""
    raw = _ok(raw)
    rows = []
    for (ds, model, seed, method), grp in raw.groupby(
        ["dataset", "model", "seed", "method"]
    ):
        erm = raw[
            (raw["dataset"] == ds)
            & (raw["model"] == model)
            & (raw["seed"] == seed)
            & (raw["method"] == "erm")
        ]
        if erm.empty:
            continue
        iid = _iid(erm)
        if iid is None:
            continue
        r_iid = float(iid["minority_recall"])
        for env in shift_envs:
            r = grp[grp["environment"] == env]
            base_r = erm[erm["environment"] == env]
            if r.empty or base_r.empty:
                continue
            r_shift = float(base_r["minority_recall"].iloc[0])
            r_method = float(r["minority_recall"].iloc[0])
            denom = r_iid - r_shift
            lrr = (r_method - r_shift) / (denom + EPS)
            if include_filter and denom <= 0.02:
                lrr = np.nan
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "environment": env,
                    "erm_iid_recall": r_iid,
                    "erm_shift_recall": r_shift,
                    "method_shift_recall": r_method,
                    "erm_recall_drop": r_iid - r_shift,
                    "lrr": lrr,
                }
            )
    return pd.DataFrame(rows)


def compute_iid_cost(
    raw: pd.DataFrame, mamr_method: str = "mamr_full"
) -> pd.DataFrame:
    """IID (mcar_05) cost of MAMR relative to ERM."""
    raw = _ok(raw)
    rows = []
    for (ds, model, seed), grp in raw.groupby(["dataset", "model", "seed"]):
        erm = grp[(grp["method"] == "erm") & (grp["environment"] == "mcar_05")]
        mamr = grp[(grp["method"] == mamr_method) & (grp["environment"] == "mcar_05")]
        if erm.empty or mamr.empty:
            continue
        e, m = erm.iloc[0], mamr.iloc[0]
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "seed": seed,
                "iid_minority_recall_delta": float(m["minority_recall"] - e["minority_recall"]),
                "iid_auprc_delta": float(m["AUPRC"] - e["AUPRC"]),
                "erm_iid_minority_recall": float(e["minority_recall"]),
                "mamr_iid_minority_recall": float(m["minority_recall"]),
                "erm_iid_auprc": float(e["AUPRC"]),
                "mamr_iid_auprc": float(m["AUPRC"]),
            }
        )
    return pd.DataFrame(rows)


def compute_big(
    raw: pd.DataFrame,
    best_standard: pd.DataFrame,
    mamr_method: str = "mamr_full",
    shift_envs: list[str] | None = None,
) -> pd.DataFrame:
    """Beyond-Imbalance Gain: MAMR vs best standard minority recall."""
    raw = _ok(raw)
    rows = []
    for (ds, model, seed, env), grp in raw.groupby(["dataset", "model", "seed", "environment"]):
        best = best_standard[
            (best_standard["dataset"] == ds)
            & (best_standard["model"] == model)
            & (best_standard["seed"] == seed)
            & (best_standard["environment"] == env)
        ]
        mamr = grp[(grp["method"] == mamr_method)]
        if best.empty or mamr.empty or best.iloc[0]["best_standard_method"] == mamr_method:
            continue
        r_best = float(best.iloc[0]["best_standard_minority_recall"])
        r_mamr = float(mamr.iloc[0]["minority_recall"])
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "seed": seed,
                "environment": env,
                "best_standard_method": best.iloc[0]["best_standard_method"],
                "best_standard_minority_recall": r_best,
                "mamr_minority_recall": r_mamr,
                "big": r_mamr - r_best,
                "best_standard_majority_recall": float(best.iloc[0]["best_standard_majority_recall"]),
                "best_standard_auprc": float(best.iloc[0]["best_standard_auprc"]),
                "mamr_majority_recall": float(mamr.iloc[0]["majority_recall"]),
                "mamr_auprc": float(mamr.iloc[0]["AUPRC"]),
            }
        )
    df = pd.DataFrame(rows)
    if shift_envs is not None:
        df = df[df["environment"].isin(shift_envs)]
    return df


def compute_best_standard(raw: pd.DataFrame) -> pd.DataFrame:
    """Best standard imbalance method by minority recall per cell."""
    raw = _ok(raw)
    standard = raw[raw["method"].isin(["class_weight", "random_oversampling", "missing_indicator"])]
    rows = []
    for (ds, model, seed, env), grp in standard.groupby(["dataset", "model", "seed", "environment"]):
        if grp.empty:
            continue
        best = grp.sort_values("minority_recall", ascending=False).iloc[0]
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "seed": seed,
                "environment": env,
                "best_standard_method": best["method"],
                "best_standard_minority_recall": float(best["minority_recall"]),
                "best_standard_majority_recall": float(best["majority_recall"]),
                "best_standard_auprc": float(best["AUPRC"]),
                "best_standard_balanced_accuracy": float(best["balanced_accuracy"]),
                "best_standard_mcc": float(best["MCC"]),
            }
        )
    return pd.DataFrame(rows)


def compute_mas(
    raw: pd.DataFrame,
    high_risk: list[str],
    control: list[str],
    mamr_method: str = "mamr_full",
) -> dict[str, float]:
    """Mechanism Alignment Score for MAMR vs ERM."""
    raw = _ok(raw)
    deltas = []
    for (ds, model, seed, env), grp in raw.groupby(["dataset", "model", "seed", "environment"]):
        erm = grp[grp["method"] == "erm"]
        mamr = grp[grp["method"] == mamr_method]
        if erm.empty or mamr.empty:
            continue
        d = float(mamr.iloc[0]["minority_recall"] - erm.iloc[0]["minority_recall"])
        deltas.append(
            {
                "dataset": ds,
                "model": model,
                "seed": seed,
                "environment": env,
                "delta": d,
                "group": "high-risk" if env in high_risk else (
                    "control" if env in control else "other"
                ),
            }
        )
    df = pd.DataFrame(deltas)
    if df.empty:
        return {"mas": np.nan, "high_risk_gain": np.nan, "control_gain": np.nan}
    hr = df[df["group"] == "high-risk"]["delta"]
    ct = df[df["group"] == "control"]["delta"]
    hr_mean = float(hr.mean()) if len(hr) else np.nan
    ct_mean = float(ct.mean()) if len(ct) else np.nan
    return {
        "mas": (hr_mean - ct_mean) if np.isfinite(hr_mean) and np.isfinite(ct_mean) else np.nan,
        "high_risk_gain": hr_mean,
        "control_gain": ct_mean,
        "high_risk_n": int(len(hr)),
        "control_n": int(len(ct)),
    }


def _dataset_mean(raw: pd.DataFrame, metric: str) -> pd.Series:
    return raw.groupby("dataset")[metric].mean()


def evaluate_gate(
    raw: pd.DataFrame,
    cfg: Any,
    compute_all: bool = True,
) -> dict[str, Any]:
    """Compute the eight Method Gate conditions and return the verdict JSON."""
    raw = _ok(raw)
    best_standard = compute_best_standard(raw)
    hr = list(cfg.high_risk_mechanisms)
    control = list(cfg.control_mechanisms)
    shift_envs = [e for e in cfg.environments if e != "mcar_05"]
    gate = cfg.gate

    # Merge MAMR cells with best_standard for the delta tables.
    mamr_cells = raw[raw["method"] == "mamr_full"]
    merged = mamr_cells.merge(
        best_standard,
        on=["dataset", "model", "seed", "environment"],
        how="left",
    )
    merged = merged[merged["environment"].isin(hr)]
    merged["minority_recall_gain"] = merged["minority_recall"] - merged["best_standard_minority_recall"]
    merged["majority_recall_delta"] = merged["majority_recall"] - merged["best_standard_majority_recall"]
    merged["auprc_delta"] = merged["AUPRC"] - merged["best_standard_auprc"]
    merged["balanced_accuracy_delta"] = merged["balanced_accuracy"] - merged["best_standard_balanced_accuracy"]
    merged["mcc_delta"] = merged["MCC"] - merged["best_standard_mcc"]

    # GO-A: datasets with mean high-risk minority gain >= threshold.
    if merged.empty:
        ds_gain_ge = 0
    else:
        ds_gain = merged.groupby("dataset")["minority_recall_gain"].mean()
        ds_gain_ge = int((ds_gain >= float(gate["go_a_min_gain"])).sum())
    GO_A = ds_gain_ge >= int(gate["go_a_min_datasets"])

    # GO-B: LRR on collapse cases.
    lrr = compute_lrr(raw, pd.Series(["mamr_full"]), shift_envs, include_filter=False)
    collapse = lrr[(lrr["method"] == "mamr_full") & (lrr["erm_recall_drop"] >= float(gate["go_b_min_recall_drop"]))]
    lrr_vals = collapse["lrr"].dropna()
    if len(lrr_vals):
        median_lrr = float(lrr_vals.median())
        pos_frac = float((lrr_vals > 0).mean())
    else:
        median_lrr = np.nan
        pos_frac = np.nan
    GO_B = (not np.isnan(median_lrr)) and median_lrr >= float(gate["go_b_min_lrr"]) and pos_frac >= float(gate["go_b_min_positive_fraction"])

    # GO-C: majority recall delta vs best_standard.
    if merged.empty:
        maj_mean, maj_worst = np.nan, np.nan
    else:
        maj_mean = float(merged["majority_recall_delta"].mean())
        maj_worst = float(merged.groupby("dataset")["majority_recall_delta"].mean().min())
    GO_C = (not np.isnan(maj_mean)) and maj_mean >= float(gate["go_c_mean_minus"]) and maj_worst >= float(gate["go_c_worst_minus"])

    # GO-D: AUPRC delta vs best_standard.
    if merged.empty:
        ap_mean, ap_worst = np.nan, np.nan
    else:
        ap_mean = float(merged["auprc_delta"].mean())
        ap_worst = float(merged.groupby("dataset")["auprc_delta"].mean().min())
    GO_D = (not np.isnan(ap_mean)) and ap_mean >= float(gate["go_d_mean_minus"]) and ap_worst >= float(gate["go_d_worst_minus"])

    # Balanced-accuracy / MCC deltas (for signal detection).
    if merged.empty:
        ba_mean, mcc_mean = np.nan, np.nan
    else:
        ba_mean = float(merged["balanced_accuracy_delta"].mean())
        mcc_mean = float(merged["mcc_delta"].mean())

    # GO-E: IID safety vs ERM.
    iid = compute_iid_cost(raw)
    if iid.empty:
        iid_mr, iid_ap = np.nan, np.nan
    else:
        iid_mr = float(iid["iid_minority_recall_delta"].mean())
        iid_ap = float(iid["iid_auprc_delta"].mean())
    GO_E = (not np.isnan(iid_mr)) and iid_mr >= float(gate["go_e_min_recall"]) and iid_ap >= float(gate["go_e_min_auprc"])

    # GO-F: BIG.
    big = compute_big(raw, best_standard, "mamr_full", shift_envs)
    if big.empty:
        big_mean, big_pos_frac = np.nan, np.nan
    else:
        big_mean = float(big["big"].mean())
        big_pos_frac = float((big["big"] > 0).mean())
    GO_F = (not np.isnan(big_mean)) and big_mean > float(gate["go_f_min_mean"]) and big_pos_frac >= float(gate["go_f_min_positive_fraction"])

    # GO-G: MAS.
    mas = compute_mas(raw, hr, control)
    mas_val = float(mas["mas"]) if np.isfinite(mas["mas"]) else np.nan
    hr_gain = float(mas["high_risk_gain"]) if np.isfinite(mas["high_risk_gain"]) else np.nan
    ct_gain = float(mas["control_gain"]) if np.isfinite(mas["control_gain"]) else np.nan
    GO_G = (not np.isnan(mas_val)) and mas_val >= float(gate["go_g_min_mas"]) and hr_gain > ct_gain

    # GO-H: seed consistency on high-risk shifts.
    if merged.empty:
        ds_seed_ok = 0
    else:
        seed_pos = merged.groupby(["dataset", "seed"])["minority_recall_gain"].mean()
        per_dataset = seed_pos.groupby("dataset").apply(lambda s: float((s > 0).mean()))
        ds_seed_ok = int((per_dataset >= float(gate["go_h_min_seed_fraction"])).sum())
    GO_H = ds_seed_ok >= int(gate["go_h_min_datasets"])

    conditions = {
        "GO_A": bool(GO_A),
        "GO_B": bool(GO_B),
        "GO_C": bool(GO_C),
        "GO_D": bool(GO_D),
        "GO_E": bool(GO_E),
        "GO_F": bool(GO_F),
        "GO_G": bool(GO_G),
        "GO_H": bool(GO_H),
    }
    n_hit = int(sum(1 for v in conditions.values() if v))

    # Signals that push towards STOP / HOLD.
    stop_signals: list[str] = []
    hold_signals: list[str] = []
    # Only call a minority gain "collapse-driven" when the gain is an illusion:
    # majority is sacrificed AND the threshold-free AUPRC AND the balanced
    # accuracy both worsen. Otherwise it is a large-but-real tradeoff (HOLD).
    if (
        not np.isnan(maj_mean)
        and maj_mean < -0.10
        and (not np.isnan(ap_mean) and ap_mean <= 0.0)
        and (not np.isnan(ba_mean) and ba_mean <= 0.0)
    ):
        stop_signals.append("minority recall gain is driven by majority collapse")
    if not np.isnan(iid_mr) and iid_mr < -0.05:
        stop_signals.append("IID minority recall is materially damaged")
    if not np.isnan(mas_val) and mas_val < 0.005:
        stop_signals.append("no high-risk vs control mechanism difference")
    # MAMR not better than class_weight / oversampling (negative mean BIG).
    if not np.isnan(big_mean) and big_mean <= 0.0:
        stop_signals.append("MAMR does not beat standard imbalance baselines on average")
    # A real majority cost is a HOLD signal rather than an immediate STOP.
    if not np.isnan(maj_mean) and maj_mean < float(gate["go_c_mean_minus"]):
        hold_signals.append("significant majority recall tradeoff")
    if not np.isnan(maj_worst) and not np.isnan(maj_mean) and (maj_mean - maj_worst) > 0.15:
        hold_signals.append("MAMR behaviour is unstable across datasets")

    verdict = _decide(
        conditions=conditions,
        n_hit=n_hit,
        stop_signals=stop_signals,
        hold_signals=hold_signals,
    )

    payload = {
        "gate": verdict,
        "conditions": conditions,
        "counts": {
            "n_hit": n_hit,
            "datasets_big_ge_050": int(ds_gain_ge) if not merged.empty else 0,
            "n_collapse_cases": int(len(collapse)) if not collapse.empty else 0,
            "positive_lrr_fraction": pos_frac,
            "median_lrr": median_lrr,
            "mas": mas_val,
            "high_risk_gain": hr_gain,
            "control_gain": ct_gain,
        },
        "metrics": {
            "mean_minority_recall_gain_vs_erm": _mean_delta_vs_erm(raw, hr),
            "mean_minority_recall_gain_vs_best_standard": float(merged["minority_recall_gain"].mean()) if not merged.empty else np.nan,
            "mean_majority_recall_delta": maj_mean,
            "worst_dataset_majority_recall_delta": maj_worst,
        "mean_auprc_delta": ap_mean,
        "worst_dataset_auprc_delta": ap_worst,
        "mean_balanced_accuracy_delta": ba_mean,
        "mean_mcc_delta": mcc_mean,
        "mean_iid_minority_recall_delta": iid_mr,
            "mean_iid_auprc_delta": iid_ap,
            "mean_big": big_mean,
            "big_positive_fraction": big_pos_frac,
        },
        "stop_signals": stop_signals,
        "hold_signals": hold_signals,
        "recommended_next_phase": (
            "Phase E4" if verdict in ("STRONG-GO", "GO") else (
                "Revise MAMR" if verdict == "HOLD" else "Stop MAMR"
            )
        ),
    }
    return validate_gate_consistency(payload)


def _mean_delta_vs_erm(raw: pd.DataFrame, high_risk: list[str]) -> float:
    raw = _ok(raw)
    deltas = []
    for (ds, model, seed, env), grp in raw.groupby(["dataset", "model", "seed", "environment"]):
        erm = grp[grp["method"] == "erm"]
        mamr = grp[grp["method"] == "mamr_full"]
        if erm.empty or mamr.empty or env not in high_risk:
            continue
        deltas.append(float(mamr.iloc[0]["minority_recall"] - erm.iloc[0]["minority_recall"]))
    return float(np.mean(deltas)) if deltas else np.nan


def _decide(
    conditions: dict[str, bool],
    n_hit: int,
    stop_signals: list[str],
    hold_signals: list[str],
) -> str:
    if n_hit <= 3 or stop_signals:
        return "STOP"
    if (
        n_hit >= 7
        and conditions["GO_A"]
        and conditions["GO_B"]
        and conditions["GO_C"]
        and conditions["GO_F"]
        and not hold_signals
    ):
        return "STRONG-GO"
    if (
        n_hit >= 6
        and conditions["GO_A"]
        and conditions["GO_C"]
        and conditions["GO_F"]
        and not hold_signals
    ):
        return "GO"
    if n_hit >= 4 or hold_signals:
        return "HOLD"
    return "STOP"


def validate_gate_consistency(payload: dict[str, Any]) -> dict[str, Any]:
    """Assert that the gate verdict, the GO conditions and the next-phase
    recommendation are mutually consistent.

    This is the single guard that keeps ``method_gate.json``, the report and the
    console verdict from diverging: a verdict of STRONG-GO / GO can only be
    reached when the mandatory GO_C (majority safety) condition is satisfied and
    no hold signal is active.
    """
    verdict = payload["gate"]
    conds = payload["conditions"]
    if verdict in ("STRONG-GO", "GO"):
        if not conds.get("GO_C", False):
            raise ValueError(
                f"Gate inconsistency: verdict={verdict!r} but GO_C=False. "
                "GO / STRONG-GO requires majority safety (GO_C)."
            )
        if payload.get("hold_signals"):
            raise ValueError(
                f"Gate inconsistency: verdict={verdict!r} but hold_signals "
                f"present: {payload['hold_signals']}."
            )
    expected = {
        "STRONG-GO": "Phase E4",
        "GO": "Phase E4",
        "HOLD": "Revise MAMR",
        "STOP": "Stop MAMR",
    }
    if payload.get("recommended_next_phase") != expected[verdict]:
        raise ValueError(
            f"Gate inconsistency: recommended_next_phase="
            f"{payload.get('recommended_next_phase')!r} but verdict={verdict!r}."
        )
    return payload

def run_statistics(
    raw: pd.DataFrame,
    cfg: Any,
    bootstrap_iterations: int = 2000,
    fdr_alpha: float = 0.05,
) -> list[dict[str, Any]]:
    """Paired MAMR-full vs best_standard tests on high-risk shifts."""
    raw = _ok(raw)
    best_standard = compute_best_standard(raw)
    hr = list(cfg.high_risk_mechanisms)
    mamr = raw[(raw["method"] == "mamr_full") & (raw["environment"].isin(hr))]
    mamr = mamr.merge(
        best_standard,
        on=["dataset", "model", "seed", "environment"],
        how="left",
    )
    metrics = {
        "minority_recall": ("best_standard_minority_recall", "minority_recall"),
        "auprc": ("best_standard_auprc", "AUPRC"),
        "majority_recall": ("best_standard_majority_recall", "majority_recall"),
    }
    rows = []
    for metric, (base_col, mamr_col) in metrics.items():
        a = pd.to_numeric(mamr[base_col], errors="coerce").to_numpy()
        b = pd.to_numeric(mamr[mamr_col], errors="coerce").to_numpy()
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        diff = b - a
        stat, p = wilcoxon_paired(a, b)
        t, pt = paired_ttest(a, b)
        lo, hi = bootstrap_ci(diff, level=cfg.statistics["confidence_level"],
                              iters=bootstrap_iterations)
        rbs = rank_biserial(a, b)
        dz = cohens_dz(a, b)
        rows.append(
            {
                "metric": metric,
                "n": int(len(diff)),
                "mean_delta": float(diff.mean()) if diff.size else np.nan,
                "median_delta": float(np.median(diff)) if diff.size else np.nan,
                "wilcoxon_stat": stat,
                "p_wilcoxon": p,
                "t_stat": t,
                "p_paired_t": pt,
                "boot_ci_low": float(lo),
                "boot_ci_high": float(hi),
                "rank_biserial": rbs,
                "cohens_dz": dz,
            }
        )
    # FDR correction over the minority-recall p-values of each mechanism.
    mechanism_rows = []
    for env in hr:
        sub = mamr[mamr["environment"] == env]
        if sub.empty:
            continue
        a = pd.to_numeric(sub["best_standard_minority_recall"], errors="coerce").to_numpy()
        b = pd.to_numeric(sub["minority_recall"], errors="coerce").to_numpy()
        mask = np.isfinite(a) & np.isfinite(b)
        a, b = a[mask], b[mask]
        stat, p = wilcoxon_paired(a, b)
        mechanism_rows.append({"environment": env, "p_wilcoxon": p, "n": int(len(a))})
    pvals = np.array([r["p_wilcoxon"] for r in mechanism_rows if np.isfinite(r["p_wilcoxon"])])
    if len(pvals):
        adj = bh_fdr(pvals, alpha=fdr_alpha)
        for r, p_adj in zip(mechanism_rows, adj):
            if np.isfinite(r["p_wilcoxon"]):
                r["p_fdr"] = float(p_adj)
    return rows + mechanism_rows
