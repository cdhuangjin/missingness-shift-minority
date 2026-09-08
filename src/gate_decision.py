"""Automatic Gate A decision and report generation."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd


def _p0(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["pipeline"] == "P0"]


def _direction_fraction(values: pd.Series) -> float:
    vals = values.dropna().astype(float)
    return float((vals > 0).mean()) if len(vals) else 0.0


def build_evidence(
    raw: pd.DataFrame,
    vuln: pd.DataFrame,
    slopes: pd.DataFrame,
    cfg: Any,
) -> pd.DataFrame:
    """Per (dataset, model) Table 6 evidence using the P0 pipeline."""
    raw_p0 = _p0(raw)
    vuln_p0 = _p0(vuln)
    gate = cfg.gate
    eps = cfg.thresholds["epsilon"]
    strong_mvg = float(gate["strong_mvg"])
    strong_recall_drop = float(gate["strong_recall_drop"])
    stable_auroc_drop = float(gate["stable_auroc_drop"])

    rows: list[dict[str, Any]] = []
    for (ds, model), group in vuln_p0.groupby(["dataset", "model"]):
        # Strongest MVG across shift environments.
        mv_groups = group.groupby("seed")
        strongest_mvg = float(group.groupby("seed")["MVG_recall"].max().mean())

        # Seed consistency: direction of the strongest MVG per seed.
        seed_best_mvg = group.groupby("seed")["MVG_recall"].max()
        seed_consistency = _direction_fraction(seed_best_mvg)

        # CC vs comparable MCAR minority recall (P0).
        cc = raw_p0[
            (raw_p0["dataset"] == ds)
            & (raw_p0["model"] == model)
            & (raw_p0["environment"] == "class_conditional")
        ]
        comp = raw_p0[
            (raw_p0["dataset"] == ds)
            & (raw_p0["model"] == model)
            & (raw_p0["environment"] == "mcar_comparable")
        ]
        ccep_rows = []
        seeded = {}
        for s in cfg.seeds:
            c_real = cc[cc["seed"] == s]["minority_recall"]
            c_comp = comp[comp["seed"] == s]["minority_recall"]
            if len(c_real) and len(c_comp):
                ccep_rows.append(float(c_comp.iloc[0] - c_real.iloc[0]))
                seeded[s] = bool(c_real.iloc[0] <= c_comp.iloc[0] - strong_recall_drop)
        ccep_recall = float(np.mean(ccep_rows)) if ccep_rows else np.nan
        cc_seed_consistency = _direction_fraction(
            pd.Series({k: (1 if v else 0) for k, v in seeded.items()})
        ) if seeded else 0.0

        # Hidden minority failure (per seed / per environment).
        hidden = []
        iid = raw_p0[
            (raw_p0["dataset"] == ds) & (raw_p0["model"] == model)
        ]
        hid_count = 0
        for s in cfg.seeds:
            iid_row = iid[(iid["seed"] == s) & (iid["environment"] == "iid_mcar")]
            if iid_row.empty:
                continue
            iid_auroc = float(iid_row["AUROC"].iloc[0])
            iid_recall = float(iid_row["minority_recall"].iloc[0])
            shifted = iid[(iid["seed"] == s) & (iid["environment"] != "iid_mcar")]
            for _, r in shifted.iterrows():
                auroc_drop = iid_auroc - float(r["AUROC"])
                recall_drop = iid_recall - float(r["minority_recall"])
                if auroc_drop <= stable_auroc_drop and recall_drop >= strong_recall_drop:
                    hid_count += 1
                    hidden.append({"seed": s, "environment": r["environment"]})
        seed_hidden = set(h["seed"] for h in hidden)
        hidden_seed_frac = len(seed_hidden) / float(len(cfg.seeds))

        # Indicator gain (P1 vs P0) for xgboost.
        mig_rows = []
        xgb_p0 = raw_p0[
            (raw_p0["dataset"] == ds) & (raw_p0["model"] == model)
        ]
        xgb_p1 = raw[
            (raw["dataset"] == ds)
            & (raw["model"] == model)
            & (raw["pipeline"] == "P1")
        ]
        for env in xgb_p0["environment"].unique():
            a = xgb_p0[xgb_p0["environment"] == env]["AUPRC"]
            b = xgb_p1[xgb_p1["environment"] == env]["AUPRC"]
            if len(a) and len(b):
                mig_rows.append(float(b.mean() - a.mean()))
        mig_auprc = float(np.mean(mig_rows)) if mig_rows else np.nan

        rows.append(
            {
                "dataset": ds,
                "model": model,
                "strongest_MVG": strongest_mvg,
                "CCEP_recall": ccep_recall,
                "hidden_failure": bool(hid_count > 0),
                "hidden_failure_count": hid_count,
                "hidden_seed_fraction": hidden_seed_frac,
                "seed_consistency": seed_consistency,
                "class_conditional_seed_consistency": cc_seed_consistency,
                "indicator_gain_auprc": mig_auprc,
            }
        )

    ev = pd.DataFrame(rows)
    return ev


def evaluate_go(
    evidence: pd.DataFrame,
    vuln: pd.DataFrame,
    slopes: pd.DataFrame,
    raw: pd.DataFrame,
    cfg: Any,
) -> dict[str, Any]:
    gate = cfg.gate
    strong_mvg = float(gate["strong_mvg"])
    strong_recall_drop = float(gate["strong_recall_drop"])
    stable_auroc_drop = float(gate["stable_auroc_drop"])
    slope_ratio = float(gate["slope_ratio"])
    min_seed = int(gate["min_seed_consistency"])
    min_datasets = int(gate["min_datasets"])

    # --- GO-A: minority vulnerability gap ---
    dataset_mvg = evidence.groupby("dataset")["strongest_MVG"].mean()
    go_a_datasets = [d for d, v in dataset_mvg.items() if v >= strong_mvg]
    # per (dataset, model) direction consistency across seeds.
    consistent = evidence[
        evidence["seed_consistency"] >= (min_seed / 3.0)
    ]
    # A dataset passes A if mean MVG >= threshold AND at least one model is
    # seed-consistent.
    go_a_pass = []
    for d in go_a_datasets:
        sub = evidence[evidence["dataset"] == d]
        if (sub["seed_consistency"] >= (min_seed / 3.0)).any():
            go_a_pass.append(d)
    GO_A = len(go_a_pass) >= min_datasets

    # --- GO-B: class-conditional causes strong minority harm ---
    cc_seed_ok = evidence[
        evidence["class_conditional_seed_consistency"] >= (min_seed / 3.0)
    ]
    # require CCEP_recall >= strong_recall_drop as well
    go_b_cases = cc_seed_ok[cc_seed_ok["CCEP_recall"] >= strong_recall_drop]
    go_b_cases = go_b_cases[go_b_cases["CCEP_recall"].notna()]
    GO_B = (
        go_b_cases.groupby("dataset").filter(
            lambda g: len(g) > 0
        )["dataset"].nunique() >= min_datasets
    )

    # --- GO-C: hidden minority failure ---
    hidden_seed_ok = evidence[
        evidence["hidden_seed_fraction"] >= (min_seed / 3.0)
    ]
    GO_C = hidden_seed_ok["dataset"].nunique() >= min_datasets

    # --- GO-D: minority slope steeper ---
    slopes_p = slopes.copy()
    slopes_p["ratio"] = slopes_p["minority_slope"].abs() / (
        slopes_p["majority_slope"].abs() + 1e-12
    )
    steep = slopes_p[
        (slopes_p["minority_slope"] < 0) & (slopes_p["ratio"] >= slope_ratio)
    ]
    GO_D = steep["dataset"].nunique() >= min_datasets

    verdicts = {
        "GO_A_minority_vulnerability_gap": GO_A,
        "GO_B_class_conditional_penalty": GO_B,
        "GO_C_hidden_minority_failure": GO_C,
        "GO_D_minority_slope": GO_D,
    }
    go_flags = [k for k, v in verdicts.items() if v]
    return {
        "verdicts": verdicts,
        "go_conditions_hit": go_flags,
        "go_a_dataset_mvg": dataset_mvg.to_dict(),
        "dataset_mvg": dataset_mvg.to_dict(),
        "consistent_count": int(consistent.shape[0]),
    }


def decide(
    evidence: pd.DataFrame,
    vuln: pd.DataFrame,
    slopes: pd.DataFrame,
    raw: pd.DataFrame,
    cfg: Any,
) -> tuple[dict[str, Any], str]:
    """Return (decision dict, gate string)."""
    go = evaluate_go(evidence, vuln, slopes, raw, cfg)
    gate = cfg.gate
    strong_mvg = float(gate["strong_mvg"])
    moderate_mvg = float(gate["moderate_mvg"])

    if go["go_conditions_hit"]:
        return go, "GO"

    # HOLD signals.
    signals: list[str] = []
    dataset_mvg = evidence.groupby("dataset")["strongest_MVG"].mean().to_dict()
    max_mvg = max(dataset_mvg.values()) if dataset_mvg else 0.0
    if moderate_mvg <= max_mvg < strong_mvg:
        signals.append("mean MVG is moderate (0.05-0.10)")

    if len(dataset_mvg) and sum(v >= strong_mvg for v in dataset_mvg.values()) == 1:
        signals.append("only one dataset shows a strong gap")

    if evidence.shape[0] and (
        evidence["dataset"].nunique() < 2 or evidence.query("strongest_MVG >= 0").shape[0] <= 1
    ):
        signals.append("the effect is not spread across datasets/models")

    slope_input = slopes if not slopes.empty else pd.DataFrame()
    if not slope_input.empty:
        mean_ratio = (
            slope_input["minority_slope"].abs()
            / (slope_input["majority_slope"].abs() + 1e-12)
        ).mean()
        if mean_ratio < 1.0:
            signals.append("minority slope is not clearly steeper")

    verdict = "HOLD" if signals else "STOP/PIVOT"
    go["hold_signals"] = signals
    go["max_dataset_mvg"] = max_mvg
    return go, verdict
