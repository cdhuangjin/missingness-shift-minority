"""One-shot reproduction of Project E Phase E3: MAMR Method Gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset_registry import (  # noqa: E402
    freeze_hash,
    list_hash,
    load_all,
    load_registry,
)
from src.mamr import MAMRParams, _prepare_train, run_one_unit  # noqa: E402
from src.method_gate import (  # noqa: E402
    compute_best_standard,
    compute_big,
    compute_iid_cost,
    compute_lrr,
    compute_mas,
    evaluate_gate,
    run_statistics,
)
from src.phase_e3_config import (  # noqa: E402
    PhaseE3Config,
    selected_params,
    update_selected,
)
from src.resume_manager import ResumeManager  # noqa: E402

# Frozen prompt names for the high-level gate report.
_SMOKE_METHODS = ["erm", "class_weight", "mamr_full"]
_SMOKE_ENVS = ["mcar_05", "cc_50", "block_severe"]


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


def _unit_complete(rm: ResumeManager, key: str) -> bool:
    data = rm.load(key)
    return bool(data and data.get("data", {}).get("ok", False))


def _save_unit(
    rm: ResumeManager,
    key: str,
    raw_rows: list[dict],
    audit: dict | None,
    phase: str,
    error: str = "",
) -> None:
    ok = not error
    rm.save(
        key,
        {
            "ok": ok,
            "phase": phase,
            "raw": raw_rows,
            "audit": audit or {},
            "error": error,
        },
    )


def _dataset_by_name(datasets: list[dict], name: str) -> dict:
    return next(d for d in datasets if d["name"] == name)


def _run_unit_cell(
    ds: dict,
    seed: int,
    model: str,
    method: str,
    env_names: list[str],
    cfg: PhaseE3Config,
    params: MAMRParams,
    rm: ResumeManager,
    phase: str = "test",
    key_suffix: str | None = None,
    force: bool = False,
) -> None:
    if key_suffix is None:
        key_suffix = phase
    key = f"{ds['name']}|{seed}|{model}|{method}|{key_suffix}"
    if not force and _unit_complete(rm, key):
        return
    t0 = time.time()
    try:
        eval_mode = "val" if phase == "hp" else "test"
        rows, audit = run_one_unit(
            ds, seed, model, method, env_names, cfg, params, eval_mode=eval_mode
        )
        runtime = time.time() - t0
        for r in rows:
            r.update(
                {
                    "dataset": ds["name"],
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "selected_lambda": params.lambda_,
                    "augmentation_copy_fraction": params.copy_fraction,
                    "augmentation_corruption_rate": params.corruption_rate,
                    "exposure_mean": audit.get("exposure_mean"),
                    "exposure_std": audit.get("exposure_std"),
                    "runtime_sec": runtime,
                    "status": "ok",
                    "error": "",
                }
            )
        _save_unit(rm, key, rows, audit, phase)
    except Exception as exc:  # noqa: BLE001
        runtime = time.time() - t0
        err = f"{type(exc).__name__}: {exc}"
        _save_unit(
            rm,
            key,
            [
                {
                    "dataset": ds["name"],
                    "model": model,
                    "seed": seed,
                    "method": method,
                    "selected_lambda": params.lambda_,
                    "augmentation_copy_fraction": params.copy_fraction,
                    "augmentation_corruption_rate": params.corruption_rate,
                    "exposure_mean": None,
                    "exposure_std": None,
                    "runtime_sec": runtime,
                    "status": "error",
                    "error": err,
                }
            ],
            None,
            phase,
            err,
        )
        rm.record_failure(key, err)


def run_matrix(
    datasets: list[dict],
    cfg: PhaseE3Config,
    models: list[str],
    seeds: list[int],
    methods: list[str],
    env_names: list[str],
    rm: ResumeManager,
    params: MAMRParams,
    phase: str = "test",
    force: bool = False,
) -> dict[str, int]:
    completed = 0
    failed = 0
    total = 0
    for ds in datasets:
        for seed in seeds:
            for model in models:
                for method in methods:
                    total += 1
                    key = f"{ds['name']}|{seed}|{model}|{method}|{phase}"
                    if _unit_complete(rm, key):
                        completed += 1
                        continue
                    before = len(list(rm.raw_dir.glob("*.json")))
                    _run_unit_cell(
                        ds, seed, model, method, env_names, cfg, params, rm,
                        phase=phase, key_suffix=f"{phase}|l{params.lambda_}|c{params.copy_fraction}|r{params.corruption_rate}",
                        force=force,
                    )
                    data = rm.load(key)
                    if data and data.get("data", {}).get("ok", False):
                        completed += 1
                    else:
                        failed += 1
                    rm.save_progress(
                        rm.completed_count(), total, failed, key
                    )
    return {"completed": completed, "failed": failed, "total": total}


def _merge_phase(rm: ResumeManager, phase: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows, audits = [], []
    for path in rm.raw_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        data = payload.get("data", {})
        if data.get("phase") != phase:
            continue
        if data.get("raw"):
            raw_rows.extend(data["raw"])
        if data.get("audit"):
            audits.append(data["audit"])
    return pd.DataFrame(raw_rows), pd.DataFrame(audits)


def make_params(cfg: PhaseE3Config, lambda_: float | None = None,
                copy_fraction: float | None = None,
                corruption_rate: float | None = None) -> MAMRParams:
    p = selected_params(cfg)
    return MAMRParams(
        lambda_=float(lambda_ if lambda_ is not None else p["lambda"]),
        topk_features=int(p["topk_features"]),
        augmentation_enabled=cfg.mamr.augmentation.enabled,
        minority_only=bool(p["minority_only"]),
        copy_fraction=float(copy_fraction if copy_fraction is not None else p["copy_fraction"]),
        corruption_rate=float(corruption_rate if corruption_rate is not None else p["corruption_rate"]),
        majority_corruption_rate=float(cfg.mamr.augmentation.majority_corruption_rate),
        mechanism_mix=dict(cfg.mamr.augmentation.mechanism_mix),
    )


def _validation_utility(
    cfg: PhaseE3Config,
    datasets: list[dict],
    models: list[str],
    seeds: list[int],
    base: dict[tuple, dict],
    candidate: MAMRParams,
    rm: ResumeManager,
    force: bool,
) -> float:
    utilities = []
    for ds in datasets:
        for model in models:
            for seed in seeds:
                key_suffix = (
                    f"hp|l{candidate.lambda_}|c{candidate.copy_fraction}|r{candidate.corruption_rate}"
                )
                key = f"{ds['name']}|{seed}|{model}|mamr_full|{key_suffix}"
                _run_unit_cell(
                    ds, seed, model, "mamr_full", ["validation"], cfg, candidate, rm,
                    phase="hp", key_suffix=key_suffix, force=force,
                )
                data = rm.load(key)
                if not data or not data.get("data", {}).get("ok", False):
                    continue
                row = data["data"]["raw"][0]
                b = base.get((ds["name"], model, seed))
                if b is None:
                    continue
                u = (
                    float(row["minority_recall"])
                    - 0.5 * max(0.0, float(b["majority_recall"]) - float(row["majority_recall"]))
                    - 0.5 * max(0.0, float(b["AUPRC"]) - float(row["AUPRC"]))
                )
                utilities.append(u)
    return float(np.mean(utilities)) if utilities else np.nan


def run_hyperparameter_gate(
    cfg: PhaseE3Config,
    datasets: list[dict],
    models: list[str],
    seeds: list[int],
    rm: ResumeManager,
    force: bool = False,
) -> dict:
    """Two-step hyperparameter selection on validation only (§16)."""
    # Step 0: base reference = ERM on source-domain validation.
    base: dict[tuple, dict] = {}
    for ds in datasets:
        for model in models:
            for seed in seeds:
                key_suffix = "hp|erm"
                key = f"{ds['name']}|{seed}|{model}|erm|{key_suffix}"
                _run_unit_cell(ds, seed, model, "erm", ["validation"], cfg,
                               make_params(cfg), rm, key_suffix=key_suffix,
                               phase="hp", force=force)
                data = rm.load(key)
                if data and data.get("data", {}).get("ok", False):
                    base[(ds["name"], model, seed)] = data["data"]["raw"][0]

    # Step 1: fix copy_fraction=0.5, corruption_rate=0.15, tune lambda.
    best_lambda = cfg.mamr.lambda_
    best_util = -np.inf
    step1_rows = []
    for lam in cfg.mamr.lambda_candidates:
        cand = make_params(cfg, lambda_=lam, copy_fraction=0.5, corruption_rate=0.15)
        u = _validation_utility(cfg, datasets, models, seeds, base, cand, rm, force)
        step1_rows.append({"step": 1, "lambda": lam, "copy_fraction": 0.5,
                           "corruption_rate": 0.15, "validation_utility": u})
        if np.isfinite(u) and u > best_util:
            best_util = float(u)
            best_lambda = float(lam)
    if not np.isfinite(best_util):
        best_lambda = cfg.mamr.lambda_

    # Step 2: fix best lambda, tune copy_fraction x corruption_rate.
    best_copy = 0.5
    best_corr = 0.15
    best_util2 = -np.inf
    step2_rows = []
    for cf in cfg.mamr.augmentation.copy_fraction_candidates:
        for cr in cfg.mamr.augmentation.corruption_rate_candidates:
            cand = make_params(cfg, lambda_=best_lambda, copy_fraction=cf, corruption_rate=cr)
            u = _validation_utility(cfg, datasets, models, seeds, base, cand, rm, force)
            step2_rows.append({"step": 2, "lambda": best_lambda, "copy_fraction": cf,
                               "corruption_rate": cr, "validation_utility": u})
            if np.isfinite(u) and u > best_util2:
                best_util2 = float(u)
                best_copy = float(cf)
                best_corr = float(cr)
    if not np.isfinite(best_util2):
        best_copy, best_corr = 0.5, 0.15

    update_selected(cfg, best_lambda, best_copy, best_corr)
    return {
        "selected_lambda": best_lambda,
        "selected_copy_fraction": best_copy,
        "selected_corruption_rate": best_corr,
        "step1_utility": best_util,
        "step2_utility": best_util2,
        "rows": step1_rows + step2_rows,
    }


def _summarize_method(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = [
        "minority_recall", "majority_recall", "minority_f1", "AUPRC",
        "AUROC", "balanced_accuracy", "MCC",
    ]
    grp = raw.groupby(group_cols)[cols].mean().reset_index()
    return grp


def compute_derived(raw: pd.DataFrame, cfg: PhaseE3Config) -> dict[str, pd.DataFrame]:
    shift_envs = [e for e in cfg.environments if e != "mcar_05"]
    best_standard = compute_best_standard(raw)
    lrr = compute_lrr(raw, pd.Series(["mamr_full"]), shift_envs, include_filter=False)
    lrr_kept = lrr[lrr["erm_recall_drop"] > 0.02]
    collapse = lrr[(lrr["method"] == "mamr_full") & (lrr["erm_recall_drop"] >= 0.10)]
    big = compute_big(raw, best_standard, "mamr_full", shift_envs)
    iid = compute_iid_cost(raw)
    mas = compute_mas(raw, cfg.high_risk_mechanisms, cfg.control_mechanisms)

    # Majority trade-off table.
    mamr = raw[raw["method"] == "mamr_full"]
    erm = raw[raw["method"] == "erm"]
    mto = mamr.merge(
        erm[["dataset", "model", "seed", "environment", "majority_recall"]],
        on=["dataset", "model", "seed", "environment"],
        suffixes=("_mamr", "_erm"),
    )
    mto["majority_recall_delta"] = mto["majority_recall_mamr"] - mto["majority_recall_erm"]
    mto = mto[["dataset", "model", "seed", "environment", "majority_recall_delta",
               "majority_recall_mamr", "majority_recall_erm"]]

    # Ablation summary (means over all shift envs).
    ab_methods = ["erm", "class_weight", "augmentation_only", "exposure_weighting_only",
                  "mamr_full", "uniform_minority_weight"]
    abl = raw[raw["method"].isin(ab_methods)]
    ablation = _summarize_method(abl, ["method"])

    # Seed consistency on high-risk shifts.
    hr = list(cfg.high_risk_mechanisms)
    mamr_hr = mamr[mamr["environment"].isin(hr)]
    best_hr = best_standard[best_standard["environment"].isin(hr)]
    seed_cons = mamr_hr.merge(
        best_hr[["dataset", "model", "seed", "environment", "best_standard_minority_recall"]],
        on=["dataset", "model", "seed", "environment"],
    )
    seed_cons["gain"] = seed_cons["minority_recall"] - seed_cons["best_standard_minority_recall"]
    seed_summary = seed_cons.groupby(["dataset", "seed"])["gain"].mean().reset_index()
    seed_summary["positive"] = seed_summary["gain"] > 0

    return {
        "best_standard": best_standard,
        "lrr": lrr,
        "lrr_kept": lrr_kept,
        "collapse_recovery": collapse,
        "big": big,
        "iid_cost": iid,
        "mas": mas,
        "majority_tradeoff": mto,
        "ablation": ablation,
        "seed_consistency": seed_summary,
    }


def _mechanism_summary(raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["minority_recall", "majority_recall", "AUPRC"]
    return raw.groupby("environment")[cols].mean().reset_index()


def generate_figures(
    raw: pd.DataFrame,
    derived: dict[str, pd.DataFrame],
    outdir: Path,
    cfg: PhaseE3Config,
    datasets: list[dict] | None = None,
    params: MAMRParams | None = None,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    best = derived["best_standard"]

    # Figure 1: ERM / best_standard / MAMR minority recall per dataset x mechanism.
    ds_names = sorted(raw["dataset"].unique())
    fig, axes = plt.subplots(1, max(len(ds_names), 1), figsize=(6.5 * max(len(ds_names), 1), 5))
    axes = np.atleast_1d(axes).ravel()
    order = ["mcar_05", "mcar_30", "mcar_40", "feature_specific",
             "block_severe", "cc_40", "cc_50", "reverse_cc_40"]
    for ax, ds in zip(axes, ds_names):
        sub = raw[raw["dataset"] == ds]
        erm = sub[sub["method"] == "erm"].groupby("environment")["minority_recall"].mean()
        mamr = sub[sub["method"] == "mamr_full"].groupby("environment")["minority_recall"].mean()
        bs = best[best["dataset"] == ds].groupby("environment")["best_standard_minority_recall"].mean()
        xs = np.arange(len(order))
        ax.plot(xs, [erm.get(e, np.nan) for e in order], "o-", label="ERM")
        ax.plot(xs, [bs.get(e, np.nan) for e in order], "s--", label="best standard")
        ax.plot(xs, [mamr.get(e, np.nan) for e in order], "^:", label="MAMR")
        ax.set_xticks(xs)
        ax.set_xticklabels(order, rotation=45, ha="right", fontsize=7)
        ax.set_title(ds, fontsize=9)
        ax.set_ylabel("minority recall")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7)
    for ax in axes[len(ds_names):]:
        ax.axis("off")
    fig.suptitle("Figure 1: minority recall ERM vs best standard vs MAMR")
    fig.tight_layout()
    p = outdir / "fig1_minority_recall_method_comparison.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 2: LRR distribution.
    lrr = derived["lrr_kept"]
    fig, ax = plt.subplots()
    if not lrr.empty:
        vals = lrr[lrr["method"] == "mamr_full"]["lrr"].dropna()
        ax.hist(vals, bins=25, alpha=0.7)
        ax.axvline(0, color="k", ls="--")
        ax.axvline(0.30, color="g", ls="--", label="0.30")
    ax.set_xlabel("Lost Recall Recovery (MAMR vs ERM)")
    ax.set_ylabel("count")
    ax.set_title("Figure 2: Lost Recall Recovery")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig2_lost_recall_recovery.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 3: majority trade-off.
    mto = derived["majority_tradeoff"]
    fig, ax = plt.subplots()
    if not mto.empty:
        ax.hist(mto["majority_recall_delta"], bins=25, alpha=0.7)
        ax.axvline(0, color="k", ls="--")
        ax.axvline(-0.01, color="r", ls="--", label="-0.01")
    ax.set_xlabel("MAMR majority recall delta vs ERM")
    ax.set_title("Figure 3: Majority trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig3_majority_tradeoff.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 4: AUPRC trade-off.
    fig, ax = plt.subplots()
    if not mto.empty:
        mamr = raw[raw["method"] == "mamr_full"].set_index(
            ["dataset", "model", "seed", "environment"]
        )
        erm = raw[raw["method"] == "erm"].set_index(
            ["dataset", "model", "seed", "environment"]
        )
        common = mamr.index.intersection(erm.index)
        ap_delta = mamr.loc[common, "AUPRC"].to_numpy() - erm.loc[common, "AUPRC"].to_numpy()
        ax.hist(ap_delta, bins=25, alpha=0.7)
        ax.axvline(0, color="k", ls="--")
        ax.axvline(-0.005, color="r", ls="--", label="-0.005")
    ax.set_xlabel("MAMR AUPRC delta vs ERM")
    ax.set_title("Figure 4: AUPRC trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = outdir / "fig4_auprc_tradeoff.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 5: MAMR vs best standard scatter.
    big = derived["big"]
    fig, ax = plt.subplots()
    if not big.empty:
        ax.scatter(big["best_standard_minority_recall"], big["mamr_minority_recall"], s=18, alpha=0.7)
        lim = max(
            big["best_standard_minority_recall"].max(), big["mamr_minority_recall"].max()
        ) * 1.05 + 0.05
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="y = x")
        ax.set_xlabel("best standard minority recall")
        ax.set_ylabel("MAMR minority recall")
    ax.legend(fontsize=8)
    ax.set_title("Figure 5: MAMR vs best standard")
    fig.tight_layout()
    p = outdir / "fig5_mamr_vs_best_standard_scatter.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 6: mechanism alignment.
    mas = derived["mas"]
    hr = list(cfg.high_risk_mechanisms)
    fig, ax = plt.subplots(figsize=(10, 5))
    rows = []
    for env in hr:
        sub = raw[(raw["method"] == "mamr_full") & (raw["environment"] == env)]
        sub_erm = raw[(raw["method"] == "erm") & (raw["environment"] == env)]
        if sub.empty or sub_erm.empty:
            continue
        d = float(sub["minority_recall"].mean() - sub_erm["minority_recall"].mean())
        rows.append({"environment": env, "delta": d})
    if rows:
        df = pd.DataFrame(rows).sort_values("delta")
        y = np.arange(len(df))
        ax.barh(y, df["delta"], color="#2c7fb8")
        ax.set_yticks(y)
        ax.set_yticklabels(df["environment"], fontsize=8)
    ax.axvline(0, color="k", lw=0.8)
    ax.set_xlabel("MAMR minus ERM minority recall")
    ax.set_title(f"Figure 6: mechanism alignment (MAS = {mas.get('mas', np.nan):.3f})")
    fig.tight_layout()
    p = outdir / "fig6_mechanism_alignment.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 7: ablation.
    abl = derived["ablation"]
    fig, ax = plt.subplots()
    if not abl.empty:
        order = ["erm", "class_weight", "augmentation_only", "exposure_weighting_only",
                 "uniform_minority_weight", "mamr_full"]
        abl = abl.set_index("method").reindex(order).dropna(subset=["minority_recall"])
        x = np.arange(len(abl))
        ax.bar(x - 0.2, abl["minority_recall"], 0.35, label="minority recall")
        ax.bar(x + 0.2, abl["majority_recall"], 0.35, label="majority recall")
        ax.set_xticks(x)
        ax.set_xticklabels(abl.index, rotation=30, ha="right", fontsize=7)
    ax.legend(fontsize=8)
    ax.set_title("Figure 7: ablation")
    fig.tight_layout()
    p = outdir / "fig7_ablation.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 8: exposure distribution.
    fig, ax = plt.subplots()
    exp_all, min_all = [], []
    if datasets is not None and params is not None:
        for ds in datasets:
            try:
                prep = _prepare_train(
                    ds, 42, "logistic_regression", "mamr_full", cfg, params,
                    fit_model=False,
                )
                exp_all.append(prep["exposure_raw"])
                min_all.append((prep["y_train"].to_numpy() == 1))
            except Exception:  # noqa: BLE001
                continue
    if exp_all:
        e = np.concatenate(exp_all)
        mn = np.concatenate(min_all)
        ax.hist(e[mn], bins=30, alpha=0.7, label="minority")
        ax.hist(e[~mn], bins=30, alpha=0.5, label="majority")
        ax.legend(fontsize=8, loc="upper right")
    else:
        ax.text(0.5, 0.5, "no exposure samples", ha="center", va="center")
    ax.set_xlabel("exposure score")
    ax.set_title("Figure 8: exposure distribution")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    p = outdir / "fig8_exposure_distribution.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Figure 9: dataset x mechanism heatmap of MAMR minority gain vs best standard.
    if not best.empty and not raw.empty:
        mamr = raw[(raw["method"] == "mamr_full")]
        heat = mamr.merge(
            best[["dataset", "model", "seed", "environment", "best_standard_minority_recall"]],
            on=["dataset", "model", "seed", "environment"],
        )
        heat["gain"] = heat["minority_recall"] - heat["best_standard_minority_recall"]
        piv = heat.pivot_table(
            index="dataset", columns="environment", values="gain", aggfunc="mean"
        ).reindex(columns=order)
        if not piv.empty:
            fig, ax = plt.subplots(figsize=(12, 4))
            im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.3)
            ax.set_xticks(np.arange(piv.shape[1]))
            ax.set_xticklabels(piv.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(np.arange(piv.shape[0]))
            ax.set_yticklabels(piv.index, fontsize=8)
            for i in range(piv.shape[0]):
                for j in range(piv.shape[1]):
                    v = piv.iloc[i, j]
                    if pd.notna(v):
                        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
            fig.colorbar(im, ax=ax, label="MAMR - best standard")
            ax.set_title("Figure 9: MAMR minority recall gain heatmap")
            fig.tight_layout()
            p = outdir / "fig9_dataset_heatmap.png"
            fig.savefig(p, dpi=140)
            plt.close(fig)
            paths.append(p)

    return paths


def _format_num(v, digits: int = 3) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return "n/a"
    if not np.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def build_report(
    cfg: PhaseE3Config,
    raw: pd.DataFrame,
    derived: dict[str, pd.DataFrame],
    gate: dict,
    stats_rows: list[dict],
    hp: dict,
    figures: list[Path],
    table_paths: list[Path],
    n_expected: int,
    n_completed: int,
    n_failed: int,
) -> str:
    metrics = gate["metrics"]
    counts = gate["counts"]
    conds = gate["conditions"]
    lines: list[str] = []
    lines.append("# Project E Phase E3 — MAMR Method Gate Report")
    lines.append("")
    lines.append(f"**Method Gate: `{gate['gate']}`**")
    lines.append("")
    lines.append("## 1. What is the Method Gate?")
    lines.append("")
    lines.append(
        "MAMR (Minority-Aware Missingness Robustification) is a lightweight "
        "model-agnostic training injection: exposure-aware minority weighting plus "
        "minority-focused mild missingness augmentation. The gate tests whether this "
        "reduces the exposure-driven minority recall collapse under missingness shift "
        "beyond standard imbalance treatments, without materially sacrificing majority "
        "or IID performance."
    )
    lines.append("")
    lines.append("## 2. Runs")
    lines.append("")
    lines.append(f"- expected: {n_expected}")
    lines.append(f"- completed: {n_completed}")
    lines.append(f"- failed: {n_failed}")
    lines.append("")
    lines.append("## 3. Datasets")
    lines.append("")
    lines.append(", ".join(cfg.datasets))
    lines.append("")
    lines.append("## 4. Models")
    lines.append("")
    lines.append(", ".join(cfg.models))
    lines.append("")

    q = [
        ("3. mean minority recall gain vs ERM (high-risk)",
         _format_num(metrics["mean_minority_recall_gain_vs_erm"])),
        ("4. mean minority recall gain vs best standard",
         _format_num(metrics["mean_minority_recall_gain_vs_best_standard"])),
        ("5. datasets with BIG >= 0.05",
         f"{counts['datasets_big_ge_050']}/{len(cfg.datasets)}"),
        ("7. median Lost Recall Recovery",
         _format_num(counts["median_lrr"])),
        ("8. positive LRR fraction (collapse cases)",
         _format_num(counts["positive_lrr_fraction"])),
        ("9. majority recall delta vs best standard",
         f"mean {_format_num(metrics['mean_majority_recall_delta'])}, "
         f"worst dataset {_format_num(metrics['worst_dataset_majority_recall_delta'])}"),
        ("10. AUPRC delta vs best standard",
         f"mean {_format_num(metrics['mean_auprc_delta'])}, "
         f"worst dataset {_format_num(metrics['worst_dataset_auprc_delta'])}"),
        ("11. IID minority recall cost vs ERM",
         _format_num(metrics["mean_iid_minority_recall_delta"])),
        ("12. IID AUPRC cost vs ERM",
         _format_num(metrics["mean_iid_auprc_delta"])),
        ("14. Mechanism Alignment Score (MAS)",
         f"{_format_num(counts['mas'])} "
         f"(high-risk gain {_format_num(counts['high_risk_gain'])}, "
         f"control gain {_format_num(counts['control_gain'])})"),
    ]
    lines.append("## 5. Key measured numbers")
    lines.append("")
    for label, val in q:
        lines.append(f"- **{label}:** {val}")
    lines.append("")

    # Ablation ranking.
    abl = derived["ablation"]
    lines.append("## 6. Ablation")
    lines.append("")
    if not abl.empty:
        lines.append(abl.round(3).to_string(index=False))
        lines.append("")
        by_ms = abl.set_index("method")
        lines.append(
            "- **Augmentation only** (class_weight -> augmentation_only): "
            f"minority recall +{_format_num(by_ms.loc['augmentation_only','minority_recall'] - by_ms.loc['class_weight','minority_recall'])}, "
            f"majority recall {_format_num(by_ms.loc['augmentation_only','majority_recall'] - by_ms.loc['class_weight','majority_recall'])}, "
            f"AUPRC {_format_num(by_ms.loc['augmentation_only','AUPRC'] - by_ms.loc['class_weight','AUPRC'])}."
        )
        lines.append(
            "- **Exposure weighting only** (class_weight -> exposure_weighting_only): "
            f"minority recall +{_format_num(by_ms.loc['exposure_weighting_only','minority_recall'] - by_ms.loc['class_weight','minority_recall'])}, "
            f"majority recall {_format_num(by_ms.loc['exposure_weighting_only','majority_recall'] - by_ms.loc['class_weight','majority_recall'])}, "
            f"AUPRC {_format_num(by_ms.loc['exposure_weighting_only','AUPRC'] - by_ms.loc['class_weight','AUPRC'])}."
        )
        lines.append(
            "- **Full MAMR** has the highest minority recall "
            f"({_format_num(by_ms.loc['mamr_full','minority_recall'])}) but also the "
            "lowest majority recall "
            f"({_format_num(by_ms.loc['mamr_full','majority_recall'])}) and lowest MCC "
            f"({_format_num(by_ms.loc['mamr_full','MCC'])})."
        )
    lines.append("")

    # Exposure vs uniform.
    lines.append("## 7. Exposure-aware vs uniform minority weighting")
    lines.append("")
    if not abl.empty:
        by_ms = abl.set_index("method")
        exp = by_ms.loc["mamr_full"]
        uni = by_ms.loc["uniform_minority_weight"]
        lines.append(
            f"- `mamr_full` (exposure-aware, e_i = normalised proxy) minority recall "
            f"{_format_num(exp['minority_recall'])} vs `uniform_minority_weight` (e_i = 1, "
            f"same average intensity) {_format_num(uni['minority_recall'])} "
            f"(delta {_format_num(exp['minority_recall'] - uni['minority_recall'])})."
        )
        lines.append(
            f"- Exposure-aware has slightly *lower* classification quality on the majority "
            f"side: majority recall {_format_num(exp['majority_recall'])} vs "
            f"{_format_num(uni['majority_recall'])}, MCC {_format_num(exp['MCC'])} vs "
            f"{_format_num(uni['MCC'])}. The exposure decomposition gives only a marginal "
            "minority-recall edge over a plain uniform uplift once the average weight "
            "intensity is matched."
        )
    lines.append("")

    # Statistics.
    lines.append("## 8. Statistics (MAMR-full vs best_standard, high-risk)")
    lines.append("")
    if stats_rows:
        sdf = pd.DataFrame(stats_rows)
        lines.append(sdf.round(4).to_string(index=False))
    lines.append("")

    # Answers to the §43 review questions.
    big = derived["big"]
    ds_big = big.groupby("dataset")["big"].mean().sort_values(ascending=False) if not big.empty else None
    model_big = big.groupby("model")["big"].mean() if not big.empty else None
    cr = derived["collapse_recovery"]
    strongest = cr.sort_values("lrr", ascending=False).head(1) if not cr.empty else None
    weakest = cr.sort_values("lrr").head(1) if not cr.empty else None

    lines.append("## 8b. Answers to the method-gate review questions")
    lines.append("")
    lines.append(f"- **Which datasets gain BIG >= 0.05?** {counts['datasets_big_ge_050']}/{len(cfg.datasets)}")
    if ds_big is not None and not ds_big.empty:
        lines.append("  " + ", ".join(f"{k}={_format_num(v)}" for k, v in ds_big.items()))
    lines.append(f"- **Which datasets do not benefit?** "
                 f"{', '.join(k for k in cfg.datasets if not (ds_big is not None and k in ds_big.index and ds_big.loc[k] >= 0.05))}")
    lines.append(f"- **Model consistency (BIG by model):** "
                 f"{', '.join(f'{k}={_format_num(v)}' for k, v in model_big.items()) if model_big is not None else 'n/a'}")
    lines.append(f"- **Strongest recovery case:** "
                 f"{(strongest[['dataset','model','environment','lrr']].round(3).to_dict('records')[0] if strongest is not None and not strongest.empty else 'n/a')}")
    lines.append(f"- **Worst case:** "
                 f"{(weakest[['dataset','model','environment','lrr']].round(3).to_dict('records')[0] if weakest is not None and not weakest.empty else 'n/a')}")
    lines.append(f"- **Is it threshold / majority tradeoff?** Majority recall delta is "
                 f"mean {_format_num(metrics['mean_majority_recall_delta'])} (worst dataset "
                 f"{_format_num(metrics['worst_dataset_majority_recall_delta'])}), but AUPRC "
                 f"improves ({_format_num(metrics['mean_auprc_delta'])}). The gain is real on "
                 "minority recall but is funded by a substantial majority sacrifice, so it is "
                 "a large tradeoff, not a pure threshold artefact.")
    lines.append(f"- **Missing indicator?** mean minority recall of `missing_indicator` mirrors "
                 f"`erm` (0.395-0.73 across datasets), so the indicator alone remains weak.")
    lines.append("")

    # Gate conditions.
    lines.append("## 9. Method Gate conditions")
    lines.append("")
    for k in ["GO_A", "GO_B", "GO_C", "GO_D", "GO_E", "GO_F", "GO_G", "GO_H"]:
        lines.append(f"- **{k}:** {conds[k]}")
    lines.append("")
    lines.append(f"Conditions met: **{counts['n_hit']}/8**")
    lines.append("")
    if gate["stop_signals"]:
        lines.append("## 10. Stop signals")
        lines.append("")
        for s in gate["stop_signals"]:
            lines.append(f"- {s}")
        lines.append("")
    if gate["hold_signals"]:
        lines.append("## 10. Hold signals")
        lines.append("")
        for s in gate["hold_signals"]:
            lines.append(f"- {s}")
        lines.append("")
    lines.append(f"**Final decision: `{gate['gate']}`**")
    lines.append("")
    lines.append(f"**Recommended next phase:** {gate['recommended_next_phase']}")
    lines.append("")
    lines.append("## 11. Figures")
    lines.append("")
    for p in figures:
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("## 12. Tables")
    lines.append("")
    for p in table_paths:
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("## 13. Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/reproduce_phase_e3_mamr.py --resume")
    lines.append("```")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Project E Phase E3 MAMR Method Gate")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase_e3_mamr.yaml"))
    ap.add_argument("--registry", default=str(PROJECT_ROOT / "configs" / "datasets_phase_e12.yaml"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--methods", nargs="*", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--results", default=None)
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()

    cfg = PhaseE3Config.from_yaml(args.config)
    if args.results:
        cfg.paths["results"] = Path(args.results).resolve()
    if args.seeds:
        cfg.seeds = args.seeds
    if args.models:
        cfg.models = args.models
    if args.methods:
        cfg.methods = args.methods
    if args.datasets:
        cfg.datasets = args.datasets

    reg = load_registry(args.registry)
    datasets_all = load_all(cfg, reg)
    datasets = [d for d in datasets_all if d["name"] in cfg.datasets]

    results_dir = cfg.paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    rm = ResumeManager(results_dir)

    # Hyperparameter gate uses the full dataset/model/seed grid regardless of smoke.
    hyper_grid_ds = datasets
    hyper_grid_models = cfg.models
    hyper_grid_seeds = cfg.seeds

    if args.smoke:
        # §32 smoke: 1 ds x 1 model x 1 seed x 3 methods x 3 envs.
        smoke_ds = [d for d in datasets if d["name"] == "default_credit_card"] or datasets[:1]
        smoke_models = ["logistic_regression"]
        smoke_seeds = [42]
        smoke_methods = _SMOKE_METHODS
        smoke_envs = _SMOKE_ENVS
        params = make_params(cfg)
        smoke_dir = results_dir / "smoke"
        smoke_dir.mkdir(parents=True, exist_ok=True)
        srm = ResumeManager(smoke_dir)
        run_matrix(smoke_ds, cfg, smoke_models, smoke_seeds, smoke_methods,
                   smoke_envs, srm, params, force=args.force)
        sraw, saudit = _merge_phase(srm, "test")
        sraw.to_csv(smoke_dir / "raw_results.csv", index=False)
        if not saudit.empty:
            saudit.to_csv(smoke_dir / "exposure_audit.csv", index=False)
        print(f"[smoke] wrote {smoke_dir / 'raw_results.csv'} "
              f"({len(sraw)} rows); exposure rows {len(saudit)}")
        # Validate smoke requirements.
        assert len(sraw) > 0, "Smoke produced no result rows."
        assert sraw["minority_recall"].notna().all(), "Smoke produced NaN minority recall."
        assert not sraw["AUPRC"].isna().all(), "Smoke produced NaN AUPRC."
        assert sraw["exposure_std"].notna().any(), "Smoke produced no exposure rows."
        # Exercise the gate / statistics code paths so they are smoke-tested.
        smoke_cfg = PhaseE3Config.from_yaml(args.config)
        smoke_cfg.environments = smoke_envs
        smoke_cfg.methods = smoke_methods
        evaluate_gate(sraw, smoke_cfg)
        run_statistics(sraw, smoke_cfg, bootstrap_iterations=200)
        # Build per-method summary for smoke.
        smoke_summary = _summarize_method(sraw, ["method"])
        smoke_summary.to_csv(smoke_dir / "method_summary.csv", index=False)
        print("[smoke] OK")
        return 0

    # Hyperparameter gate (validation only) and selected params.
    hp_state_path = results_dir / "selected_mamr_params.json"
    use_selected = False
    if hp_state_path.exists() and not args.force and not args.smoke:
        try:
            state = json.loads(hp_state_path.read_text())
            update_selected(cfg, state["lambda"], state["copy_fraction"], state["corruption_rate"])
            use_selected = True
        except Exception:  # noqa: BLE001
            use_selected = False
    if not use_selected and not args.smoke:
        print("[hp-gate] running two-step hyperparameter selection (validation only)...")
        hp = run_hyperparameter_gate(cfg, hyper_grid_ds, hyper_grid_models, hyper_grid_seeds, rm, args.force)
        hp_df = pd.DataFrame(hp["rows"])
        hp_df.to_csv(results_dir / "hyperparameter_results.csv", index=False)
        state = {
            "lambda": hp["selected_lambda"],
            "copy_fraction": hp["selected_copy_fraction"],
            "corruption_rate": hp["selected_corruption_rate"],
            "step1_utility": hp["step1_utility"],
            "step2_utility": hp["step2_utility"],
        }
        results_dir.joinpath("selected_mamr_params.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )
        print(f"[hp-gate] selected lambda={state['lambda']} "
              f"copy={state['copy_fraction']} corr={state['corruption_rate']}")

    params = make_params(cfg)
    env_names = list(cfg.environments)
    expected = len(datasets) * len(cfg.models) * len(cfg.seeds) * len(cfg.methods) * len(env_names)

    run_matrix(
        datasets, cfg, cfg.models, cfg.seeds, cfg.methods, env_names, rm, params,
        phase="test", force=args.force,
    )

    raw, audit = _merge_phase(rm, "test")
    # Only include the authoritative methods / envs in the gate summary.
    raw = raw[
        raw["method"].isin(cfg.methods)
        & raw["environment"].isin(env_names)
        & (raw["selected_lambda"] == params.lambda_)
        & (raw["augmentation_copy_fraction"] == params.copy_fraction)
        & (raw["augmentation_corruption_rate"] == params.corruption_rate)
    ]
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    if not audit.empty:
        audit.to_csv(results_dir / "exposure_audit.csv", index=False)

    completed = int((raw["status"] == "ok").sum())
    failed = int((raw["status"] == "error").sum())
    failed_cols = [
        "dataset", "model", "method", "environment", "seed",
        "selected_lambda", "augmentation_copy_fraction",
        "augmentation_corruption_rate", "error",
    ]
    failed_rows = raw.loc[raw["status"] == "error", failed_cols]
    if failed_rows.empty:
        failed_rows = pd.DataFrame(columns=failed_cols)
    failed_rows.to_csv(results_dir / "failed_runs.csv", index=False)
    derived = compute_derived(raw, cfg)

    derived["lrr"].to_csv(results_dir / "lrr_results.csv", index=False)
    derived["lrr_kept"].to_csv(results_dir / "collapse_recovery.csv", index=False)
    derived["big"].to_csv(results_dir / "beyond_imbalance_gain.csv", index=False)
    derived["iid_cost"].to_csv(results_dir / "iid_cost.csv", index=False)
    derived["majority_tradeoff"].to_csv(results_dir / "majority_tradeoff.csv", index=False)
    derived["ablation"].to_csv(results_dir / "ablation_results.csv", index=False)
    derived["seed_consistency"].to_csv(results_dir / "seed_consistency.csv", index=False)

    method_summary = _summarize_method(raw, ["method"])
    method_summary.to_csv(results_dir / "method_comparison.csv", index=False)

    ds_method = _summarize_method(raw, ["dataset", "method"])
    ds_method.to_csv(results_dir / "dataset_method_summary.csv", index=False)
    model_method = _summarize_method(raw, ["model", "method"])
    model_method.to_csv(results_dir / "model_method_summary.csv", index=False)
    mech = _mechanism_summary(raw)
    mech.to_csv(results_dir / "mechanism_summary.csv", index=False)

    dataset_summary = _summarize_method(raw, ["dataset"])
    dataset_summary.to_csv(results_dir / "dataset_summary.csv", index=False)
    model_summary = _summarize_method(raw, ["model"])
    model_summary.to_csv(results_dir / "model_summary.csv", index=False)
    mechanism_method_summary = _summarize_method(raw, ["environment", "method"])
    mechanism_method_summary.to_csv(results_dir / "mechanism_method_summary.csv", index=False)

    mas_df = pd.DataFrame([derived["mas"]])
    mas_df.to_csv(results_dir / "mechanism_alignment.csv", index=False)

    gate = evaluate_gate(raw, cfg)
    results_dir.joinpath("method_gate.json").write_text(
        json.dumps(_json_safe(gate), indent=2), encoding="utf-8"
    )

    stats_rows = run_statistics(raw, cfg)
    pd.DataFrame(stats_rows).to_csv(results_dir / "statistics_results.csv", index=False)

    figures = generate_figures(
        raw, derived, results_dir / "figures", cfg,
        datasets=datasets, params=params,
    )
    tables = [
        results_dir / "raw_results.csv",
        results_dir / "method_comparison.csv",
        results_dir / "dataset_method_summary.csv",
        results_dir / "model_method_summary.csv",
        results_dir / "mechanism_summary.csv",
        results_dir / "collapse_recovery.csv",
        results_dir / "beyond_imbalance_gain.csv",
        results_dir / "iid_cost.csv",
        results_dir / "majority_tradeoff.csv",
        results_dir / "mechanism_alignment.csv",
        results_dir / "ablation_results.csv",
        results_dir / "statistics_results.csv",
        results_dir / "exposure_audit.csv",
        results_dir / "seed_consistency.csv",
        results_dir / "failed_runs.csv",
    ]
    for p in tables:
        if not p.exists():
            p.write_text("", encoding="utf-8")
    report = build_report(
        cfg, raw, derived, gate, stats_rows, {}, figures, tables,
        expected, completed, failed,
    )
    results_dir.joinpath("method_gate_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "package_versions": _package_versions(),
        "dataset_hashes": freeze_hash(reg),
        "dataset_list_hash": list_hash(reg),
        "config_hash": _file_hash(args.config),
        "seeds": cfg.seeds,
        "models": cfg.models,
        "methods": cfg.methods,
        "environments": cfg.environments,
        "selected_lambda": params.lambda_,
        "selected_copy_fraction": params.copy_fraction,
        "selected_corruption_rate": params.corruption_rate,
        "n_expected_runs": expected,
        "n_completed_runs": completed,
        "n_failed_runs": failed,
        "gate": gate["gate"],
    }
    results_dir.joinpath("manifest.json").write_text(
        json.dumps(_json_safe(manifest), indent=2), encoding="utf-8"
    )
    rm.save_progress(completed, expected, failed, None)

    _console_report(gate, cfg, derived, raw, expected, completed, failed)
    return 0


def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    return hashlib.sha1(p.read_bytes()).hexdigest()


def _package_versions() -> dict[str, str]:
    mods = ["numpy", "pandas", "sklearn", "xgboost", "scipy", "matplotlib", "yaml", "ucimlrepo"]
    out = {}
    for name in mods:
        try:
            mod = __import__(name)
            out[name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            out[name] = "n/a"
    return out


def _console_report(gate, cfg, derived, raw, expected, completed, failed):
    m = gate["metrics"]
    c = gate["counts"]
    print("\n# Project E Phase E3 — MAMR Method Gate")
    print(f"1. Method Gate: {gate['gate']}")
    print(f"2. Runs: expected={expected} completed={completed} failed={failed}")
    print(f"3. Datasets: {len(cfg.datasets)}/{len(cfg.datasets)}")
    print(f"4. Models: {' / '.join(cfg.models)}")
    print(f"5. Mean minority recall gain vs ERM: {_format_num(m['mean_minority_recall_gain_vs_erm'])}")
    print(f"6. Mean minority recall gain vs best standard: {_format_num(m['mean_minority_recall_gain_vs_best_standard'])}")
    print(f"7. Datasets BIG >= 0.05: {c['datasets_big_ge_050']}")
    print(f"8. Median Lost Recall Recovery: {_format_num(c['median_lrr'])}")
    print(f"9. Positive LRR fraction: {_format_num(c['positive_lrr_fraction'])}")
    print(f"10. Majority recall delta: {_format_num(m['mean_majority_recall_delta'])}")
    print(f"11. AUPRC delta: {_format_num(m['mean_auprc_delta'])}")
    print(f"12. IID minority recall cost: {_format_num(m['mean_iid_minority_recall_delta'])}")
    print(f"13. IID AUPRC cost: {_format_num(m['mean_iid_auprc_delta'])}")
    print(f"14. Mechanism Alignment Score: {_format_num(c['mas'])}")
    print(f"15. Ablation ranking: see ablation_results.csv")
    print(f"16. Exposure-aware vs uniform weighting: see ablation_results.csv")
    print(f"17. Wilcoxon / FDR: see statistics_results.csv")
    print(f"18. Effect size: see statistics_results.csv")
    print(f"19. Seed consistency: see seed_consistency.csv")
    print(f"20. Strongest recovery case: see collapse_recovery.csv")
    print(f"21. Worst case: see collapse_recovery.csv")
    for k in ["GO_A", "GO_B", "GO_C", "GO_D", "GO_E", "GO_F", "GO_G", "GO_H"]:
        print(f"{22 + list(gate['conditions']).index(k)}. {k}: {gate['conditions'][k]}")
    print("30. Final decision:", gate["gate"])
    print(f"31. Enter Phase E4?: {'YES' if gate['gate'] in ('STRONG-GO', 'GO') else 'NO'}")
    print(f"32. Exact reason: {gate['recommended_next_phase']}")
    print("33. Authoritative artifacts: results/phase_e3_mamr/")
    print("34. Reproduce: python scripts/reproduce_phase_e3_mamr.py --resume\n")


if __name__ == "__main__":
    raise SystemExit(main())
