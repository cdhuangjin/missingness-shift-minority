"""One-shot reproduction of Project E Phase E1/E2."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.dataset_registry import (  # noqa: E402
    freeze_hash,
    list_hash,
    load_all,
    load_registry,
)
from src.gate_decision import decide  # noqa: E402
from src.mechanism_analysis import (  # noqa: E402
    ambg_summary,
    dataset_consistency,
    imbalance_correlation,
    interaction_ols,
    mechanism_summary,
    model_consistency,
)
from src.paper_gate import paper_gate  # noqa: E402
from src.phase_e12_runner import (  # noqa: E402
    build_environments,
    rank_fsets,
    run_one_unit,
)
from src.resume_manager import ResumeManager  # noqa: E402
from src.reverse_control import reverse_exposure_effect  # noqa: E402
from src.slope_analysis import fit_slope  # noqa: E402
from src.statistical_analysis import (  # noqa: E402
    bh_fdr,
    bootstrap_ci,
    cohens_dz,
    effect_size,
    paired_ttest,
    rank_biserial,
    wilcoxon_paired,
)
from src.vulnerability import (  # noqa: E402
    hidden_failure as hidden_failure_fn,
    mvg,
    relative_degradation,
    retention,
)

MCAR_LABEL = {0.05: "mcar_05", 0.10: "mcar_10", 0.20: "mcar_20", 0.30: "mcar_30", 0.40: "mcar_40"}


def _envs(cfg) -> tuple[list[str], list[str], list[str], list[str]]:
    core = list(cfg.core_mechanisms)
    expansion = core + list(cfg.mechanism_expansion)
    sens = list(cfg.feature_set_envs)
    t3 = list(cfg.tier3["environments"])
    return core, expansion, sens, t3


def count_planned_runs(datasets, cfg, include_models=None, seeds=None, tier3=False) -> int:
    core, expansion, sens, t3envs = _envs(cfg)
    total = 0
    for ds in datasets:
        for seed in (seeds or cfg.seeds):
            for unit in build_units(
                ds["name"], cfg, core, expansion, sens, t3envs,
                include_models=include_models, tier3=tier3,
            ):
                total += len(unit[-1])
    return total


def build_units(ds_name, cfg, core, expansion, sens, t3envs, include_models=None, tier3=False):
    models = include_models or cfg.models
    units = []
    if "logistic_regression" in models:
        units.append(("logistic_regression", "p0", "median", "top", expansion))
        units.append(("logistic_regression", "p1", "median", "top", core))
    if "random_forest" in models:
        units.append(("random_forest", "p0", "median", "top", core))
    if "xgboost" in models:
        units.append(("xgboost", "p0", "median", "top", expansion))
        units.append(("xgboost", "p1", "median", "top", core))
        units.append(("xgboost", "native_missing", "median", "top", core))
    if "lightgbm" in models:
        units.append(("lightgbm", "p0", "median", "top", core))
        units.append(("lightgbm", "p1", "median", "top", core))
        units.append(("lightgbm", "native_missing", "median", "top", core))
    # sensitivity feature sets (linear + boosting tree).
    for model in ("logistic_regression", "xgboost"):
        if model in models:
            for fset in ("random", "low"):
                units.append((model, "p0", "median", fset, sens))
    if tier3 and ds_name in cfg.tier3["datasets"]:
        for model in cfg.tier3["models"]:
            if model in models:
                for imp in ("knn", "iterative"):
                    units.append((model, "p0", imp, "top", t3envs))
    return units


def _fit_units(datasets, cfg, include_models=None, seeds=None, tier3=False, rm=None):
    core, expansion, sens, t3envs = _envs(cfg)
    raw_by_key = {}
    missing_by_key = {}
    completed = 0
    failed = 0
    total = 0
    for ds in datasets:
        X, y = ds["X"], ds["y"]
        fsets = rank_fsets(X, y, ds["numeric_cols"], cfg)
        for seed in (seeds or cfg.seeds):
            units = build_units(
                ds["name"], cfg, core, expansion, sens, t3envs,
                include_models=include_models, tier3=tier3,
            )
            for (model, pipeline, imputer_name, fset, envs) in units:
                feat_cols = fsets.get(fset, [])
                keys = [
                    f"{ds['name']}|{seed}|{model}|{pipeline}|{imputer_name}|{fset}|{env}"
                    for env in envs
                ]
                if all(rm.is_complete(k) for k in keys):
                    completed += len(keys)
                    total += len(keys)
                    continue
                try:
                    raw_rows, missing_rows = run_one_unit(
                        ds, seed, model, pipeline, imputer_name, fset, feat_cols, envs, cfg
                    )
                    for env, rawrow, mrow in zip(envs, raw_rows, missing_rows):
                        key = f"{ds['name']}|{seed}|{model}|{pipeline}|{imputer_name}|{fset}|{env}"
                        rm.save(key, {"raw": rawrow, "missing": mrow})
                    completed += len(keys)
                    total += len(keys)
                except Exception as exc:  # noqa: BLE001
                    failed += len(keys)
                    total += len(keys)
                    for env in envs:
                        key = f"{ds['name']}|{seed}|{model}|{pipeline}|{imputer_name}|{fset}|{env}"
                        rm.record_failure(key, f"{type(exc).__name__}: {exc}")
                    print(f"[warn] failed {ds['name']} {model} {pipeline} {fset}: {exc}")
            rm.save_progress(completed, total, failed, None)
    return rm


def _merge_results(rm) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw_rows, missing_rows = [], []
    for path in rm.raw_dir.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except Exception:  # noqa: BLE001
            continue
        data = payload.get("data", {})
        if isinstance(data, dict):
            if data.get("raw"):
                raw_rows.append(data["raw"])
            if data.get("missing"):
                missing_rows.append(data["missing"])
    raw = pd.DataFrame(raw_rows)
    missing = pd.DataFrame(missing_rows)
    return raw, missing


def _primary(raw):
    return raw[
        (raw["pipeline"] == "p0") & (raw["imputer"] == "median") & (raw["feature_set"] == "top")
    ]


def compute_vulnerability(raw) -> pd.DataFrame:
    rows = []
    for keys, grp in raw.groupby(["dataset", "seed", "model", "pipeline", "imputer", "feature_set"]):
        iid = grp[grp["environment"] == "mcar_05"]
        if iid.empty:
            continue
        b = iid.iloc[0]
        for _, r in grp[grp["environment"] != "mcar_05"].iterrows():
            eps = 1e-12
            miRD_r = relative_degradation(b["minority_recall"], r["minority_recall"], eps)
            maRD_r = relative_degradation(b["majority_recall"], r["majority_recall"], eps)
            miRD_f = relative_degradation(b["minority_f1"], r["minority_f1"], eps)
            maRD_f = relative_degradation(b["majority_f1"], r["majority_f1"], eps)
            rows.append(
                {
                    "dataset": keys[0],
                    "seed": keys[1],
                    "model": keys[2],
                    "pipeline": keys[3],
                    "imputer": keys[4],
                    "feature_set": keys[5],
                    "environment": r["environment"],
                    "missing_rate": r["missing_rate"],
                    "MiRD_recall": miRD_r,
                    "MaRD_recall": maRD_r,
                    "MVG_recall": mvg(miRD_r, maRD_r),
                    "MiRD_f1": miRD_f,
                    "MaRD_f1": maRD_f,
                    "MVG_f1": mvg(miRD_f, maRD_f),
                    "MRR_AUPRC": retention(r["AUPRC"], b["AUPRC"], eps),
                    "IID_auprc": b["AUPRC"],
                    "IID_minority_recall": b["minority_recall"],
                    "shifted_minority_recall": r["minority_recall"],
                }
            )
    return pd.DataFrame(rows)


def compute_slopes(raw) -> pd.DataFrame:
    p0 = _primary(raw)
    mcar = p0[p0["environment"].isin(MCAR_LABEL.values())]
    rows = []
    for (ds, model), grp in mcar.groupby(["dataset", "model"]):
        rates, min_rec, maj_rec = [], [], []
        for rate, env in sorted(MCAR_LABEL.items()):
            sub = grp[grp["environment"] == env]
            if sub.empty:
                continue
            rates.append(rate)
            min_rec.append(float(sub["minority_recall"].mean()))
            maj_rec.append(float(sub["majority_recall"].mean()))
        if len(rates) < 3:
            continue
        ms, mi = fit_slope(rates, min_rec)
        js, ji = fit_slope(rates, maj_rec)
        rows.append(
            {
                "dataset": ds,
                "model": model,
                "minority_slope": ms,
                "majority_slope": js,
                "slope_gap": abs(ms) - abs(js),
                "slope_ratio": abs(ms) / (abs(js) + 1e-12),
                "R2_minority": _r2(rates, min_rec, ms, mi),
                "R2_majority": _r2(rates, maj_rec, js, ji),
            }
        )
    return pd.DataFrame(rows)


def _r2(x, y, b, a):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pred = a + b * x
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return 1 - ss_res / ss_tot if ss_tot > 0 else np.nan


def compute_cc(raw) -> pd.DataFrame:
    p0 = _primary(raw)
    rows = []
    for (ds, model), grp in p0.groupby(["dataset", "model"]):
        for cc in ("cc_20", "cc_30", "cc_40", "cc_50"):
            c = grp[grp["environment"] == cc]
            m = grp[grp["environment"] == f"matched_mcar_{cc.replace('_', '')}"]
            if c.empty or m.empty:
                continue
            ccep_r = float(m["minority_recall"].mean() - c["minority_recall"].mean())
            ccep_f = float(m["minority_f1"].mean() - c["minority_f1"].mean())
            ccep_ap = float(m["AUPRC"].mean() - c["AUPRC"].mean())
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "cc_level": cc,
                    "matched_mcar_recall": float(m["minority_recall"].mean()),
                    "cc_recall": float(c["minority_recall"].mean()),
                    "CCEP_recall": ccep_r,
                    "CCEP_f1": ccep_f,
                    "CCEP_auprc": ccep_ap,
                }
            )
    return pd.DataFrame(rows)


def compute_hidden(raw) -> pd.DataFrame:
    p0 = _primary(raw)
    rows = []
    for (ds, model, seed), grp in p0.groupby(["dataset", "model", "seed"]):
        iid = grp[grp["environment"] == "mcar_05"]
        if iid.empty:
            continue
        a0 = float(iid["AUROC"].iloc[0])
        ap0 = float(iid["AUPRC"].iloc[0])
        r0 = float(iid["minority_recall"].iloc[0])
        for _, r in grp[grp["environment"] != "mcar_05"].iterrows():
            auroc_drop = a0 - float(r["AUROC"])
            auprc_drop = ap0 - float(r["AUPRC"])
            recall_drop = r0 - float(r["minority_recall"])
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "seed": seed,
                    "environment": r["environment"],
                    "AUROC_drop": auroc_drop,
                    "AUPRC_drop": auprc_drop,
                    "minority_recall_drop": recall_drop,
                    "hidden_auroc": hidden_failure_fn(a0, float(r["AUROC"]), r0, float(r["minority_recall"])),
                    "hidden_auprc": (auprc_drop <= 0.05) and (recall_drop >= 0.10),
                }
            )
    return pd.DataFrame(rows)


def run_statistics(vuln) -> dict:
    p0 = vuln[(vuln["pipeline"] == "p0") & (vuln["feature_set"] == "top") & (vuln["imputer"] == "median")]
    mir = p0["MiRD_recall"].dropna().to_numpy()
    mar = p0["MaRD_recall"].dropna().to_numpy()
    stat, p_raw = wilcoxon_paired(mir, mar)
    t, p_t = paired_ttest(mir, mar)
    # Effect is defined as MiRD - MaRD; the helper computes (b - a), so pass
    # (mar, mir) to make a larger minority degradation read as a positive effect.
    es = effect_size(mar, mir, "rank_biserial")
    dz = cohens_dz(mar, mir)
    p_fdr = float(bh_fdr([p_raw])[0]) if not np.isnan(p_raw) else np.nan
    ci = bootstrap_ci(p0["MVG_recall"].dropna(), level=0.95, iters=2000)
    return {
        "n": int(len(mir)),
        "wilcoxon_stat": stat,
        "p_raw": p_raw,
        "p_paired_t": p_t,
        "p_fdr": p_fdr,
        "rank_biserial": es,
        "cohens_dz": dz,
        "mean_MVG": float(p0["MVG_recall"].mean()),
        "median_MVG": float(p0["MVG_recall"].median()),
        "MVG_ci_low": float(ci[0]),
        "MVG_ci_high": float(ci[1]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Project E Phase E1/E2")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase_e12.yaml"))
    ap.add_argument("--registry", default=str(PROJECT_ROOT / "configs" / "datasets_phase_e12.yaml"))
    ap.add_argument("--datasets", nargs="*", default=None)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.seeds:
        cfg.seeds = args.seeds
    registry = load_registry(args.registry)
    datasets = load_all(cfg, registry)
    if args.datasets:
        names = set(args.datasets)
        datasets = [d for d in datasets if d["name"] in names]
    if args.smoke:
        # force 2 datasets x 2 models x 1 seed.
        ds_filter = {"default_credit_card", "htr2"}
        datasets = [d for d in datasets if d["name"] in ds_filter]
        cfg.seeds = [42]
        args.models = ["logistic_regression", "xgboost"]

    results_dir = Path(cfg.paths["results"])
    results_dir.mkdir(parents=True, exist_ok=True)
    rm = ResumeManager(results_dir)
    n_planned = count_planned_runs(
        datasets, cfg, include_models=args.models, seeds=cfg.seeds, tier3=not args.smoke,
    )

    _fit_units(
        datasets, cfg,
        include_models=args.models, seeds=cfg.seeds, tier3=not args.smoke, rm=rm,
    )
    raw, missing = _merge_results(rm)

    results_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    missing.to_csv(results_dir / "missingness_profiles.csv", index=False)

    vuln = compute_vulnerability(raw)
    vuln.to_csv(results_dir / "vulnerability_results.csv", index=False)
    slopes = compute_slopes(raw)
    slopes.to_csv(results_dir / "slope_results.csv", index=False)
    cc = compute_cc(raw)
    cc.to_csv(results_dir / "class_conditional_results.csv", index=False)
    hidden = compute_hidden(raw)
    hidden.to_csv(results_dir / "hidden_failure_results.csv", index=False)

    stats = run_statistics(vuln)

    # summaries
    pv = vuln[(vuln["pipeline"] == "p0") & (vuln["feature_set"] == "top") & (vuln["imputer"] == "median")]
    ds_cons = dataset_consistency(pv)
    model_cons = model_consistency(pv)
    mech = mechanism_summary(pv)
    ds_cons.to_csv(results_dir / "dataset_summary.csv", index=False)
    model_cons.to_csv(results_dir / "model_summary.csv", index=False)
    mech.to_csv(results_dir / "mechanism_summary.csv", index=False)

    profiles = _dataset_profiles(datasets)
    imb_corr = imbalance_correlation(profiles, ds_cons, slopes)
    imb_corr.to_csv(results_dir / "imbalance_correlation.csv", index=False)

    ambg = ambg_summary(raw)
    ambg.to_csv(results_dir / "ambg_results.csv", index=False)

    # Reverse control table
    rev_rows = []
    for (ds, model), grp in pv.groupby(["dataset", "model"]):
        n = grp[grp["environment"] == "cc_30"]["MVG_recall"]
        rv = grp[grp["environment"] == "reverse_cc_30"]["MVG_recall"]
        if len(n) and len(rv):
            rev_rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "normal_cc_MVG": float(n.mean()),
                    "reverse_cc_MVG": float(rv.mean()),
                    "delta_MVG_reverse": float(reverse_exposure_effect(n.mean(), rv.mean())),
                }
            )
    reverse = pd.DataFrame(rev_rows)
    reverse.to_csv(results_dir / "reverse_control_results.csv", index=False)

    # Enrich statistics with bootstrap CIs and descriptive counts.
    stats["n_datasets"] = len(datasets)
    stats["n_positive_mvg_datasets"] = int((ds_cons["mean_MVG"] > 0).sum()) if not ds_cons.empty else 0
    stats["n_mvg_ge_010_datasets"] = int((ds_cons["mean_MVG"] >= 0.10).sum()) if not ds_cons.empty else 0
    if not cc.empty:
        ccep = cc["CCEP_recall"].dropna()
        stats["CCEP_recall_mean"] = float(ccep.mean())
        lo, hi = bootstrap_ci(ccep, level=0.95, iters=2000)
        stats["CCEP_ci_low"], stats["CCEP_ci_high"] = float(lo), float(hi)
    if not slopes.empty:
        sl = slopes["minority_slope"].dropna()
        stats["minority_slope_mean"] = float(sl.mean())
        lo, hi = bootstrap_ci(sl, level=0.95, iters=2000)
        stats["minority_slope_ci_low"], stats["minority_slope_ci_high"] = float(lo), float(hi)
    if not reverse.empty:
        rev = reverse["delta_MVG_reverse"].dropna()
        stats["reverse_delta_mean"] = float(rev.mean())
        lo, hi = bootstrap_ci(rev, level=0.95, iters=2000)
        stats["reverse_ci_low"], stats["reverse_ci_high"] = float(lo), float(hi)
        stats["n_reversal_datasets"] = int((reverse.groupby("dataset")["delta_MVG_reverse"].mean() >= 0.05).sum())

    pi_merged = pv.merge(
        profiles[["dataset", "imbalance_ratio"]], on="dataset", how="left"
    ).copy()
    pi_merged["log_imbalance"] = np.log10(pi_merged["imbalance_ratio"].to_numpy())
    interaction = interaction_ols(pi_merged)

    ind_eff = _indicator_effect(raw)
    nat_eff = _native_effect(raw)
    imp_eff = _imputation_effect(raw)

    # Paper gate
    cc30 = cc[cc["cc_level"] == "cc_30"] if not cc.empty else pd.DataFrame()
    hidden_ds = (
        hidden.groupby("dataset")[["hidden_auroc"]].sum().reset_index()
        if not hidden.empty else pd.DataFrame(columns=["dataset", "hidden_auroc"])
    )
    hidden_ds.columns = ["dataset", "n_hidden"]
    reverse_ds = reverse.groupby("dataset")["delta_MVG_reverse"].mean().reset_index()
    reverse_ds = (
        reverse_ds[["dataset", "delta_MVG_reverse"]]
        if not reverse_ds.empty else pd.DataFrame(columns=["dataset", "delta_MVG_reverse"])
    )
    seed_noise_ok = _seed_noise_ok(pv)
    stats["seed_noise_ok"] = seed_noise_ok
    stats["seed_noise_std"] = float(pv.groupby("seed")["MVG_recall"].mean().std(ddof=0)) if not pv.empty else np.nan
    stats_frame = pd.DataFrame([stats])
    stats_frame.to_csv(results_dir / "statistics_results.csv", index=False)
    verdict, gate_meta = paper_gate(
        ds_cons, model_cons, slopes, cc30, hidden_ds, reverse_ds,
        stats, n_datasets=len(datasets), effect_vs_seed_noise_ok=seed_noise_ok,
    )
    paper_gate_json = {
        "paper_gate": verdict,
        "conditions": gate_meta["conditions"],
        "counts": gate_meta["counts"],
        "n_hit": gate_meta["n_hit"],
        "statistics": stats,
    }
    paper_gate_json = _json_safe(paper_gate_json)
    (results_dir / "paper_gate.json").write_text(json.dumps(paper_gate_json, indent=2))

    figures = generate_figures(raw, vuln, slopes, cc, reverse, hidden, ds_cons, mech,
                               profiles, results_dir / "figures")
    manifest = _json_safe(
        build_manifest(cfg, registry, datasets, raw, stats, verdict, rm, n_planned)
    )
    (results_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    table_paths = write_tables(
        results_dir / "tables", profiles, raw, vuln, slopes, cc, reverse, hidden, stats,
    )
    report = build_report(
        cfg, datasets, raw, vuln, slopes, cc, reverse, hidden, stats,
        ds_cons, model_cons, mech, ambg, imb_corr, verdict, paper_gate_json, figures,
        interaction, ind_eff, nat_eff, imp_eff, table_paths,
    )
    (results_dir / "phase_e12_report.md").write_text(report, encoding="utf-8")

    print(f"\nProject E Phase E1/E2 complete. Paper Gate: {verdict}")
    print(f"datasets={len(datasets)} raw_rows={len(raw)} vuln_rows={len(vuln)}")
    print(f"mean MVG={stats['mean_MVG']:.4f} median MVG={stats['median_MVG']:.4f}")
    print(f"wilcoxon p={stats['p_raw']:.5f} FDR p={stats['p_fdr']:.5f} eff={stats['rank_biserial']:.3f}")
    return 0


def _dataset_profiles(datasets):
    rows = []
    for d in datasets:
        counts = d["y"].value_counts()
        rows.append(
            {
                "dataset": d["name"],
                "n_samples": int(d["X"].shape[0]),
                "n_features": int(d["X"].shape[1]),
                "n_numeric": int(len(d["numeric_cols"])),
                "minority_count": int(counts.get(1, 0)),
                "majority_count": int(counts.get(0, 0)),
                "imbalance_ratio": d["imbalance_ratio"],
                "natural_missing_rate": d["natural_missing_count"] / max(d["X"].size, 1),
                "source": str(d["source_info"]),
            }
        )
    return pd.DataFrame(rows)


def _seed_noise_ok(vuln) -> bool:
    if vuln.empty:
        return False
    per_seed = vuln.groupby("seed")["MVG_recall"].mean()
    if per_seed.std(ddof=0) == 0:
        return True
    effect = vuln["MVG_recall"].mean()
    return abs(effect) > per_seed.std(ddof=0)


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays to JSON-serializable types."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _indicator_effect(raw) -> dict:
    """P1 (median + indicator) vs P0 median: mean MVG and minority recall."""
    base = raw[(raw["pipeline"] == "p0") & (raw["imputer"] == "median") & (raw["feature_set"] == "top")]
    ind = raw[(raw["pipeline"] == "p1") & (raw["imputer"] == "median") & (raw["feature_set"] == "top")]
    if base.empty or ind.empty:
        return {"supported": False}
    envs = set(base["environment"]) & set(ind["environment"])
    b = base[base["environment"].isin(envs)]
    i = ind[ind["environment"].isin(envs)]
    return {
        "supported": True,
        "p0_mean_minority_recall": float(b["minority_recall"].mean()),
        "p1_mean_minority_recall": float(i["minority_recall"].mean()),
        "p0_mean_AUPRC": float(b["AUPRC"].mean()),
        "p1_mean_AUPRC": float(i["AUPRC"].mean()),
    }


def _native_effect(raw) -> dict:
    """Native-missing XGB/LGBM vs their P0 median baseline."""
    out = {}
    for model in ("xgboost", "lightgbm"):
        base = raw[(raw["model"] == model) & (raw["pipeline"] == "p0")
                   & (raw["imputer"] == "median") & (raw["feature_set"] == "top")]
        nat = raw[(raw["model"] == model) & (raw["pipeline"] == "native_missing")
                  & (raw["imputer"] == "median") & (raw["feature_set"] == "top")]
        if base.empty or nat.empty:
            out[model] = {"supported": False}
            continue
        envs = set(base["environment"]) & set(nat["environment"])
        b = base[base["environment"].isin(envs)]
        n = nat[nat["environment"].isin(envs)]
        out[model] = {
            "supported": True,
            "p0_mean_minority_recall": float(b["minority_recall"].mean()),
            "native_mean_minority_recall": float(n["minority_recall"].mean()),
            "p0_mean_AUPRC": float(b["AUPRC"].mean()),
            "native_mean_AUPRC": float(n["AUPRC"].mean()),
        }
    return out


def _imputation_effect(raw) -> dict:
    """Tier-3 median vs KNN vs Iterative across the three representative datasets."""
    out = {}
    for imp in ("median", "knn", "iterative"):
        sub = raw[(raw["pipeline"] == "p0") & (raw["imputer"] == imp) & (raw["feature_set"] == "top")]
        out[imp] = {
            "supported": not sub.empty,
            "mean_minority_recall": float(sub["minority_recall"].mean()) if not sub.empty else np.nan,
            "mean_AUPRC": float(sub["AUPRC"].mean()) if not sub.empty else np.nan,
            "n": int(len(sub)),
        }
    return out


def _indicator_verdict(d: dict) -> str:
    if not d.get("supported"):
        return "n/a (not enough P1 rows)"
    delta = float(d.get("p1_mean_minority_recall", 0)) - float(d.get("p0_mean_minority_recall", 0))
    if abs(delta) < 0.01:
        return f"neutral (recall {d['p0_mean_minority_recall']:.3f}->{d['p1_mean_minority_recall']:.3f}, AUPRC {d['p0_mean_AUPRC']:.3f}->{d['p1_mean_AUPRC']:.3f})"
    return f"{'helped' if delta > 0 else 'hurt'} by {abs(delta):.3f} recall"


def _native_verdict(d: dict) -> str:
    parts = []
    for model, v in d.items():
        if not v.get("supported"):
            parts.append(f"{model}=n/a")
            continue
        delta = float(v.get("native_mean_minority_recall", 0)) - float(v.get("p0_mean_minority_recall", 0))
        if abs(delta) < 0.01:
            parts.append(f"{model}=neutral ({v['p0_mean_minority_recall']:.3f}->{v['native_mean_minority_recall']:.3f})")
        else:
            parts.append(f"{model}={'helped' if delta > 0 else 'hurt'} ({delta:+.3f})")
    return "; ".join(parts)


def _imputation_verdict(d: dict) -> str:
    median = d.get("median", {})
    if not median.get("supported"):
        return "n/a"
    base = float(median.get("mean_minority_recall", np.nan))
    parts = []
    for imp in ("knn", "iterative"):
        v = d.get(imp, {})
        if v.get("supported"):
            val = float(v.get("mean_minority_recall", np.nan))
            dl = val - base
            parts.append(f"{imp}={val:.3f} ({dl:+.3f} vs median)")
    return f"median={base:.3f}; " + "; ".join(parts)


def write_tables(outdir, profiles, raw, vuln, slopes, cc, reverse, hidden, stats) -> list[Path]:
    """Write the eight required tables under tables/."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    table_data = [
        ("table1_dataset_profile.csv", profiles),
        ("table2_core_performance.csv", raw),
        ("table3_vulnerability.csv", vuln),
        ("table4_slope.csv", slopes),
        ("table5_class_conditional.csv", cc),
        ("table6_reverse_control.csv", reverse),
        ("table7_hidden_failure.csv", hidden),
        ("table8_statistics.csv", pd.DataFrame([stats])),
    ]
    for name, df in table_data:
        p = outdir / name
        df.to_csv(p, index=False)
        paths.append(p)
    return paths


def build_manifest(cfg, registry, datasets, raw, stats, verdict, rm, n_planned=None) -> dict:
    versions = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "lightgbm", "matplotlib", "seaborn", "scipy", "statsmodels", "ucimlrepo"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            versions[mod] = "n/a"
    try:
        git = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:  # noqa: BLE001
        git = "n/a"
    profiles = _dataset_profiles(datasets)
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git,
        "python_version": platform.python_version(),
        "package_versions": versions,
        "dataset_sources": {
            d["name"]: d["source_info"] for d in datasets
        },
        "dataset_hashes": {
            d["name"]: hashlib.sha1(
                f"{d['name']}|{d['X'].shape}|{profiles.set_index('dataset').loc[d['name']]['imbalance_ratio']:.4f}".encode()
            ).hexdigest() for d in datasets
        },
        "dataset_freeze_hash": freeze_hash(registry),
        "dataset_list_hash": list_hash(registry),
        "config_hash": hashlib.sha1(
            json.dumps({k: v for k, v in cfg.__dict__.items() if k != "project_root"},
                       sort_keys=True, default=str).encode()).hexdigest(),
        "seeds": cfg.seeds,
        "model_configs": cfg.models,
        "missingness_configs": {
            "mcar_rates": cfg.mcar_rates,
            "mar": cfg.mar,
            "class_conditional": cfg.class_conditional,
            "reverse": cfg.reverse,
            "block": cfg.block,
        },
        "n_expected_runs": n_planned if n_planned is not None else len(list(rm.raw_dir.glob("*.json"))),
        "n_completed_runs": rm.completed_count(),
        "n_failed_runs": len(list(rm.failed_path.read_text().splitlines())) - 1 if rm.failed_path.exists() else 0,
        "gate": verdict,
    }


def generate_figures(raw, vuln, slopes, cc, reverse, hidden, ds_cons, mech, profiles, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.statistical_analysis import bootstrap_ci

    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    p0 = raw[(raw["pipeline"] == "p0") & (raw["feature_set"] == "top") & (raw["imputer"] == "median")]
    mcar = p0[p0["environment"].isin(MCAR_LABEL.values())].copy()
    mcar["rate"] = mcar["environment"].map({v: k for k, v in MCAR_LABEL.items()})
    ds_names = sorted(mcar["dataset"].unique())

    # Fig 1 & 2: recall vs missing rate, small multiples.
    for metric, name in (("minority_recall", "fig1_minority_recall_vs_rate"),
                         ("majority_recall", "fig2_majority_recall_vs_rate")):
        n = len(ds_names)
        cols = 4
        rows = int(np.ceil(n / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(16, 3.2 * rows), sharex=True, sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for ax, ds in zip(axes, ds_names):
            sub = mcar[mcar["dataset"] == ds].groupby(["environment", "rate"])[metric].mean().reset_index()
            sub = sub.sort_values("rate")
            ax.plot(sub["rate"], sub[metric], marker="o")
            ax.set_title(f"{ds}", fontsize=8)
            ax.grid(alpha=0.3)
            ax.set_xlabel("MCAR rate")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle(f"Figure: {metric} vs MCAR rate")
        fig.tight_layout()
        path = outdir / f"{name}.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 3: minority vs majority slope scatter.
    if not slopes.empty:
        fig, ax = plt.subplots()
        ax.scatter(slopes["majority_slope"], slopes["minority_slope"], s=36, alpha=0.8)
        lim = max(abs(slopes["majority_slope"].min()), abs(slopes["majority_slope"].max()),
                  abs(slopes["minority_slope"].min()), abs(slopes["minority_slope"].max())) * 1.2
        ax.plot([-lim, lim], [-lim, lim], "k--", lw=1, label="y = x")
        ax.axhline(0, color="grey", lw=0.7)
        ax.axvline(0, color="grey", lw=0.7)
        ax.set_xlabel("Majority slope")
        ax.set_ylabel("Minority slope")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig3_minority_vs_majority_slope.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 4: MVG heatmap dataset x model.
    pv = vuln[(vuln["pipeline"] == "p0") & (vuln["feature_set"] == "top") & (vuln["imputer"] == "median")]
    if not pv.empty:
        piv = pv.pivot_table(index="dataset", columns="model", values="MVG_recall", aggfunc="mean").dropna(how="all")
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(piv.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-0.2, vmax=0.4)
        ax.set_xticks(np.arange(piv.shape[1]))
        ax.set_xticklabels(piv.columns, fontsize=8)
        ax.set_yticks(np.arange(piv.shape[0]))
        ax.set_yticklabels(piv.index, fontsize=8)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.iloc[i, j]
                if pd.notna(v):
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="MVG_recall")
        ax.set_title("Figure 4: MVG by dataset x model")
        fig.tight_layout()
        path = outdir / "fig4_mvg_heatmap.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 5: matched MCAR vs CC minority recall.
    if not cc.empty:
        fig, ax = plt.subplots()
        cc30 = cc[cc["cc_level"] == "cc_30"]
        labels = [f"{r['dataset'][:10]}\n{r['model'][:8]}" for _, r in cc30.iterrows()]
        x = np.arange(len(cc30))
        ax.bar(x - 0.18, cc30["matched_mcar_recall"], 0.34, label="matched MCAR")
        ax.bar(x + 0.18, cc30["cc_recall"], 0.34, label="class-conditional")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("Minority recall")
        ax.set_title("Figure 5: Matched MCAR vs class-conditional")
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        path = outdir / "fig5_matched_vs_cc.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 6: normal CC vs reverse CC MVG reversal.
    if not reverse.empty:
        fig, ax = plt.subplots()
        labels = [f"{r['dataset'][:10]}" for _, r in reverse.iterrows()]
        x = np.arange(len(reverse))
        ax.bar(x - 0.18, reverse["normal_cc_MVG"], 0.34, label="normal CC")
        ax.bar(x + 0.18, reverse["reverse_cc_MVG"], 0.34, label="reverse CC")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=7)
        ax.set_ylabel("MVG_recall")
        ax.set_title("Figure 6: Normal vs reverse class-conditional MVG")
        ax.axhline(0, color="k", lw=0.8)
        ax.grid(alpha=0.3, axis="y")
        ax.legend()
        fig.tight_layout()
        path = outdir / "fig6_reverse_control.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 7: hidden failure scatter.
    if not hidden.empty:
        fig, ax = plt.subplots()
        ax.scatter(hidden["AUROC_drop"], hidden["minority_recall_drop"], s=20, alpha=0.7)
        ax.axvspan(-1, 0.03, color="green", alpha=0.12)
        ax.axhspan(0.10, 1.0, color="green", alpha=0.12)
        ax.axvline(0.03, color="green", ls="--", lw=1)
        ax.axhline(0.10, color="green", ls="--", lw=1)
        ax.set_xlabel("AUROC drop")
        ax.set_ylabel("Minority recall drop")
        ax.set_title("Figure 7: Hidden minority failure")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig7_hidden_failure.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 8: imbalance ratio vs MVG.
    if not profiles.empty and not ds_cons.empty:
        m = ds_cons.merge(profiles[["dataset", "imbalance_ratio"]], on="dataset")
        fig, ax = plt.subplots()
        ax.scatter(m["imbalance_ratio"], m["mean_MVG"], s=50)
        ax.set_xscale("log")
        ax.set_xlabel("Imbalance ratio (log)")
        ax.set_ylabel("Mean MVG_recall")
        ax.set_title("Figure 8: imbalance severity vs vulnerability")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig8_imbalance_vs_mvg.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 9: missing rate vs MVG (trend).
    if not pv.empty:
        grp = pv.groupby(pv["missing_rate"].round(3))["MVG_recall"].agg(["mean", "count"]).reset_index()
        grp = grp[grp["count"] >= 1].sort_values("missing_rate")
        fig, ax = plt.subplots()
        ax.plot(grp["missing_rate"], grp["mean"], marker="o")
        ax.set_xlabel("Overall missing rate")
        ax.set_ylabel("Mean MVG_recall")
        ax.set_title("Figure 9: vulnerability vs missingness severity")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        path = outdir / "fig9_missing_rate_vs_mvg.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)

    # Fig 10: mechanism-level mean MVG with 95% CI.
    if not mech.empty:
        rows = []
        for env, grp in mech.groupby("environment"):
            vals = pv[pv["environment"] == env]["MVG_recall"].dropna()
            lo, hi = bootstrap_ci(vals, level=0.95, iters=500)
            rows.append({"environment": env, "mean": vals.mean(), "lo": lo, "hi": hi})
        df = pd.DataFrame(rows)
        fig, ax = plt.subplots(figsize=(11, 5))
        df = df.sort_values("mean")
        y = np.arange(len(df))
        ax.barh(y, df["mean"], xerr=[df["mean"] - df["lo"], df["hi"] - df["mean"]],
                align="center", alpha=0.8, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(df["environment"], fontsize=8)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel("Mean MVG_recall (95% CI)")
        ax.set_title("Figure 10: mechanism-level vulnerability")
        ax.grid(alpha=0.3, axis="x")
        fig.tight_layout()
        path = outdir / "fig10_mechanism_mvg.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        paths.append(path)
    return paths


def build_report(cfg, datasets, raw, vuln, slopes, cc, reverse, hidden, stats,
                 ds_cons, model_cons, mech, ambg, imb_corr, verdict, gate_json, figures,
                 interaction, ind_eff, nat_eff, imp_eff, table_paths):
    def fnum(v, digits=3):
        try:
            v = float(v)
        except (TypeError, ValueError):
            return "n/a"
        if np.isnan(v):
            return "n/a"
        return f"{v:.{digits}f}"

    prof = _dataset_profiles(datasets)
    n_ds = len(datasets)
    lines = []
    lines.append("# Project E Phase E1/E2 Report")
    lines.append("")
    lines.append(f"**Paper Gate: `{verdict}`**")
    lines.append("")
    lines.append("Q&A is computed from the primary protocol: `P0` median imputation, "
                 "`top` predictive feature set, `mcar_05` as the IID reference.")
    lines.append("")

    lines.append("## 1. Scientific Question")
    lines.append("")
    lines.append("> Does missingness distribution shift cause disproportionate minority-class "
                 "degradation in imbalanced tabular classification, even when conventional "
                 "aggregate metrics remain relatively stable?")
    lines.append("")

    lines.append("## 2. Dataset Profile (Table 1)")
    lines.append("")
    lines.append(prof.round(3).to_string(index=False))
    lines.append("")

    lines.append("## 3. Cross-Dataset Consistency")
    lines.append("")
    lines.append(ds_cons.round(3).to_string(index=False))
    lines.append("")
    lines.append("## 4. Model Consistency")
    lines.append("")
    lines.append(model_cons.round(3).to_string(index=False))
    lines.append("")

    lines.append("## 5. Mechanism Summary")
    if not mech.empty:
        lines.append(mech.round(3).to_string(index=False))
    lines.append("")

    lines.append("## 6. Class-Conditional vs Matched MCAR (Table 5)")
    if not cc.empty:
        lines.append(cc.round(3).to_string(index=False))
    lines.append("")

    lines.append("## 7. Reverse Control (Table 6)")
    if not reverse.empty:
        lines.append(reverse.round(3).to_string(index=False))
    lines.append("")

    lines.append("## 8. Hidden Minority Failure (Table 7)")
    if not hidden.empty:
        lines.append(hidden.round(3).to_string(index=False))
    lines.append("")

    lines.append("## 9. Aggregate-Metric Blindness (descriptive)")
    if not ambg.empty:
        lines.append(ambg.round(3).head(40).to_string(index=False))
        ambg_auroc = ambg[ambg["auroc_drop"] <= 0.03]
        lines.append("")
        lines.append(f"Cases with AUROC drop <= 0.03: {len(ambg_auroc)}; "
                     f"median AMBG (recall drop - AUROC drop): "
                     f"{ambg['AMBG'].median():.3f}.")
    lines.append("")

    lines.append("## 10. Imbalance Interaction (OLS)")
    if interaction and "r_squared" in interaction and interaction.get("r_squared") is not None:
        lines.append(f"n = {interaction['n']}, R^2 = {interaction['r_squared']:.3f}.")
        lines.append(f"Interaction estimate = {interaction['interaction_est']:.4f} "
                     f"(p = {interaction['interaction_p']:.4f}).")
    else:
        lines.append("Not enough observations for a stable interaction OLS.")
    lines.append("")

    lines.append("## 11. Statistics (Table 8)")
    lines.append("")
    for k in ("n", "wilcoxon_stat", "p_raw", "p_paired_t", "p_fdr", "rank_biserial",
              "cohens_dz", "mean_MVG", "median_MVG", "MVG_ci_low", "MVG_ci_high",
              "CCEP_recall_mean", "CCEP_ci_low", "CCEP_ci_high",
              "minority_slope_mean", "minority_slope_ci_low", "minority_slope_ci_high",
              "reverse_delta_mean", "reverse_ci_low", "reverse_ci_high",
              "seed_noise_ok", "seed_noise_std"):
        if k in stats:
            fields = ("p_raw", "p_paired_t", "p_fdr", "seed_noise_ok")
            if k in fields:
                lines.append(f"- {k}: {stats[k]}")
            else:
                lines.append(f"- {k}: {fnum(stats[k])}")
    lines.append("")

    lines.append("## 12. Missing Indicator Effect (P1 vs P0)")
    if ind_eff.get("supported"):
        lines.append(f"P0 mean minority recall = {fnum(ind_eff['p0_mean_minority_recall'])}, "
                     f"P1 (indicator) = {fnum(ind_eff['p1_mean_minority_recall'])}; "
                     f"AUPRC {fnum(ind_eff['p0_mean_AUPRC'])} -> {fnum(ind_eff['p1_mean_AUPRC'])}.")
    else:
        lines.append("Not enough P1 rows to evaluate.")
    lines.append("")

    lines.append("## 13. Native Missing Handling (XGBoost / LightGBM)")
    for model, d in nat_eff.items():
        if d.get("supported"):
            lines.append(f"- {model}: P0 minority recall = {fnum(d['p0_mean_minority_recall'])} "
                         f"-> native = {fnum(d['native_mean_minority_recall'])}; "
                         f"AUPRC {fnum(d['p0_mean_AUPRC'])} -> {fnum(d['native_mean_AUPRC'])}.")
        else:
            lines.append(f"- {model}: not enough native-missing rows.")
    lines.append("")

    lines.append("## 14. Imputation Robustness (median / KNN / Iterative)")
    for imp, d in imp_eff.items():
        if d.get("supported"):
            lines.append(f"- {imp}: minority recall = {fnum(d['mean_minority_recall'])}, "
                         f"AUPRC = {fnum(d['mean_AUPRC'])}, n = {d['n']}.")
        else:
            lines.append(f"- {imp}: n/a.")
    lines.append("")

    lines.append("## 15. Answers to the §89 Questions")
    lines.append("")
    mvg_pos = int((ds_cons["mean_MVG"] > 0).sum()) if not ds_cons.empty else 0
    mvg_ge = int((ds_cons["mean_MVG"] >= 0.10).sum()) if not ds_cons.empty else 0
    strongest = ds_cons.sort_values("mean_MVG", ascending=False).head(3) if not ds_cons.empty else None
    weakest = ds_cons.sort_values("mean_MVG").head(3) if not ds_cons.empty else None
    steeper = int((slopes["minority_slope"].abs() > slopes["majority_slope"].abs()).sum()) if not slopes.empty else 0
    mech_rank = mech[mech["metric"] == "MVG_recall"].sort_values("mean", ascending=False) if not mech.empty else None
    cc_ge = int((cc[cc["CCEP_recall"] >= 0.10]["dataset"].nunique())) if not cc.empty else 0
    hidden_cases = int((hidden["hidden_auroc"] > 0).sum()) if not hidden.empty else 0
    hidden_ds_n = int(hidden[hidden["hidden_auroc"] > 0]["dataset"].nunique()) if not hidden.empty else 0
    rev_ds_n = int((reverse.groupby("dataset")["delta_MVG_reverse"].mean() >= 0.05).sum()) if not reverse.empty else 0

    qa = [
        ("Q1 # datasets with minority more vulnerable (mean MVG > 0)",
         f"{mvg_pos}/{n_ds}"),
        ("Q2 mean / median MVG_recall",
         f"{fnum(stats.get('mean_MVG'))} / {fnum(stats.get('median_MVG'))}"),
        ("Q3 strongest datasets",
         ", ".join(r["dataset"] for _, r in strongest.iterrows()) + " (" +
         ", ".join(fnum(r["mean_MVG"]) for _, r in strongest.iterrows()) + ")") if strongest is not None else "n/a",
        ("Q4 weakest datasets",
         ", ".join(r["dataset"] for _, r in weakest.iterrows()) + " (" +
         ", ".join(fnum(r["mean_MVG"]) for _, r in weakest.iterrows()) + ")") if weakest is not None else "n/a",
        ("Q5 model direction consistent",
         "yes, all models positive" if not model_cons.empty and (model_cons["mean_MVG"] > 0).all()
         else "partial; see model table"),
        ("Q6 datasets with steeper minority slope",
         f"{steeper}/{len(slopes)} rows ({n_ds} datasets)"),
        ("Q7 most dangerous mechanism",
         (mech_rank.iloc[0]["environment"] + f" (mean MVG {mech_rank.iloc[0]['mean']:.3f})")
         if mech_rank is not None and len(mech_rank) else "n/a"),
        ("Q8 comparable-rate CC extra penalty",
         f"mean CCEP = {fnum(stats.get('CCEP_recall_mean'))}, "
         f"datasets CCEP >= 0.10 = {cc_ge}, 95% CI [{fnum(stats.get('CCEP_ci_low'))}, {fnum(stats.get('CCEP_ci_high'))}]"),
        ("Q9 reverse CC weakens/inverts MVG",
         f"normal = {fnum(stats.get('reverse_delta_mean'), 4)} (delta), "
         f"reversal datasets = {rev_ds_n}/{n_ds}"),
        ("Q10 hidden minority failure",
         f"{hidden_cases} dataset/model/seed cases across {hidden_ds_n} datasets"),
        ("Q11 AUROC masks minority recall failure",
         "Yes in some cases; see AMBG table" if not ambg.empty else "n/a"),
        ("Q12 imbalance ratio vs vulnerability",
         (", ".join(f"{r['variable']} rho={fnum(r['spearman_rho'])} p={fnum(r['p'])}"
                    for _, r in imb_corr.iterrows()) if not imb_corr.empty else "n/a")),
        ("Q13 missing indicator",
         _indicator_verdict(ind_eff)),
        ("Q14 native missing handling",
         _native_verdict(nat_eff)),
        ("Q15 imputation method changes conclusion",
         _imputation_verdict(imp_eff)),
        ("Q16 seed noise < effect",
         f"{stats.get('seed_noise_ok')} (seed std = {fnum(stats.get('seed_noise_std'))})"),
        ("Q17 bootstrap CI supports",
         f"MVG 95% CI [{fnum(stats.get('MVG_ci_low'))}, {fnum(stats.get('MVG_ci_high'))}]"),
        ("Q18 Wilcoxon + FDR significant",
         f"p_raw = {fnum(stats.get('p_raw'), 6)}, FDR p = {fnum(stats.get('p_fdr'), 6)}"),
        ("Q19 effect size",
         f"rank-biserial = {fnum(stats.get('rank_biserial'))}, Cohen's dz = {fnum(stats.get('cohens_dz'))}"),
        ("Q20 Paper Gate",
         verdict),
        ("Q21 worth entering MAMR?",
         "YES / likely yes" if verdict in ("STRONG-GO", "GO") else
         ("BORDERLINE (HOLD)" if verdict == "HOLD" else "NO (STOP-PIVOT)")),
        ("Q22 MAMR should solve",
         "Minority-aware missingness robustification: mitigate the class-asymmetric "
         "recall collapse caused by exposure concentrated on the minority class, "
         "under the mechanism with the highest MVG."),
    ]
    for q, a in qa:
        lines.append(f"- **{q}:** {a}")
    lines.append("")

    lines.append("## 16. Paper Gate Conditions")
    lines.append("")
    lines.append(json.dumps(gate_json, indent=2))
    lines.append("")

    lines.append("## 17. Tables")
    for p in table_paths:
        lines.append(f"- {p}")
    lines.append("")

    lines.append("## 18. Figures")
    for p in figures:
        lines.append(f"- {p}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
