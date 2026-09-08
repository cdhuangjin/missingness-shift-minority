"""One-shot reproduction of Project E Gate A.

Flow: download/load data -> split -> source missingness -> fit preprocessing ->
train models -> build target missingness environments -> evaluate -> compute
vulnerability -> slope analysis -> figures -> automatic gate -> report/manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config, resolve  # noqa: E402
from src.data_loader import load_all_datasets  # noqa: E402
from src.gate_decision import build_evidence, decide  # noqa: E402
from src.metrics import compute_metrics, threshold_05  # noqa: E402
from src.missingness_generator import (  # noqa: E402
    apply_mask,
    generate_class_conditional,
    generate_feature_dependent,
    generate_mcar,
    measure_missingness,
    pick_driver_col,
)
from src.models import make_model  # noqa: E402
from src.plotting import generate_all_figures  # noqa: E402
from src.preprocessing import (  # noqa: E402
    MedianModeImputer,
    apply_categorical_encoders,
    fit_categorical_encoders,
    rank_numeric_features,
    stratified_split,
)
from src.slope_analysis import fit_slope, slope_gap, slope_ratio  # noqa: E402
from src.vulnerability import (  # noqa: E402
    mvg,
    relative_degradation,
    retention,
)


MCAR_RATE = {"iid_mcar": 0.05, "mild_mcar": 0.10, "severe_mcar": 0.30}


def _env_seed(seed: int, env: str, idx: int) -> int:
    return int(seed * 100_003 + (idx + 1) * 10_007)


def _build_test_matrix(
    X_masked: pd.DataFrame,
    imputer: MedianModeImputer,
    encoders: dict[str, dict],
    indicator_cols: list[str],
    add_indicator: bool,
) -> pd.DataFrame:
    out = imputer.transform(X_masked)
    if encoders:
        out = apply_categorical_encoders(out, encoders)
    if add_indicator:
        for col in indicator_cols:
            ind = X_masked[col].isna().astype(int)
            out[f"{col}__missing_indicator"] = ind.to_numpy()
    return out


def _fit_scaler(matrix: pd.DataFrame):
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    scaler.fit(matrix.select_dtypes(include=[np.number]))
    return scaler


def _apply_scaler(matrix: pd.DataFrame, scaler) -> pd.DataFrame:
    num = matrix.select_dtypes(include=[np.number]).columns
    arr = scaler.transform(matrix[num])
    out = matrix.copy()
    out[num] = arr
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Reproduce Project E Gate A")
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "gate_a.yaml"),
        help="path to gate_a.yaml",
    )
    parser.add_argument("--datasets", nargs="*", default=None, help="restrict dataset names")
    parser.add_argument("--models", nargs="*", default=None, help="restrict model names")
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=None, help="restrict seed list"
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.seeds:
        cfg.seeds = args.seeds
    if args.models:
        cfg.models = args.models

    results_dir = resolve(cfg, "results")
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / "figures"

    datasets = load_all_datasets(cfg, names=args.datasets)

    raw_rows: list[dict] = []
    missing_rows: list[dict] = []
    vuln_rows: list[dict] = []

    for ds in datasets:
        X, y = ds["X"], ds["y"]
        numeric_cols = ds["numeric_cols"]
        categorical_cols = ds["categorical_cols"]
        topk = rank_numeric_features(
            X, y, numeric_cols, cfg.features["max_affected"], cfg.features["rank_method"]
        )
        driver = pick_driver_col(X, topk, numeric_cols)
        ds_print = ds["name"]

        for seed in cfg.seeds:
            train_idx, val_idx, test_idx = stratified_split(X, y, cfg.split, seed)
            X_train_raw = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_val_raw = X.iloc[val_idx]
            y_val = y.iloc[val_idx]
            X_test_raw = X.iloc[test_idx]
            y_test = y.iloc[test_idx]

            # Source MCAR missingness on train/val (top-k numeric features only).
            source_rate = cfg.source_missingness["rate"]
            train_mask = generate_mcar(
                X_train_raw, topk, source_rate, np.random.default_rng(_env_seed(seed, "source_train", 0))
            )
            val_mask = generate_mcar(
                X_val_raw, topk, source_rate, np.random.default_rng(_env_seed(seed, "source_val", 1))
            )
            X_train_src = apply_mask(X_train_raw, train_mask)
            X_val_src = apply_mask(X_val_raw, val_mask)

            imputer = MedianModeImputer().fit(X_train_src)
            encoders = fit_categorical_encoders(X_train_src, categorical_cols)

            # Build target environments on the (clean) test copy.
            env_masks: dict[str, pd.DataFrame] = {}
            env_cfgs = cfg.target_environments
            for idx, (env_name, env_cfg) in enumerate(env_cfgs.items(), start=2):
                mech = env_cfg["mechanism"]
                if mech == "mcar":
                    mask = generate_mcar(
                        X_test_raw, topk, float(env_cfg["rate"]),
                        np.random.default_rng(_env_seed(seed, env_name, idx)),
                    )
                elif mech == "feature_dependent":
                    mask = generate_feature_dependent(
                        X_test_raw, topk, driver, float(env_cfg["target_rate"]),
                        float(env_cfg.get("driver_scale", 2.0)),
                        np.random.default_rng(_env_seed(seed, env_name, idx)),
                    )
                elif mech == "class_conditional":
                    mask = generate_class_conditional(
                        X_test_raw, topk, y_test, float(env_cfg["minority_rate"]),
                        float(env_cfg["majority_rate"]),
                        np.random.default_rng(_env_seed(seed, env_name, idx)),
                    )
                else:
                    raise ValueError(f"Unknown mechanism: {mech}")
                env_masks[env_name] = mask

            # Added: MCAR at the class-conditional *measured* overall rate, for a
            # matched-exposure comparator (used by CCEP and GO-B).
            cc_mask = env_masks["class_conditional"]
            cc_meas = measure_missingness(
                X_test_raw, apply_mask(X_test_raw, cc_mask), y_test, cols=topk
            )
            comp_rate = cc_meas["overall_missing_rate"]
            env_masks["mcar_comparable"] = generate_mcar(
                X_test_raw, topk, comp_rate,
                np.random.default_rng(_env_seed(seed, "mcar_comparable", 99)),
            )

            # Record missingness profiles.
            for env_name, mask in env_masks.items():
                masked = apply_mask(X_test_raw, mask)
                meas = measure_missingness(X_test_raw, masked, y_test, cols=topk)
                missing_rows.append(
                    {
                        "dataset": ds_print,
                        "seed": seed,
                        "environment": env_name,
                        "overall_missing_rate": meas["overall_missing_rate"],
                        "minority_missing_rate": meas["minority_missing_rate"],
                        "majority_missing_rate": meas["majority_missing_rate"],
                        "top_features_affected": ",".join(topk),
                    }
                )

            # Missingness profile for source train/val too.
            for split_name, (mask, split_X, split_y) in {
                "source_train": (train_mask, X_train_raw, y_train),
                "source_val": (val_mask, X_val_raw, y_val),
            }.items():
                meas = measure_missingness(
                    split_X, apply_mask(split_X, mask), split_y, cols=topk
                )
                missing_rows.append(
                    {
                        "dataset": ds_print,
                        "seed": seed,
                        "environment": split_name,
                        "overall_missing_rate": meas["overall_missing_rate"],
                        "minority_missing_rate": meas["minority_missing_rate"],
                        "majority_missing_rate": meas["majority_missing_rate"],
                        "top_features_affected": ",".join(topk),
                    }
                )

            for model in cfg.models:
                p1_models = cfg.pipelines.get("p1_models", [])
                for pipeline in ("P0", "P1"):
                    if pipeline == "P1" and model not in p1_models:
                        continue
                    indicator_cols = (
                        topk
                        if pipeline == "P1" and cfg.pipelines.get("indicator_for") == "topk_numeric"
                        else []
                    )
                    X_tr = _build_test_matrix(
                        X_train_src, imputer, encoders, indicator_cols,
                        add_indicator=pipeline == "P1",
                    )
                    X_va = _build_test_matrix(
                        X_val_src, imputer, encoders, indicator_cols,
                        add_indicator=pipeline == "P1",
                    )
                    scaler = None
                    if model == "logistic_regression":
                        scaler = _fit_scaler(X_tr)
                        X_tr = _apply_scaler(X_tr, scaler)
                        X_va = _apply_scaler(X_va, scaler)

                    estimator = make_model(model, seed)
                    estimator.fit(X_tr, y_train)

                    y_val_pred = threshold_05(estimator.predict_proba(X_va)[:, 1])
                    val_metrics = compute_metrics(y_val, estimator.predict_proba(X_va)[:, 1], y_val_pred)

                    # Evaluate every environment (including IID and the comparator).
                    for env_name, mask in env_masks.items():
                        X_test_masked = apply_mask(X_test_raw, mask)
                        X_te = _build_test_matrix(
                            X_test_masked, imputer, encoders, indicator_cols,
                            add_indicator=pipeline == "P1",
                        )
                        if scaler is not None:
                            X_te = _apply_scaler(X_te, scaler)
                        proba = estimator.predict_proba(X_te)[:, 1]
                        pred = threshold_05(proba)
                        metrics = compute_metrics(y_test, proba, pred)
                        row = {
                            "dataset": ds_print,
                            "seed": seed,
                            "model": model,
                            "pipeline": pipeline,
                            "environment": env_name,
                            "val_minority_recall": val_metrics["minority_recall"],
                            **{k: v for k, v in metrics.items() if k not in ("val_minority_recall",)},
                        }
                        raw_rows.append(row)

    raw = pd.DataFrame(raw_rows)
    missing = pd.DataFrame(missing_rows)

    # ---- Vulnerability table ----
    eps = cfg.thresholds["epsilon"]
    for (ds, seed, model, pipeline), grp in raw.groupby(
        ["dataset", "seed", "model", "pipeline"]
    ):
        iid = grp[grp["environment"] == "iid_mcar"].iloc[0]
        for env in ("mild_mcar", "severe_mcar", "mar_like", "class_conditional", "mcar_comparable"):
            shift = grp[grp["environment"] == env]
            if shift.empty:
                continue
            sr = shift.iloc[0]
            miRD_r = relative_degradation(iid["minority_recall"], sr["minority_recall"], eps)
            maRD_r = relative_degradation(iid["majority_recall"], sr["majority_recall"], eps)
            miRD_f = relative_degradation(iid["minority_f1"], sr["minority_f1"], eps)
            maRD_f = relative_degradation(iid["majority_f1"], sr["majority_f1"], eps)
            mrrauprc = retention(sr["AUPRC"], iid["AUPRC"], eps)
            vuln_rows.append(
                {
                    "dataset": ds,
                    "seed": seed,
                    "model": model,
                    "pipeline": pipeline,
                    "environment": env,
                    "MiRD_recall": miRD_r,
                    "MaRD_recall": maRD_r,
                    "MVG_recall": mvg(miRD_r, maRD_r),
                    "MiRD_f1": miRD_f,
                    "MaRD_f1": maRD_f,
                    "MVG_f1": mvg(miRD_f, maRD_f),
                    "MRR_AUPRC": mrrauprc,
                }
            )
    vuln = pd.DataFrame(vuln_rows)

    # ---- Slope table (P0, averaged across seeds per (dataset, model)) ----
    slope_rows = []
    p0 = raw[raw["pipeline"] == "P0"]
    mcar = p0[p0["environment"].isin(MCAR_RATE)]
    for (ds, model), grp in mcar.groupby(["dataset", "model"]):
        rates = []
        min_rec, maj_rec = [], []
        for env, rate in MCAR_RATE.items():
            sub = grp[grp["environment"] == env]
            rates.append(rate)
            min_rec.append(float(sub["minority_recall"].mean()))
            maj_rec.append(float(sub["majority_recall"].mean()))
        ms, _ = fit_slope(rates, min_rec)
        js, _ = fit_slope(rates, maj_rec)
        slope_rows.append(
            {
                "dataset": ds,
                "model": model,
                "minority_slope": ms,
                "majority_slope": js,
                "slope_gap": slope_gap(ms, js),
                "slope_ratio": slope_ratio(ms, js),
            }
        )
    slopes = pd.DataFrame(slope_rows)

    # ---- Persist tables ----
    raw.to_csv(results_dir / "raw_results.csv", index=False)
    missing.to_csv(results_dir / "missingness_profiles.csv", index=False)
    vuln.to_csv(results_dir / "vulnerability_results.csv", index=False)
    slopes.to_csv(results_dir / "slope_results.csv", index=False)

    # ---- Gate decision ----
    evidence = build_evidence(raw, vuln, slopes, cfg)
    go, verdict = decide(evidence, vuln, slopes, raw, cfg)

    decision = {
        "gate": verdict,
        "verdicts": go["verdicts"],
        "go_conditions_hit": go.get("go_conditions_hit", []),
        "dataset_mvg": go.get("dataset_mvg", {}),
        "hold_signals": go.get("hold_signals", []),
        "min_datasets": cfg.gate["min_datasets"],
        "strong_mvg": cfg.gate["strong_mvg"],
    }
    (results_dir / "gate_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    # ---- Figures ----
    figure_paths = generate_all_figures(raw, vuln, slopes, figures_dir)

    # ---- Manifest ----
    manifest = build_manifest(cfg, datasets, raw, missing, slopes)
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    # ---- Report ----
    report = build_report(
        cfg, datasets, raw, missing, vuln, slopes, evidence, decision, manifest, figure_paths
    )
    (results_dir / "gate_report.md").write_text(report, encoding="utf-8")

    print("\nProject E Gate A complete. Verdict:", verdict)
    print("Wrote tables/report/figures under:", results_dir)
    print_decision_summary(decision, evidence, slopes, raw)
    return 0


def build_manifest(cfg, datasets, raw, missing, slopes) -> dict:
    now = datetime.now().isoformat(timespec="seconds")
    import platform

    package_versions = {}
    for mod in ("numpy", "pandas", "sklearn", "xgboost", "matplotlib", "seaborn", "yaml", "ucimlrepo"):
        try:
            m = __import__(mod)
            package_versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # noqa: BLE001
            package_versions[mod] = "n/a"
    processed = resolve(cfg, "processed_data")
    dataset_hashes = {}
    for d in datasets:
        cache = processed / f"{d['name']}.csv"
        if cache.exists():
            dataset_hashes[d["name"]] = hashlib.sha1(cache.read_bytes()).hexdigest()
        else:
            dataset_hashes[d["name"]] = "not-cached"

    try:
        git_commit = (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], cwd=str(PROJECT_ROOT)
            )
            .decode()
            .strip()
        )
    except Exception:  # noqa: BLE001
        git_commit = "n/a"

    return {
        "timestamp": now,
        "python_version": platform.python_version(),
        "package_versions": package_versions,
        "dataset_sources": {
            d["name"]: d["source_info"]
            for d in datasets
        },
        "dataset_hashes": dataset_hashes,
        "seeds": cfg.seeds,
        "missingness_config_hash": hashlib.sha1(
            json.dumps(cfg.target_environments, sort_keys=True).encode()
        ).hexdigest(),
        "config_hash": hashlib.sha1(
            json.dumps({k: v for k, v in cfg.__dict__.items() if k != "project_root"}, sort_keys=True, default=str).encode()
        ).hexdigest(),
        "git_commit": git_commit,
        "n_raw_rows": int(len(raw)),
        "n_missing_profile_rows": int(len(missing)),
        "n_slope_rows": int(len(slopes)),
    }


def build_report(cfg, datasets, raw, missing, vuln, slopes, evidence, decision, manifest, figure_paths) -> str:
    lines: list[str] = []
    lines.append("# Project E Gate A Report")
    lines.append("")
    lines.append(f"**Automatic gate: `{decision['gate']}`**")
    lines.append("")
    lines.append("## 1. Gate verdicts")
    for k, v in decision["verdicts"].items():
        lines.append(f"- {k}: `{'TRUE' if v else 'FALSE'}`")
    lines.append("")
    lines.append("## 2. Dataset profile")
    lines.append("| dataset | n_samples | n_numeric | minority/(majority) | imbalance | natural_missing |")
    lines.append("|---|---|---|---|---|---|")
    for d in datasets:
        x_shape = d["X"].shape
        counts = d["y"].value_counts()
        lines.append(
            f"| {d['name']} | {x_shape[0]} | {len(d['numeric_cols'])} | "
            f"{int(counts.get(1, 0))}/{int(counts.get(0, 0))} | "
            f"{d['imbalance_ratio']:.2f} | {d['natural_missing_count']} |"
        )
    lines.append("")
    lines.append("## 3. Key numbers")
    p0 = raw[raw["pipeline"] == "P0"]
    for env_name in ("severe_mcar", "class_conditional"):
        sub = p0[p0["environment"] == env_name]
        if sub.empty:
            continue
        lines.append(
            f"- Environment `{env_name}`: mean AUPRC = {sub['AUPRC'].mean():.4f}, "
            f"mean minority recall = {sub['minority_recall'].mean():.4f}, "
            f"mean majority recall = {sub['majority_recall'].mean():.4f}."
        )
    if not vuln.empty:
        strong = vuln[vuln["environment"] == "severe_mcar"]
        if not strong.empty:
            lines.append(
                f"- Mean MVG_recall (severe MCAR): {strong['MVG_recall'].mean():.4f}; "
                f"mean MiRD_recall {strong['MiRD_recall'].mean():.4f}, "
                f"MaRD_recall {strong['MaRD_recall'].mean():.4f}."
            )
    if not slopes.empty:
        ms = slopes["minority_slope"].mean()
        js = slopes["majority_slope"].mean()
        lines.append(
            f"- Mean slope: minority {ms:.4f}, majority {js:.4f}; "
            f"slope ratio = {slopes['slope_ratio'].mean():.2f}."
        )
    lines.append("")
    lines.append("## 4. Evidence table")
    if not evidence.empty:
        ev = evidence.round(3).to_string(index=False)
        lines.append("```")
        lines.append(ev)
        lines.append("```")
    lines.append("")
    lines.append("## 5. Figures")
    for p in figure_paths:
        if p:
            lines.append(f"- `{p.name}`")
    lines.append("")
    lines.append("## 6. Artifacts")
    lines.append("- raw_results.csv")
    lines.append("- missingness_profiles.csv")
    lines.append("- vulnerability_results.csv")
    lines.append("- slope_results.csv")
    lines.append("- gate_decision.json")
    lines.append("- manifest.json")
    lines.append("")
    lines.append("## 7. Manifest")
    lines.append("```json")
    lines.append(json.dumps(manifest, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## 8. Methodology notes")
    lines.append("- `P0` = median/mode imputation (source-train imputer)."
                 " `P1` = imputation + missingness indicators (XGBoost only).")
    lines.append("- `strongest_MVG` uses the *worst* shift environment per seed, "
                 "so it is an upper envelope for the class-asymmetric gap.")
    lines.append("- `mcar_comparable` is an auxiliary environment whose missing "
                 "rate matches the class-conditional *measured* overall rate; it is "
                 "used only for CCEP and GO-B.")
    lines.append("- Feature ranking, imputer and (for logistic regression) the scaler "
                 "are fitted on source train only; test labels are never used for "
                 "parameter selection.")
    lines.append("")
    lines.append("## 9. Gate report Q&A")
    qa = answer_questions(cfg, raw, vuln, slopes, evidence, decision)
    lines.extend(qa)
    lines.append("")
    return "\n".join(lines)


def answer_questions(cfg, raw, vuln, slopes, evidence, decision) -> list[str]:
    import numpy as np

    p0 = raw[raw["pipeline"] == "P0"]
    out: list[str] = []

    iid = p0[p0["environment"] == "iid_mcar"]
    severe = p0[p0["environment"] == "severe_mcar"]
    cc = p0[p0["environment"] == "class_conditional"]
    comp = p0[p0["environment"] == "mcar_comparable"]

    out.append(f"1. Missingness shift reduces performance? "
               f"IID mean AUPRC = {iid['AUPRC'].mean():.3f} vs severe = "
               f"{severe['AUPRC'].mean():.3f} (mean AUPRC retention "
               f"{severe['AUPRC'].mean() / (iid['AUPRC'].mean() + 1e-12):.3f}).")

    if not vuln.empty:
        sv = vuln[vuln["environment"] == "severe_mcar"]
        out.append(f"2. Minority drops more? Severe mean MiRD_recall = "
                   f"{sv['MiRD_recall'].mean():.3f}, MaRD_recall = "
                   f"{sv['MaRD_recall'].mean():.3f}, MVG_recall = "
                   f"{sv['MVG_recall'].mean():.3f}.")

    if not vuln.empty:
        rows = []
        for env_name in ("mild_mcar", "severe_mcar", "mar_like", "class_conditional"):
            sub = vuln[vuln["environment"] == env_name]
            rows.append((env_name, sub["MiRD_recall"].mean()))
        rows.sort(key=lambda x: x[1], reverse=True)
        out.append("3. Most dangerous mechanism (mean MiRD_recall): " +
                   "; ".join(f"{n}={float(v):.3f}" for n, v in rows) + ".")

    if not comp.empty and not cc.empty:
        ccep = comp["minority_recall"].mean() - cc["minority_recall"].mean()
        out.append(f"4. Comparable-rate penalty? MCAR-comparable minority recall = "
                   f"{comp['minority_recall'].mean():.3f}, class-conditional = "
                   f"{cc['minority_recall'].mean():.3f}, CCEP = {ccep:.3f}.")

    hidden_cases = evidence[evidence["hidden_failure"]] if not evidence.empty else pd.DataFrame()
    out.append(f"5. Hidden minority failure? "
               f"{'YES' if not hidden_cases.empty else 'NO'} "
               f"({len(hidden_cases)} dataset/model case(s); "
               f"hidden_failure_count total = "
               f"{int(evidence['hidden_failure_count'].sum()) if not evidence.empty else 0}).")

    if not slopes.empty:
        out.append(f"6. Minority slope steeper? mean minority slope = "
                   f"{slopes['minority_slope'].mean():.3f}, majority slope = "
                   f"{slopes['majority_slope'].mean():.3f}, ratio = "
                   f"{slopes['slope_ratio'].mean():.2f}.")

    if not evidence.empty and evidence["indicator_gain_auprc"].notna().any():
        mig_vals = evidence["indicator_gain_auprc"].dropna()
        out.append(f"7. Missing indicator? mean indicator AUPRC gain = "
                   f"{mig_vals.mean():.3f} "
                   f"({'helps' if mig_vals.mean() > 0.005 else 'neutral/hurts'} only if "
                   f"materially above seed noise).")

    if not evidence.empty:
        both = evidence.pivot_table(
            index="dataset", columns="model", values="strongest_MVG"
        )
        if "xgboost" in both and "logistic_regression" in both:
            agree = int(((both["xgboost"] > 0) & (both["logistic_regression"] > 0)).sum())
            out.append(f"8. XGBoost vs LR direction consistent? {agree}/{len(both)} "
                       f"datasets show a positive MVG in both models "
                       f"(mean XGB strongest_MVG = {both['xgboost'].mean():.3f}, "
                       f"LR = {both['logistic_regression'].mean():.3f}).")

    if not evidence.empty:
        cons = evidence["seed_consistency"].mean()
        out.append(f"9. Effect vs seed noise? mean MVG direction consistency = "
                   f"{cons:.2f} (fraction of seeds showing minority harmed more); "
                   f"verdict requires >= {cfg.gate['min_seed_consistency']}, i.e. "
                   f"{cfg.gate['min_seed_consistency']}/{len(cfg.seeds)}.")

    if not evidence.empty:
        best = evidence.groupby("dataset")["strongest_MVG"].mean().idxmax()
        out.append(f"10. Strongest dataset: {best} "
                   f"(mean strongest_MVG = "
                   f"{evidence.groupby('dataset')['strongest_MVG'].mean().max():.3f}).")

    out.append(f"11. Gate = {decision['gate']}. Conditions hit: "
               f"{', '.join(decision['go_conditions_hit']) or 'none'}.")

    # 12. At least five concrete numbers.
    if not slopes.empty:
        ms, js = slopes["minority_slope"].mean(), slopes["majority_slope"].mean()
        mvg_severe = (
            f"{vuln[vuln['environment'] == 'severe_mcar']['MVG_recall'].mean():.3f}"
            if not vuln.empty and not vuln[vuln["environment"] == "severe_mcar"].empty
            else "n/a"
        )
        out.append(f"12. Concrete numbers: severe mean AUPRC {severe['AUPRC'].mean():.3f}; "
                   f"severe mean minority_recall {severe['minority_recall'].mean():.3f}; "
                   f"CC mean minority_recall {cc['minority_recall'].mean():.3f}; "
                   f"mean MVG_recall severe {mvg_severe}; "
                   f"slope ratio {slopes['slope_ratio'].mean():.2f} "
                   f"(minority {ms:.3f} vs majority {js:.3f}).")
    return out


def print_decision_summary(decision, evidence, slopes, raw) -> None:
    print("\n=== Project E Gate A ===")
    print("1. Gate:", decision["gate"])
    print("   conditions hit:", decision["go_conditions_hit"] or "none")
    p0 = raw[raw["pipeline"] == "P0"]
    for env_name in ("severe_mcar", "class_conditional"):
        sub = p0[p0["environment"] == env_name]
        if not sub.empty:
            print(
                f"   {env_name}: AUPRC={sub['AUPRC'].mean():.3f} "
                f"minority_recall={sub['minority_recall'].mean():.3f}"
            )
    if not slopes.empty:
        print(
            f"   slopes: minority={slopes['minority_slope'].mean():.3f} "
            f"majority={slopes['majority_slope'].mean():.3f}"
        )
    if not evidence.empty:
        print("   strongest MVG per dataset:")
        print(evidence.groupby("dataset")["strongest_MVG"].mean().round(3).to_string())
    print("10. Main conclusion: see results/gate_a/gate_report.md")


if __name__ == "__main__":
    raise SystemExit(main())
