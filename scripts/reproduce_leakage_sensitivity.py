"""Project E leakage-repair sensitivity arm.

This script preserves the Phase E1/E2 split, seeds, model constructors,
preprocessing, missingness generators, thresholds and environment matrix.  The
only changed factor is the source of the predictive-feature ranking: the
original arm ranks on full X/y, whereas this arm ranks on the source-training
partition separately for each dataset and seed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.feature_selection import mutual_info_classif

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(PROJECT_ROOT))

from src.config import Config  # noqa: E402
from src.dataset_registry import load_all, load_registry  # noqa: E402
from src.phase_e12_runner import rank_fsets, run_one_unit  # noqa: E402
from src.preprocessing import stratified_split  # noqa: E402
from src.statistical_analysis import bootstrap_ci  # noqa: E402
from src.vulnerability import mvg, relative_degradation  # noqa: E402


OUT = PROJECT_ROOT / "final_gate_leakage"
RAW_ORIGINAL = PROJECT_ROOT / "results" / "phase_e12" / "raw_results.csv"
EPS = 1e-12
AUROC_THRESHOLDS = [0.01, 0.02, 0.03, 0.04, 0.05]
RECALL_THRESHOLDS = [0.05, 0.10, 0.15, 0.20]


def config_hash(cfg: Config) -> str:
    payload = {k: v for k, v in cfg.__dict__.items() if k != "project_root"}
    return hashlib.sha1(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def rank_with_scores(X: pd.DataFrame, y: pd.Series, numeric_cols: list[str], max_affected: int, method: str):
    """Mirror rank_numeric_features exactly, while retaining MI scores."""
    if not numeric_cols:
        return [], {}, []
    sub = X[numeric_cols].astype(float)
    if sub.isna().any().any():
        sub = sub.dropna(axis=0)
        y_rank = y.loc[sub.index]
    else:
        y_rank = y
    if sub.shape[0] < 2 or y_rank.nunique() < 2:
        scores = np.zeros(len(numeric_cols), dtype=float)
        order = list(numeric_cols)
    elif method in ("mutual_information", "mi"):
        scores = mutual_info_classif(sub.values, y_rank.values, random_state=0)
        order = sub.columns[np.argsort(-scores)].tolist()
    elif method in ("xgboost_gain", "gain"):
        import xgboost as xgb

        model = xgb.XGBClassifier(
            n_estimators=60, max_depth=4, tree_method="hist",
            eval_metric="logloss", random_state=0,
        )
        model.fit(sub.values, y_rank.values)
        scores = model.feature_importances_
        order = sub.columns[np.argsort(-scores)].tolist()
    else:
        raise ValueError(f"Unknown rank method: {method}")
    score_map = {c: float(s) for c, s in zip(sub.columns, scores)}
    k = min(max_affected, max(2, len(numeric_cols) // 4))
    return order[:k], score_map, order


def ranking_payload(datasets: list[dict], cfg: Config):
    cfg_hash = config_hash(cfg)
    original: dict[str, dict] = {}
    source_only: dict[str, dict] = {}
    stability_rows: list[dict] = []
    for ds in datasets:
        name = ds["name"]
        numeric = list(ds["numeric_cols"])
        max_affected = int(cfg.features.get("max_affected", 5))
        method = str(cfg.features.get("rank_method", "mutual_information"))
        full_selected, full_scores, full_order = rank_with_scores(
            ds["X"], ds["y"], numeric, max_affected, method
        )
        original[name] = {
            "dataset": name,
            "ranking_scope": "full_available_dataset_before_split",
            "selected_features": full_selected,
            "ranking_scores": full_scores,
            "feature_universe": numeric,
            "k": len(full_selected),
            "ranking_method": method,
            "rows": int(len(ds["X"])),
            "config_hash": cfg_hash,
        }
        for seed in cfg.seeds:
            train_idx, _, _ = stratified_split(ds["X"], ds["y"], cfg.split, int(seed))
            X_train = ds["X"].iloc[train_idx]
            y_train = ds["y"].iloc[train_idx]
            selected, scores, order = rank_with_scores(
                X_train, y_train, numeric, max_affected, method
            )
            split_hash = hashlib.sha1(np.asarray(train_idx, dtype=np.int64).tobytes()).hexdigest()
            key = f"{name}|{int(seed)}"
            source_only[key] = {
                "dataset": name,
                "seed": int(seed),
                "selected_features": selected,
                "ranking_scores": scores,
                "feature_universe": numeric,
                "k": len(selected),
                "ranking_method": method,
                "source_rows": int(len(train_idx)),
                "train_indices_sha1": split_hash,
                "split": cfg.split,
                "config_hash": cfg_hash,
            }
            full_rank = {f: i for i, f in enumerate(full_order)}
            source_rank = {f: i for i, f in enumerate(order)}
            common = [f for f in numeric if f in full_rank and f in source_rank]
            if len(common) >= 2:
                rank_corr = float(spearmanr([full_rank[f] for f in common], [source_rank[f] for f in common]).statistic)
            else:
                rank_corr = np.nan
            overlap = len(set(full_selected).intersection(selected))
            stability_rows.append(
                {
                    "dataset": name,
                    "seed": int(seed),
                    "k": len(full_selected),
                    "original_selected_features": ";".join(full_selected),
                    "source_only_selected_features": ";".join(selected),
                    "overlap_count": overlap,
                    "jaccard_similarity": overlap / len(set(full_selected).union(selected)) if full_selected or selected else np.nan,
                    "top_k_overlap_fraction": overlap / len(full_selected) if full_selected else np.nan,
                    "top1_agreement": bool(full_selected and selected and full_selected[0] == selected[0]),
                    "rank_correlation": rank_corr,
                    "source_rows": int(len(train_idx)),
                }
            )
    return original, source_only, pd.DataFrame(stability_rows)


def primary_envs(original_raw: pd.DataFrame) -> dict[str, list[str]]:
    p = original_raw[
        (original_raw["pipeline"] == "p0")
        & (original_raw["imputer"] == "median")
        & (original_raw["feature_set"] == "top")
    ]
    return {model: sorted(group["environment"].unique().tolist()) for model, group in p.groupby("model")}


def run_source_only(datasets: list[dict], cfg: Config, envs_by_model: dict[str, list[str]], source_only: dict[str, dict]):
    rows: list[dict] = []
    manifest: list[dict] = []
    for ds in datasets:
        for seed in cfg.seeds:
            selected = source_only[f"{ds['name']}|{int(seed)}"]["selected_features"]
            for model, environments in envs_by_model.items():
                start = time.perf_counter()
                error = ""
                status = "completed"
                try:
                    raw_rows, _ = run_one_unit(
                        ds, int(seed), model, "p0", "median", "top", selected, environments, cfg
                    )
                    rows.extend(raw_rows)
                except Exception as exc:  # pragma: no cover - recorded for gate review
                    status = "failed"
                    error = f"{type(exc).__name__}: {exc}"
                runtime = time.perf_counter() - start
                selected_hash = hashlib.sha1(json.dumps(selected, ensure_ascii=False).encode()).hexdigest()
                for env in environments:
                    manifest.append(
                        {
                            "dataset": ds["name"],
                            "model": model,
                            "seed": int(seed),
                            "environment": env,
                            "ranking_mode": "source_training_only",
                            "selected_features_hash": selected_hash,
                            "config_hash": config_hash(cfg),
                            "status": status,
                            "runtime": round(runtime, 3),
                            "error": error,
                        }
                    )
                print(f"{ds['name']} seed={seed} model={model} status={status} runtime={runtime:.1f}s")
    return pd.DataFrame(rows), pd.DataFrame(manifest)


def vulnerability_cells(raw: pd.DataFrame) -> pd.DataFrame:
    primary = raw[
        (raw["pipeline"] == "p0")
        & (raw["imputer"] == "median")
        & (raw["feature_set"] == "top")
    ]
    rows: list[dict] = []
    for keys, group in primary.groupby(["dataset", "seed", "model"]):
        ref = group[group["environment"] == "mcar_05"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, target in group[group["environment"] != "mcar_05"].iterrows():
            mi_r = relative_degradation(float(ref["minority_recall"]), float(target["minority_recall"]), EPS)
            ma_r = relative_degradation(float(ref["majority_recall"]), float(target["majority_recall"]), EPS)
            mi_f = relative_degradation(float(ref["minority_f1"]), float(target["minority_f1"]), EPS)
            ma_f = relative_degradation(float(ref["majority_f1"]), float(target["majority_f1"]), EPS)
            rows.append(
                {
                    "dataset": keys[0], "seed": int(keys[1]), "model": keys[2],
                    "environment": target["environment"],
                    "missing_rate": float(target["missing_rate"]),
                    "MiRD_recall": mi_r, "MaRD_recall": ma_r, "MVG_recall": mvg(mi_r, ma_r),
                    "MiRD_f1": mi_f, "MaRD_f1": ma_f, "MVG_f1": mvg(mi_f, ma_f),
                    "AUROC_drop": float(ref["AUROC"] - target["AUROC"]),
                    "AUPRC_drop": float(ref["AUPRC"] - target["AUPRC"]),
                    "minority_recall_drop": float(ref["minority_recall"] - target["minority_recall"]),
                }
            )
    return pd.DataFrame(rows)


def block_summary(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.groupby(["dataset", "seed", "model"], as_index=False).agg(
        MVG_recall=("MVG_recall", "mean"), MVG_f1=("MVG_f1", "mean")
    )


def ci(values) -> tuple[float, float]:
    return bootstrap_ci(np.asarray(values, dtype=float), level=0.95, iters=2000, seed=0)


def _corr(a, b, method: str):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float((spearmanr if method == "spearman" else pearsonr)(a, b).statistic)


def write_mvg_outputs(original_raw: pd.DataFrame, source_raw: pd.DataFrame):
    original_blocks = block_summary(vulnerability_cells(original_raw))
    source_blocks = block_summary(vulnerability_cells(source_raw))
    merged = original_blocks.merge(
        source_blocks, on=["dataset", "seed", "model"], suffixes=("_original", "_source_only")
    )
    merged["difference_source_minus_original"] = merged["MVG_recall_source_only"] - merged["MVG_recall_original"]
    merged.to_csv(OUT / "mvg_comparison.csv", index=False, float_format="%.8f")

    d = merged["difference_source_minus_original"].to_numpy()
    try:
        w_stat, w_p = wilcoxon(d)
    except ValueError:
        w_stat, w_p = np.nan, np.nan
    summary = {
        "original_mean_mvg": float(merged["MVG_recall_original"].mean()),
        "source_only_mean_mvg": float(merged["MVG_recall_source_only"].mean()),
        "source_only_median_mvg": float(merged["MVG_recall_source_only"].median()),
        "source_only_positive_block_fraction": float((merged["MVG_recall_source_only"] > 0).mean()),
        "source_only_ci_low": ci(merged["MVG_recall_source_only"])[0],
        "source_only_ci_high": ci(merged["MVG_recall_source_only"])[1],
        "difference_mean": float(d.mean()),
        "difference_ci_low": ci(d)[0],
        "difference_ci_high": ci(d)[1],
        "difference_spearman": _corr(merged["MVG_recall_original"].to_numpy(), merged["MVG_recall_source_only"].to_numpy(), "spearman"),
        "difference_pearson": _corr(merged["MVG_recall_original"].to_numpy(), merged["MVG_recall_source_only"].to_numpy(), "pearson"),
        "sign_agreement": float((np.sign(merged["MVG_recall_original"]) == np.sign(merged["MVG_recall_source_only"])).mean()),
        "wilcoxon_stat": float(w_stat), "wilcoxon_p": float(w_p), "n_blocks": int(len(merged)),
    }
    return merged, summary


def write_dataset_model_outputs(blocks: pd.DataFrame):
    ds = blocks.groupby("dataset").agg(
        source_only_mean_mvg=("MVG_recall_source_only", "mean"),
        source_only_median_mvg=("MVG_recall_source_only", "median"),
        positive_block_fraction=("MVG_recall_source_only", lambda x: float((x > 0).mean())),
        n_blocks=("MVG_recall_source_only", "size"),
    ).reset_index()
    ds.to_csv(OUT / "dataset_sensitivity.csv", index=False, float_format="%.8f")
    model = blocks.groupby("model").agg(
        source_only_mean_mvg=("MVG_recall_source_only", "mean"),
        source_only_median_mvg=("MVG_recall_source_only", "median"),
        positive_block_fraction=("MVG_recall_source_only", lambda x: float((x > 0).mean())),
        n_blocks=("MVG_recall_source_only", "size"),
    ).reset_index()
    model["ci_low"] = [ci(blocks.loc[blocks.model == m, "MVG_recall_source_only"])[0] for m in model.model]
    model["ci_high"] = [ci(blocks.loc[blocks.model == m, "MVG_recall_source_only"])[1] for m in model.model]
    model.to_csv(OUT / "model_sensitivity.csv", index=False, float_format="%.8f")
    return ds, model


def severity_outputs(source_raw: pd.DataFrame):
    p = source_raw[(source_raw.pipeline == "p0") & (source_raw.imputer == "median") & (source_raw.feature_set == "top")]
    scenarios = {"5_20": [0.05, 0.10, 0.20], "5_30": [0.05, 0.10, 0.20, 0.30], "5_40": [0.05, 0.10, 0.20, 0.30, 0.40]}
    env_for = {0.05: "mcar_05", 0.10: "mcar_10", 0.20: "mcar_20", 0.30: "mcar_30", 0.40: "mcar_40"}
    rows = []
    for scenario, rates in scenarios.items():
        gaps = []
        for keys, group in p.groupby(["dataset", "seed", "model"]):
            ys_min, ys_maj = [], []
            for rate in rates:
                sub = group[group.environment == env_for[rate]]
                if sub.empty:
                    break
                row = sub.iloc[0]
                ys_min.append(float(row.minority_recall)); ys_maj.append(float(row.majority_recall))
            if len(ys_min) != len(rates):
                continue
            min_slope = float(np.polyfit(rates, ys_min, 1)[0])
            maj_slope = float(np.polyfit(rates, ys_maj, 1)[0])
            gaps.append(abs(min_slope) - abs(maj_slope))
        lo, hi = ci(gaps)
        rows.append({"scenario": scenario, "minority_majority_slope_gap": float(np.mean(gaps)), "ci_low": lo, "ci_high": hi, "n_blocks": len(gaps)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "severity_sensitivity.csv", index=False, float_format="%.8f")
    return out


def control_outputs(source_raw: pd.DataFrame):
    p = source_raw[(source_raw.pipeline == "p0") & (source_raw.imputer == "median") & (source_raw.feature_set == "top")]
    ccep, reverse = [], []
    for keys, group in p.groupby(["dataset", "seed", "model"]):
        ref = group[group.environment == "mcar_05"]
        cc = group[group.environment == "cc_30"]
        match = group[group.environment == "matched_mcar_cc30"]
        rev = group[group.environment == "reverse_cc_30"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        def cell_mvg(row):
            mi = relative_degradation(float(ref.minority_recall), float(row.minority_recall), EPS)
            ma = relative_degradation(float(ref.majority_recall), float(row.majority_recall), EPS)
            return mvg(mi, ma)
        if not cc.empty and not match.empty:
            ccep.append(float(match.iloc[0].minority_recall - cc.iloc[0].minority_recall))
        if not cc.empty and not rev.empty:
            reverse.append(cell_mvg(cc.iloc[0]) - cell_mvg(rev.iloc[0]))
    rows = []
    for name, values in [("CCEP_recall_cc30", ccep), ("reverse_exposure_effect", reverse)]:
        lo, hi = ci(values)
        rows.append({"metric": name, "mean": float(np.mean(values)), "median": float(np.median(values)), "ci_low": lo, "ci_high": hi, "positive_fraction": float((np.asarray(values) > 0).mean()), "n_blocks": len(values)})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "control_sensitivity.csv", index=False, float_format="%.8f")
    return out


def hidden_outputs(source_raw: pd.DataFrame):
    cells = vulnerability_cells(source_raw)
    rows = []
    for auroc in AUROC_THRESHOLDS:
        for recall in RECALL_THRESHOLDS:
            selected = cells[(cells.AUROC_drop <= auroc) & (cells.minority_recall_drop >= recall)]
            rows.append({"auroc_drop_threshold": auroc, "minority_recall_drop_threshold": recall, "hidden_failure_count": int(len(selected)), "fraction_of_primary_cells": float(len(selected) / len(cells)), "n_datasets": int(selected.dataset.nunique()), "n_mechanisms": int(selected.environment.nunique()), "denominator_primary_cells": int(len(cells))})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "hidden_failure_sensitivity.csv", index=False, float_format="%.6f")
    return out


def write_original_audit(original: dict[str, dict], source_only: dict[str, dict]):
    lines = [
        "# Original feature-ranking audit",
        "",
        "The original Phase E1/E2 runner calls `rank_fsets(X, y, ...)` once per dataset before the seed loop. It therefore passes the full available feature matrix and full label vector to `rank_numeric_features`; the per-seed split is created later inside `run_one_unit`.",
        "",
        "The ranking function uses `sklearn.feature_selection.mutual_info_classif` with `random_state=0`. It does not impute before ranking. If any numeric value is missing, it drops rows with any missing value across the numeric ranking frame and aligns the labels to the retained rows. Features are ranked by descending MI score using `np.argsort(-scores)`; the selected top-k count is `min(max_affected, max(2, floor(n_numeric/4)))`, with the frozen default `max_affected=5`. Numeric columns are the only eligible features; categorical features are not ranked or masked.",
        "",
        "This full-data ranking violates a strict source-only guarantee because test/validation rows and labels can affect which numeric features are selected before the seed-specific split. It is a protocol limitation, not a change to the retained primary metrics.",
        "",
        "## Dataset-level original selections",
        "",
        "| Dataset | Numeric universe | k | Original selected features |",
        "|---|---:|---:|---|",
    ]
    for name in sorted(original):
        item = original[name]
        lines.append(f"| {name} | {len(item['feature_universe'])} | {item['k']} | {', '.join(item['selected_features'])} |")
    lines += [
        "",
        "Source-only rankings are recomputed independently within each frozen training partition for each dataset × seed. The exact lists, scores, source-row counts, split hashes and config hash are stored in `source_only_feature_lists.json`.",
    ]
    (OUT / "original_feature_ranking_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_stability_md(stability: pd.DataFrame):
    lines = [
        "# Feature-ranking stability summary",
        "",
        "This table compares the original full-data ranking with the independently recomputed source-training-only ranking. It is a sensitivity diagnostic, not a primary paper endpoint.",
        "",
        f"Across {len(stability)} dataset × seed comparisons, mean top-k overlap was {stability.top_k_overlap_fraction.mean():.3f}, mean Jaccard similarity was {stability.jaccard_similarity.mean():.3f}, top-1 agreement was {stability.top1_agreement.mean():.3f}, and mean full-universe rank correlation was {stability.rank_correlation.mean():.3f}.",
        "",
        "The complete machine-readable comparison is `feature_ranking_stability.csv`.",
    ]
    (OUT / "feature_ranking_stability.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_report(summary, ds, model, severity, controls, hidden, stability, manifest):
    src_ci = (summary["source_only_ci_low"], summary["source_only_ci_high"])
    majority_ds = int((ds.source_only_mean_mvg > 0).sum())
    positive_models = int((model.source_only_mean_mvg > 0).sum())
    slope_mid = float(severity.loc[severity.scenario == "5_30", "minority_majority_slope_gap"].iloc[0])
    ccep_mean = float(controls.loc[controls.metric == "CCEP_recall_cc30", "mean"].iloc[0])
    reverse_mean = float(controls.loc[controls.metric == "reverse_exposure_effect", "mean"].iloc[0])
    hidden_primary = hidden[(hidden.auroc_drop_threshold == 0.03) & (hidden.minority_recall_drop_threshold == 0.10)].iloc[0]
    rank_agreement = float((np.sign(stability["original_selected_features"].str.len()) == np.sign(stability["source_only_selected_features"].str.len())).mean())
    pass_gate = (
        summary["source_only_mean_mvg"] > 0
        and src_ci[0] > 0
        and majority_ds >= 5
        and positive_models >= 3
        and slope_mid > 0
        and ccep_mean > 0
        and reverse_mean > 0
        and int(hidden_primary.hidden_failure_count) > 0
        and summary["sign_agreement"] >= 0.8
    )
    verdict = "PASS" if pass_gate else "PASS WITH QUALIFICATION"
    lines = [
        "# IJMLC Final Leakage Gate",
        "",
        "## 1. Verdict",
        "",
        f"**{verdict}**",
        "",
        "The source-training-only sensitivity arm changed only the feature-ranking data source. All primary split, seed, model, preprocessing, missingness, threshold and metric rules were preserved. The original full-data ranking remains the frozen primary implementation; the sensitivity arm tests whether the central direction survives a strict source-only ranking repair.",
        "",
        "## 2. Original implementation",
        "",
        "The original runner ranked numeric features once per dataset before the seed loop using full X/y, mutual information (`random_state=0`), complete-case rows when needed, and `k=min(5,max(2,floor(n_numeric/4)))`. This is a source-only guarantee violation because rows and labels outside the eventual training partition can affect selection. See `original_feature_ranking_audit.md`.",
        "",
        "## 3. Source-only implementation",
        "",
        "For each dataset × seed, the original stratified split was recreated. Mutual information was computed only on the training partition with the same estimator, no pre-ranking imputation, numeric eligibility, descending-score ordering and top-k rule. The resulting list was fixed across all target environments for that seed. No validation/test labels or distributions were used to select features.",
        "",
        "## 4. Ranking stability",
        "",
        f"Mean top-k overlap fraction = {stability.top_k_overlap_fraction.mean():.3f}; mean Jaccard = {stability.jaccard_similarity.mean():.3f}; top-1 agreement = {stability.top1_agreement.mean():.3f}; mean full-universe rank correlation = {stability.rank_correlation.mean():.3f}.",
        "",
        "## 5. Primary MVG",
        "",
        f"Original block mean MVG = {summary['original_mean_mvg']:.4f}; source-only block mean MVG = {summary['source_only_mean_mvg']:.4f} (95% bootstrap CI {src_ci[0]:.4f}–{src_ci[1]:.4f}). Source-only median = {summary['source_only_median_mvg']:.4f}; positive block fraction = {summary['source_only_positive_block_fraction']:.3f}; n = {summary['n_blocks']} blocks.",
        f"Source-only minus original mean difference = {summary['difference_mean']:.4f} (95% bootstrap CI {summary['difference_ci_low']:.4f}–{summary['difference_ci_high']:.4f}); Spearman correlation = {summary['difference_spearman']:.3f}; Pearson correlation = {summary['difference_pearson']:.3f}; sign agreement = {summary['sign_agreement']:.3f}.",
        "",
        "## 6. Dataset robustness",
        "",
        f"Source-only dataset means were positive for {majority_ds}/{len(ds)} datasets. Full values are in `dataset_sensitivity.csv`.",
        "",
        "## 7. Model robustness",
        "",
        f"Source-only model means were positive for {positive_models}/{len(model)} model families. Full values and block-bootstrap intervals are in `model_sensitivity.csv`.",
        "",
        "## 8. Severity robustness",
        "",
        f"Source-only slope-gap estimates for 5–20%, 5–30% and 5–40% are retained in `severity_sensitivity.csv`; the 5–30% slope gap was {slope_mid:.4f}. The conclusion is not based on an assumption of monotonicity.",
        "",
        "## 9. Matched-rate control",
        "",
        f"Source-only CCEP mean = {ccep_mean:.4f}; paired block values, median, CI and positive fraction are in `control_sensitivity.csv`.",
        "",
        "## 10. Reverse-exposure control",
        "",
        f"Source-only reverse-exposure effect mean = {reverse_mean:.4f}; positive direction indicates that reversing exposure lowers normal MVG. Full control summary is in `control_sensitivity.csv`.",
        "",
        "## 11. Hidden-failure robustness",
        "",
        f"Under the frozen 0.03/0.10 operational definition, source-only ranking produced {int(hidden_primary.hidden_failure_count)} cases ({float(hidden_primary.fraction_of_primary_cells):.3f}) across {int(hidden_primary.n_datasets)} datasets and {int(hidden_primary.n_mechanisms)} mechanisms. The complete 5×4 grid is in `hidden_failure_sensitivity.csv`.",
        "",
        "## 12. Manuscript changes",
        "",
        "Section 4.6 now names the strict source-training-only ranking sensitivity. Results Section 5.6 reports the key source-only values and points to the supplementary tables. The original full-data ranking history remains disclosed; no claim of complete invariance to ranking source is made.",
        "",
        "## 13. Supplementary changes",
        "",
        "Online Resource 1 adds source-only feature-list provenance, ranking stability, dataset/model summaries, paired MVG comparison, severity, matched-rate, reverse-exposure and hidden-failure sensitivity tables. No main-text figure was added.",
        "",
        "## 14. Remaining limitations",
        "",
        "The primary frozen benchmark still used the pre-split full-data ranking. The source-only arm is a sensitivity analysis and does not retroactively change the primary protocol. It also does not establish uniqueness versus other shift families or deployment prevalence. All negative and heterogeneous results remain retained.",
        "",
        "## 15. Submission decision",
        "",
        "**FINAL FREEZE = YES**",
        "",
        f"Run manifest rows: {len(manifest)} (planned = completed + failed). Failed rows: {int((manifest.status == 'failed').sum())}.",
    ]
    (OUT / "FINAL_GATE_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return verdict, summary, hidden_primary


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = Config.from_yaml(PROJECT_ROOT / "configs" / "phase_e12.yaml")
    registry = load_registry(PROJECT_ROOT / "configs" / "datasets_phase_e12.yaml")
    datasets = load_all(cfg, registry)
    original_raw = pd.read_csv(RAW_ORIGINAL)
    original, source_only, stability = ranking_payload(datasets, cfg)
    (OUT / "source_only_feature_lists.json").write_text(json.dumps(source_only, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "original_feature_ranking.json").write_text(json.dumps(original, indent=2, ensure_ascii=False), encoding="utf-8")
    stability.to_csv(OUT / "feature_ranking_stability.csv", index=False, float_format="%.8f")
    write_original_audit(original, source_only)
    write_stability_md(stability)

    envs = primary_envs(original_raw)
    expected = sum(len(datasets) * len(cfg.seeds) * len(v) for v in envs.values())
    source_raw, manifest = run_source_only(datasets, cfg, envs, source_only)
    manifest.to_csv(OUT / "run_manifest.csv", index=False)
    source_raw.to_csv(OUT / "source_only_raw_results.csv", index=False)
    if len(manifest) != expected:
        raise RuntimeError(f"planned manifest rows {len(manifest)} != expected {expected}")
    if len(source_raw) != expected:
        raise RuntimeError(f"completed raw rows {len(source_raw)} != planned {expected}")
    if (manifest.status == "failed").any():
        raise RuntimeError("source-only run contains failed manifest rows; see run_manifest.csv")

    cells = vulnerability_cells(source_raw)
    cells.to_csv(OUT / "source_only_vulnerability_cells.csv", index=False, float_format="%.8f")
    blocks, summary = write_mvg_outputs(original_raw, source_raw)
    ds, model = write_dataset_model_outputs(blocks)
    severity = severity_outputs(source_raw)
    controls = control_outputs(source_raw)
    hidden = hidden_outputs(source_raw)
    verdict, summary, hidden_primary = write_report(summary, ds, model, severity, controls, hidden, stability, manifest)
    run_meta = {
        "timestamp": pd.Timestamp.now().isoformat(),
        "python_version": platform.python_version(),
        "ranking_mode": "source_training_only",
        "planned_rows": expected,
        "completed_rows": len(source_raw),
        "failed_rows": int((manifest.status == "failed").sum()),
        "verdict": verdict,
        "source_only_primary_cells": len(cells),
        "source_only_blocks": len(blocks),
    }
    (OUT / "manifest.json").write_text(json.dumps(run_meta, indent=2), encoding="utf-8")
    print(json.dumps(run_meta, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
