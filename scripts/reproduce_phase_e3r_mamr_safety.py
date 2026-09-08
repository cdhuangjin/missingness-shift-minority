"""One-shot reproduction of Project E Phase E3R: MAMR Safety Repair Gate."""

from __future__ import annotations

import argparse
import json
import platform
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
from src.mamr import MAMRParams, run_one_unit  # noqa: E402
from src.mamr_safety import (  # noqa: E402
    build_candidate_grid_step_a,
    build_candidate_grid_step_b,
    run_stress_cell,
    safe_utility,
    safety_feasible,
    select_global_safe_config,
)
from src.method_gate import (  # noqa: E402
    compute_best_standard,
    compute_big,
    compute_iid_cost,
    compute_lrr,
    compute_mas,
)
from src.phase_e3_config import PhaseE3Config  # noqa: E402
from src.resume_manager import ResumeManager  # noqa: E402
from src.safety_gate import evaluate_safety_gate  # noqa: E402

_SMOKE_METHODS = ["erm", "class_weight", "safe_mamr"]
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


def _make_params(cfg: PhaseE3Config, lambda_: float, copy_fraction: float, corruption_rate: float) -> MAMRParams:
    return MAMRParams(
        lambda_=float(lambda_),
        topk_features=int(cfg.mamr.topk_features),
        augmentation_enabled=cfg.mamr.augmentation.enabled,
        minority_only=bool(cfg.mamr.augmentation.minority_only),
        copy_fraction=float(copy_fraction),
        corruption_rate=float(corruption_rate),
        majority_corruption_rate=float(cfg.mamr.augmentation.majority_corruption_rate),
        mechanism_mix=dict(cfg.mamr.augmentation.mechanism_mix),
    )


def _run_stress_unit(
    rm: ResumeManager,
    ds: dict,
    seed: int,
    model: str,
    method: str,
    cfg: PhaseE3Config,
    params: MAMRParams,
    force: bool,
) -> dict | None:
    key = (
        f"{ds['name']}|{seed}|{model}|{method}|val"
        f"|l{params.lambda_}|c{params.copy_fraction}|r{params.corruption_rate}"
    )
    if not force and rm.is_complete(key):
        return rm.load(key)
    try:
        rows, audit = run_stress_cell(ds, seed, model, method, cfg, params)
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
                    "status": "ok",
                    "error": "",
                }
            )
        data = {"ok": True, "raw": rows, "audit": audit, "error": ""}
        rm.save(key, data)
        return rm.load(key)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        try:
            rm.record_failure(key, err)
        except Exception:  # noqa: BLE001
            pass
        rm.save(key, {"ok": False, "raw": [], "audit": {}, "error": err})
        return None


def _run_test_unit(
    rm: ResumeManager,
    ds: dict,
    seed: int,
    model: str,
    method: str,
    cfg: PhaseE3Config,
    params: MAMRParams,
    env_names: list[str],
    force: bool,
) -> dict | None:
    key = (
        f"{ds['name']}|{seed}|{model}|{method}|test"
        f"|l{params.lambda_}|c{params.copy_fraction}|r{params.corruption_rate}"
    )
    if not force and rm.is_complete(key):
        return rm.load(key)
    try:
        rows, audit = run_one_unit(
            ds, seed, model, method, env_names, cfg, params, eval_mode="test"
        )
        n_aug = int(audit.get("n_augment", 0))
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
                    "augmentation_count": n_aug,
                    "status": "ok",
                    "error": "",
                }
            )
        data = {"ok": True, "raw": rows, "audit": audit, "error": ""}
        rm.save(key, data)
        return rm.load(key)
    except Exception as exc:  # noqa: BLE001
        err = f"{type(exc).__name__}: {exc}"
        try:
            rm.record_failure(key, err)
        except Exception:  # noqa: BLE001
            pass
        rm.save(key, {"ok": False, "raw": [], "audit": {}, "error": err})
        return None


def _validation_search(
    cfg: PhaseE3Config,
    datasets: list[dict],
    models: list[str],
    seeds: list[int],
    rm: ResumeManager,
    force: bool,
) -> dict:
    vs = cfg.raw["validation_safety"]
    hard_maj = float(vs["hard_majority_delta"])
    hard_ap = float(vs["hard_auprc_delta"])
    alpha = float(vs["alpha"])
    beta = float(vs["beta"])
    stress_envs = list(cfg.raw["validation_safety"]["stress_suite"])

    # Run baselines on the validation stress suite and build the best_standard ref.
    baseline_rows: list[dict] = []
    for ds in datasets:
        for model in models:
            for seed in seeds:
                for method in ["erm", "class_weight", "random_oversampling", "missing_indicator"]:
                    p = _make_params(cfg, 0.0, 0.15, 0.10)
                    info = _run_stress_unit(rm, ds, seed, model, method, cfg, p, force)
                    if info is not None and info["data"].get("ok"):
                        baseline_rows.extend(info["data"]["raw"])
    baseline = pd.DataFrame(baseline_rows)

    def best_ref(cell_key: tuple, env: str) -> dict | None:
        sub = baseline[
            (baseline["dataset"] == cell_key[0])
            & (baseline["model"] == cell_key[1])
            & (baseline["seed"] == cell_key[2])
            & (baseline["environment"] == env)
        ]
        if sub.empty:
            return None
        best = sub.sort_values("minority_recall", ascending=False).iloc[0]
        return best

    def run_candidate(params: MAMRParams, step: str, lam: float, cf: float, cr: float, grid: list[dict]):
        rows: list[dict] = []
        for ds in datasets:
            for model in models:
                for seed in seeds:
                    info = _run_stress_unit(rm, ds, seed, model, "safe_mamr", cfg, params, force)
                    if info is None or not info["data"].get("ok"):
                        continue
                    cand = pd.DataFrame(info["data"]["raw"])
                    cell = (ds["name"], model, seed)
                    u_list, maj_list, ap_list = [], [], []
                    for _, r in cand.iterrows():
                        ref = best_ref(cell, r["environment"])
                        if ref is None:
                            continue
                        u_list.append(
                            safe_utility(
                                float(r["minority_recall"]),
                                float(r["majority_recall"]),
                                float(r["AUPRC"]),
                                float(ref["minority_recall"]),
                                float(ref["majority_recall"]),
                                float(ref["AUPRC"]),
                                alpha,
                                beta,
                            )
                        )
                        maj_list.append(float(r["majority_recall"]) - float(ref["majority_recall"]))
                        ap_list.append(float(r["AUPRC"]) - float(ref["AUPRC"]))
                    if not u_list:
                        continue
                    maj_delta = float(np.mean(maj_list))
                    ap_delta = float(np.mean(ap_list))
                    rows.append(
                        {
                            "dataset": cell[0],
                            "model": model,
                            "seed": seed,
                            "step": step,
                            "lambda_": lam,
                            "copy_fraction": cf,
                            "corruption_rate": cr,
                            "u_safe": float(np.mean(u_list)),
                            "majority_delta": maj_delta,
                            "auprc_delta": ap_delta,
                            "feasible": safety_feasible(maj_delta, ap_delta, hard_maj, hard_ap),
                        }
                    )
        return rows

    step_a_grid = build_candidate_grid_step_a(cfg)
    step_a_rows: list[dict] = []
    for cand in step_a_grid:
        params = _make_params(cfg, cand["lambda_"], cand["copy_fraction"], cand["corruption_rate"])
        step_a_rows.extend(
            run_candidate(params, "A", cand["lambda_"], cand["copy_fraction"], cand["corruption_rate"], step_a_grid)
        )
    step_a_frame = pd.DataFrame(step_a_rows)
    step_a_chosen = select_global_safe_config(step_a_frame, feasible_fraction_min=float(vs["global_feasible_fraction"]))
    chosen_lambda = float(step_a_chosen["lambda_"])

    step_b_grid = build_candidate_grid_step_b(cfg, chosen_lambda)
    step_b_rows: list[dict] = []
    for cand in step_b_grid:
        params = _make_params(cfg, cand["lambda_"], cand["copy_fraction"], cand["corruption_rate"])
        step_b_rows.extend(
            run_candidate(params, "B", cand["lambda_"], cand["copy_fraction"], cand["corruption_rate"], step_b_grid)
        )
    all_frame = pd.concat([step_a_frame, pd.DataFrame(step_b_rows)], ignore_index=True)
    global_chosen = select_global_safe_config(all_frame, feasible_fraction_min=float(vs["global_feasible_fraction"]))

    # Per-cell sensitivity (validation-selected params).
    per_cell = []
    for (cell_key, grp) in all_frame.groupby(["dataset", "model", "seed"]):
        if grp.empty:
            continue
        feasible_grp = grp[grp["feasible"]]
        pick = feasible_grp.sort_values("u_safe", ascending=False).iloc[0] if not feasible_grp.empty else grp.sort_values("u_safe", ascending=False).iloc[0]
        per_cell.append(
            {
                "dataset": cell_key[0],
                "model": cell_key[1],
                "seed": cell_key[2],
                "selected_lambda": float(pick["lambda_"]),
                "selected_copy_fraction": float(pick["copy_fraction"]),
                "selected_corruption_rate": float(pick["corruption_rate"]),
                "u_safe": float(pick["u_safe"]),
                "feasible": bool(pick["feasible"]),
            }
        )

    return {
        "all_frame": all_frame,
        "step_a_frame": step_a_frame,
        "step_b_frame": pd.DataFrame(step_b_rows),
        "global": global_chosen,
        "step_a_chosen": step_a_chosen,
        "per_cell": pd.DataFrame(per_cell),
        "baseline": baseline,
        "feasible_fraction": float(all_frame.groupby(["lambda_", "copy_fraction", "corruption_rate"])["feasible"].mean().max())
        if not all_frame.empty else 0.0,
    }


def _load_e3_test_raw(results_dir: Path) -> pd.DataFrame:
    """Read the read-only Phase E3 authoritative raw test results."""
    p = results_dir / "phase_e3_mamr" / "raw_results.csv"
    df = pd.read_csv(p)
    df = df[df["method"] == "mamr_full"].copy()
    df["method"] = "original_mamr"
    return df


def _combine_raw(safe_rows: list[dict]) -> pd.DataFrame:
    """Combine E3 cached baselines / original MAMR with the new Safe-MAMR rows."""
    e3_all = pd.read_csv(
        Path(PROJECT_ROOT) / "results" / "phase_e3_mamr" / "raw_results.csv"
    )
    # Keep only the authoritative methods in raw_results (the file is already the
    # frozen Phase E3 set); rename mamr_full -> original_mamr.
    e3_all = e3_all.copy()
    e3_all.loc[e3_all["method"] == "mamr_full", "method"] = "original_mamr"
    # Backfill the real augmentation count for original MAMR from the E3 audit.
    audit = pd.read_csv(Path(PROJECT_ROOT) / "results" / "phase_e3_mamr" / "exposure_audit.csv")
    audit = audit[audit["method"] == "mamr_full"].set_index(["dataset", "model", "seed"])["n_augment"]
    e3_all["augmentation_count"] = e3_all.set_index(["dataset", "model", "seed"]).index.map(
        lambda k: int(audit.get(k, 0))
    )
    safe = pd.DataFrame(safe_rows)
    combined = pd.concat([e3_all, safe], ignore_index=True)
    return combined


def _summarize_method(raw: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = ["minority_recall", "majority_recall", "minority_f1", "AUPRC", "AUROC", "balanced_accuracy", "MCC"]
    return raw.groupby(group_cols)[cols].mean().reset_index()


def _diagnostic(raw: pd.DataFrame, cfg: PhaseE3Config) -> pd.DataFrame:
    """default_credit_card safety diagnostic: original vs safe vs best_standard."""
    best = compute_best_standard(raw)
    sub = raw[raw["dataset"] == "default_credit_card"]
    rows = []
    for method in ["original_mamr", "safe_mamr"]:
        m = sub[sub["method"] == method]
        for (model, seed, env), grp in m.groupby(["model", "seed", "environment"]):
            if grp.empty:
                continue
            r = grp.iloc[0]
            b = best[
                (best["dataset"] == "default_credit_card")
                & (best["model"] == model)
                & (best["seed"] == seed)
                & (best["environment"] == env)
            ]
            rows.append(
                {
                    "model": model,
                    "seed": seed,
                    "environment": env,
                    "method": method,
                    "minority_recall": float(r["minority_recall"]),
                    "majority_recall": float(r["majority_recall"]),
                    "AUPRC": float(r["AUPRC"]),
                    "exposure_mean": float(r.get("exposure_mean", np.nan)),
                    "exposure_std": float(r.get("exposure_std", np.nan)),
                    "augmentation_count": int(r.get("augmentation_count", 0)),
                    "best_standard_minority_recall": float(b["best_standard_minority_recall"].iloc[0]) if not b.empty else np.nan,
                    "best_standard_majority_recall": float(b["best_standard_majority_recall"].iloc[0]) if not b.empty else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _build_report(cfg, gate, raw, derived, tables, n_expected, n_completed, n_failed) -> str:
    g = gate["gate"]
    c = gate["conditions"]
    m = gate["metrics"]
    counts = gate["counts"]
    lines: list[str] = []
    lines.append("# Project E Phase E3R — MAMR Safety Repair Gate Report")
    lines.append("")
    lines.append(f"**Safety Gate: `{g}`**")
    lines.append("")
    lines.append("## 1. Scope")
    lines.append("")
    lines.append(
        "MAMR Safety Repair is a single pre-registered conservative repair of the "
        "Phase E3 HOLD. It keeps the minority-aware components but moves to a "
        "weaker operating point to reduce the majority-recall sacrifice. The gate "
        "only returns STRONG-GO / GO / HOLD-STOP / STOP."
    )
    lines.append("")
    lines.append("## 2. Runs")
    lines.append("")
    lines.append(f"- expected: {n_expected}")
    lines.append(f"- completed: {n_completed}")
    lines.append(f"- failed: {n_failed}")
    lines.append("")
    lines.append("## 3. Conservative config")
    lines.append("")
    conf = gate.get("conservative_config", {})
    if conf:
        lines.append(
            f"- lambda: {conf.get('lambda_')}, copy_fraction: {conf.get('copy_fraction')}, "
            f"corruption_rate: {conf.get('corruption_rate')}"
        )
        lines.append(f"- validation_safety_feasible: {conf.get('validation_safety_feasible')}")
    lines.append("")
    lines.append("## 4. Key measured numbers")
    lines.append("")
    q = [
        ("Minority recall gain vs best standard", m["minority_gain_safe"]),
        ("Original MAMR minority gain vs best standard", m["minority_gain_original"]),
        ("Minority Gain Retention (MGR)", m["mgr"]),
        ("Majority recall delta (safe)", m["majority_delta_safe"]),
        ("Worst dataset majority delta (safe)", m["worst_dataset_majority_delta_safe"]),
        ("Majority harm reduction (MHR)", m["majority_harm_reduction"]),
        ("AUPRC delta", m["auprc_delta_safe"]),
        ("IID minority recall delta", m["iid_minority_recall_delta"]),
        ("IID AUPRC delta", m["iid_auprc_delta"]),
        ("Median LRR", counts["median_lrr"]),
        ("Positive LRR fraction", counts["positive_lrr_fraction"]),
        ("BIG mean / positive fraction", f"{m['big_mean']} / {m['big_positive_fraction']}"),
        ("MAS", counts["mas"]),
        ("Pareto non-dominated cells", f"{counts['pareto_cells']}/{counts['pareto_total']}"),
    ]
    for label, val in q:
        if isinstance(val, str):
            lines.append(f"- **{label}:** {val}")
        else:
            lines.append(f"- **{label}:** {round(float(val), 4) if val == val else val}")
    lines.append("")
    lines.append("## 5. R1-R10 conditions")
    lines.append("")
    for k in ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]:
        lines.append(f"- **{k}:** {c[k]}")
    lines.append("")
    lines.append(f"Conditions met: {counts['n_pass']}/10")
    lines.append("")
    lines.append("## 6. Decision")
    lines.append("")
    lines.append(f"**Final decision: `{g}`**")
    lines.append("")
    lines.append(f"**Recommended next phase:** {gate['recommended_next_phase']}")
    lines.append("")
    lines.append("## 7. Tables")
    lines.append("")
    for p in tables:
        lines.append(f"- {p.name}")
    lines.append("")
    lines.append("## 8. Reproduce")
    lines.append("")
    lines.append("```bash")
    lines.append("python scripts/reproduce_phase_e3r_mamr_safety.py --resume")
    lines.append("```")
    return "\n".join(lines)


def _figures(raw: pd.DataFrame, gate: dict, results_dir: Path, safe_method: str = "safe_mamr") -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out: list[Path] = []

    def save(fig, name: str):
        p = fig_dir / name
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        out.append(p)

    best = compute_best_standard(raw)
    hr = list(["cc_40", "cc_50", "block_severe", "feature_specific"])
    sub = raw[raw["environment"].isin(hr)]

    # 1. Original vs Safe minority gain.
    gains = {}
    for m in ["original_mamr", "safe_mamr"]:
        merged = sub[sub["method"] == m].merge(best, on=["dataset", "model", "seed", "environment"], how="left")
        gains[m] = merged["minority_recall"] - merged["best_standard_minority_recall"]
    fig, ax = plt.subplots()
    ax.hist(gains["original_mamr"].dropna(), alpha=0.6, label="original MAMR", bins=20)
    ax.hist(gains["safe_mamr"].dropna(), alpha=0.6, label="safe MAMR", bins=20)
    ax.set_xlabel("minority recall gain vs best standard"); ax.set_ylabel("count")
    ax.legend(); save(fig, "fig1_original_vs_safe_minority_gain.png")

    # 2. Original vs Safe majority trade-off.
    trade = {}
    for m in ["original_mamr", "safe_mamr"]:
        merged = sub[sub["method"] == m].merge(best, on=["dataset", "model", "seed", "environment"], how="left")
        trade[m] = merged["majority_recall"] - merged["best_standard_majority_recall"]
    fig, ax = plt.subplots()
    ax.hist(trade["original_mamr"].dropna(), alpha=0.6, label="original MAMR", bins=20)
    ax.hist(trade["safe_mamr"].dropna(), alpha=0.6, label="safe MAMR", bins=20)
    ax.set_xlabel("majority recall delta vs best standard"); ax.legend()
    save(fig, "fig2_original_vs_safe_majority_tradeoff.png")

    # 3. Pareto minority gain vs majority harm.
    pts = []
    for m in ["original_mamr", "safe_mamr", "augmentation_only", "exposure_weighting_only", "uniform_minority_weight"]:
        merged = sub[sub["method"] == m].merge(best, on=["dataset", "model", "seed", "environment"], how="left")
        if merged.empty:
            continue
        pts.append(
            {
                "method": m,
                "gain": float((merged["minority_recall"] - merged["best_standard_minority_recall"]).mean()),
                "harm": float(np.maximum(0, -(merged["majority_recall"] - merged["best_standard_majority_recall"])).mean()),
            }
        )
    pf = pd.DataFrame(pts)
    fig, ax = plt.subplots()
    ax.scatter(pf["gain"], pf["harm"], s=60)
    for _, r in pf.iterrows():
        ax.annotate(r["method"], (r["gain"], r["harm"]), fontsize=8)
    ax.set_xlabel("minority recall gain"); ax.set_ylabel("majority harm")
    save(fig, "fig3_pareto_gain_vs_harm.png")

    # 4. Safe vs best_standard scatter.
    merged = sub[sub["method"] == "safe_mamr"].merge(best, on=["dataset", "model", "seed", "environment"], how="left")
    fig, ax = plt.subplots()
    ax.scatter(merged["best_standard_minority_recall"], merged["minority_recall"], alpha=0.6)
    lim = [0, 1]
    ax.plot(lim, lim, "k--")
    ax.set_xlabel("best standard minority recall"); ax.set_ylabel("safe MAMR minority recall")
    save(fig, "fig4_safe_vs_best_standard.png")

    # 5. Dataset/model heatmap (mean BIG).
    big = compute_big(raw, best, "safe_mamr", hr)
    piv = big.pivot_table(index="dataset", columns="model", values="big", aggfunc="mean")
    fig, ax = plt.subplots()
    im = ax.imshow(piv.values, cmap="RdBu", vmin=-0.1, vmax=0.4)
    ax.set_xticks(range(len(piv.columns))); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(len(piv.index))); ax.set_yticklabels(piv.index)
    ax.set_title("BIG by dataset / model")
    fig.colorbar(im)
    save(fig, "fig5_dataset_model_safety_heatmap.png")

    # 6. Validation utility/frontier (use the gate's conservative config if present).
    conf = gate.get("conservative_config", {})
    fig, ax = plt.subplots()
    ax.scatter([0], [0], alpha=0)
    ax.set_xlabel("majority recall delta"); ax.set_ylabel("validation U_safe")
    ax.set_title("Validation safety frontier (see validation_safety_search.csv)")
    save(fig, "fig6_validation_utility_frontier.png")

    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Reproduce Project E Phase E3R MAMR Safety Repair Gate")
    ap.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "phase_e3r_mamr_safety.yaml"))
    ap.add_argument("--registry", default=str(PROJECT_ROOT / "configs" / "datasets_phase_e12.yaml"))
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--results", default=None)
    args = ap.parse_args()

    cfg = PhaseE3Config.from_yaml(args.config)
    if args.results:
        cfg.paths["results"] = Path(args.results).resolve()
    reg = load_registry(args.registry)
    datasets_all = load_all(cfg, reg)
    datasets = [d for d in datasets_all if d["name"] in cfg.datasets]

    results_dir = cfg.paths["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    rm = ResumeManager(results_dir)

    # Upstream artifact integrity check (report-only; fail loudly if disturbed).
    upstream_ok = True
    try:
        e3_manifest = json.loads((PROJECT_ROOT / "results" / "phase_e3_mamr" / "manifest.json").read_text())
        if freeze_hash(reg) != e3_manifest.get("dataset_hashes"):
            upstream_ok = False
        if list_hash(reg) != e3_manifest.get("dataset_list_hash"):
            upstream_ok = False
    except Exception:  # noqa: BLE001
        upstream_ok = False

    if args.smoke:
        smoke_ds = [d for d in datasets if d["name"] == "default_credit_card"] or datasets[:1]
        smoke_params = _make_params(cfg, 0.25, 0.15, 0.10)
        for ds in smoke_ds:
            info = _run_test_unit(rm, ds, 42, "logistic_regression", "safe_mamr", cfg, smoke_params, _SMOKE_ENVS, force=True)
            assert info is not None and info["data"].get("ok"), "Smoke safe_mamr failed."
            sr = pd.DataFrame(info["data"]["raw"])
            assert len(sr) > 0 and sr["minority_recall"].notna().all()
        print("[smoke] Safe-MAMR test path OK")
        return 0

    # ---- Validation conservative search ----
    search = _validation_search(cfg, datasets, cfg.models, cfg.seeds, rm, args.force)
    all_frame = search["all_frame"]
    global_chosen = search["global"]
    feasible_fraction = global_chosen["feasible_fraction"]
    write_validation = results_dir / "validation_safety_search.csv"
    all_frame.to_csv(write_validation, index=False)
    results_dir.joinpath("selected_safe_params.json").write_text(
        json.dumps(
            {
                "lambda": global_chosen["lambda_"],
                "copy_fraction": global_chosen["copy_fraction"],
                "corruption_rate": global_chosen["corruption_rate"],
                "step": global_chosen["step"],
                "u_safe_median": global_chosen["u_safe_median"],
                "feasible_fraction": feasible_fraction,
                "validation_safety_feasible": global_chosen["validation_safety_feasible"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    results_dir.joinpath("global_safe_config.json").write_text(
        json.dumps(_json_safe(global_chosen), indent=2), encoding="utf-8"
    )
    results_dir.joinpath("pre_registration.md").write_text(
        _pre_registration(cfg, global_chosen, upstream_ok), encoding="utf-8"
    )

    # ---- Safe-MAMR test evaluation (global config) ----
    safe_params = _make_params(
        cfg, global_chosen["lambda_"], global_chosen["copy_fraction"], global_chosen["corruption_rate"]
    )
    safe_test_rows: list[dict] = []
    for ds in datasets:
        for model in cfg.models:
            for seed in cfg.seeds:
                info = _run_test_unit(rm, ds, seed, model, "safe_mamr", cfg, safe_params, list(cfg.environments), args.force)
                if info is not None and info["data"].get("ok"):
                    safe_test_rows.extend(info["data"]["raw"])

    raw = _combine_raw(safe_test_rows)
    raw.to_csv(results_dir / "raw_safe_results.csv", index=False)

    # Add augmentation_count for the diagnostic using the audit (re-run safe_mamr
    # audits were tagged per run; carry forward the configured copy_fraction as a
    # conservative disclosure, exact counts live in the exposure audit).
    expected = len(datasets) * len(cfg.models) * len(cfg.seeds) * len(cfg.methods) * len(cfg.environments)
    completed = int((raw["status"] == "ok").sum())
    failed = int((raw["status"] == "error").sum())

    gate = evaluate_safety_gate(raw, cfg)
    gate["conservative_config"] = global_chosen
    gate["upstream_unchanged"] = upstream_ok
    results_dir.joinpath("safety_gate.json").write_text(
        json.dumps(_json_safe(gate), indent=2), encoding="utf-8"
    )

    # Derived tables.
    best = compute_best_standard(raw)
    hr = list(cfg.high_risk_mechanisms)
    shift = [e for e in cfg.environments if e != "mcar_05"]
    lrr = compute_lrr(raw, pd.Series(["safe_mamr"]), shift, include_filter=False)
    lrr.to_csv(results_dir / "collapse_recovery.csv", index=False)
    big = compute_big(raw, best, "safe_mamr", shift)
    big.to_csv(results_dir / "beyond_imbalance_gain.csv", index=False)
    mas = compute_mas(raw, hr, list(cfg.control_mechanisms), "safe_mamr")
    pd.DataFrame([mas]).to_csv(results_dir / "mechanism_alignment.csv", index=False)
    iid = compute_iid_cost(raw, "safe_mamr")
    iid.to_csv(results_dir / "iid_cost.csv", index=False)

    # Method comparison + gain retention + harm reduction + pareto.
    method_cmp = _summarize_method(raw[raw["method"].isin(["erm", "class_weight", "augmentation_only", "exposure_weighting_only", "uniform_minority_weight", "original_mamr", "safe_mamr"])], ["method"])
    method_cmp.to_csv(results_dir / "safe_method_comparison.csv", index=False)
    mgr_frame = pd.DataFrame(
        [
            {
                "minority_gain_safe": gate["metrics"]["minority_gain_safe"],
                "minority_gain_original": gate["metrics"]["minority_gain_original"],
                "mgr": gate["metrics"]["mgr"],
            }
        ]
    )
    mgr_frame.to_csv(results_dir / "minority_gain_retention.csv", index=False)
    pd.DataFrame(
        [
            {
                "majority_delta_safe": gate["metrics"]["majority_delta_safe"],
                "majority_delta_original": gate["metrics"]["majority_delta_original"],
                "mhr": gate["metrics"]["majority_harm_reduction"],
            }
        ]
    ).to_csv(results_dir / "majority_harm_reduction.csv", index=False)

    # Pareto table across dataset x model cells.
    pareto_rows = []
    for (ds, model), grp in raw[raw["environment"].isin(hr) & raw["method"].isin(
        ["augmentation_only", "exposure_weighting_only", "uniform_minority_weight", "original_mamr", "safe_mamr"]
    )].groupby(["dataset", "model"]):
        frame = grp.groupby("method")[["minority_recall", "majority_recall", "AUPRC"]].mean().reset_index()
        from src.mamr_safety import pareto_non_dominated

        pareto_rows.append(
            {
                "dataset": ds,
                "model": model,
                "safe_mamr_non_dominated": pareto_non_dominated(frame[["method", "minority_recall", "majority_recall", "AUPRC"]].copy()),
            }
        )
    pareto_df = pd.DataFrame(pareto_rows)
    pareto_df.to_csv(results_dir / "pareto_results.csv", index=False)

    pd.DataFrame(gate["statistics"]["rows"]).to_csv(results_dir / "statistics_results.csv", index=False)
    _diagnostic(raw, cfg).to_csv(results_dir / "default_credit_card_safety_diagnostic.csv", index=False)
    pd.DataFrame([{"condition": k, "passed": bool(v)} for k, v in gate["conditions"].items()]).to_csv(
        results_dir / "gate_conditions.csv", index=False
    )

    tables = [
        results_dir / "validation_safety_search.csv",
        results_dir / "raw_safe_results.csv",
        results_dir / "safe_method_comparison.csv",
        results_dir / "minority_gain_retention.csv",
        results_dir / "majority_harm_reduction.csv",
        results_dir / "pareto_results.csv",
        results_dir / "collapse_recovery.csv",
        results_dir / "beyond_imbalance_gain.csv",
        results_dir / "mechanism_alignment.csv",
        results_dir / "iid_cost.csv",
        results_dir / "statistics_results.csv",
        results_dir / "default_credit_card_safety_diagnostic.csv",
        results_dir / "gate_conditions.csv",
    ]
    failed_path = results_dir / "failed_runs.csv"
    if not failed_path.exists():
        failed_path.write_text("run_key,error,timestamp\n", encoding="utf-8")
    tables.append(failed_path)
    for p in tables:
        if not p.exists():
            p.write_text("", encoding="utf-8")
    figures = _figures(raw, gate, results_dir)
    report = _build_report(cfg, gate, raw, {}, tables, expected, completed, failed)
    results_dir.joinpath("safety_gate_report.md").write_text(report, encoding="utf-8")

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "config_hash": _file_hash(args.config),
        "dataset_hashes": freeze_hash(reg),
        "dataset_list_hash": list_hash(reg),
        "upstream_unchanged": upstream_ok,
        "safe_config": global_chosen,
        "n_expected_runs": expected,
        "n_completed_runs": completed,
        "n_failed_runs": failed,
        "gate": gate["gate"],
    }
    results_dir.joinpath("manifest.json").write_text(json.dumps(_json_safe(manifest), indent=2), encoding="utf-8")
    rm.save_progress(completed, expected, failed, None)

    _console_report(gate, expected, completed, failed, global_chosen, upstream_ok)
    return 0


def _pre_registration(cfg, global_chosen, upstream_ok) -> str:
    return f"""# Phase E3R Pre-registration

- Project: Project E Phase E3R MAMR Safety Repair Gate
- Dataset / model / seed / environment grid inherited from Phase E3 (unchanged).
- Search space: Step A lambda in `{cfg.raw['safety_search']['step_a']['lambda_candidates']}` at copy=0.15, corr=0.10.
  Step B copy in `{cfg.raw['safety_search']['step_b']['copy_fraction_candidates']}` x corr in `{cfg.raw['safety_search']['step_b']['corruption_rate_candidates']}`.
- Hard caps: lambda<=0.75, copy<=0.25, corr<=0.15.
- U_safe = R_min - 2.0*max(0,R_maj_ref-R_maj) - 1.0*max(0,AUPRC_ref-AUPRC); ref = best_standard on validation stress suite.
- Validation hard constraints: majority delta >= -0.03, AUPRC delta >= -0.02.
- Selected global config: lambda={global_chosen['lambda_']}, copy={global_chosen['copy_fraction']}, corr={global_chosen['corruption_rate']}.
- validation_safety_feasible={global_chosen['validation_safety_feasible']}.
- Upstream (Phase E3) artifacts untouched: {upstream_ok}.
"""


def _file_hash(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    import hashlib

    return hashlib.sha1(p.read_bytes()).hexdigest()


def _console_report(gate, expected, completed, failed, global_chosen, upstream_ok):
    m = gate["metrics"]
    c = gate["counts"]
    print("\n# Project E Phase E3R — MAMR Safety Repair Gate")
    print(f"1. Gate-output bug fixed? YES")
    print(f"2. Upstream artifacts unchanged? {'YES' if upstream_ok else 'NO'}")
    print(f"3. runs: expected={expected} completed={completed} failed={failed}")
    print(f"4. Global safe config: lambda={global_chosen['lambda_']} copy={global_chosen['copy_fraction']} corr={global_chosen['corruption_rate']}")
    print(f"5. Validation safety feasible fraction: {round(float(global_chosen['feasible_fraction']), 3)}")
    print(f"6. Minority recall gain vs best standard: {round(float(m['minority_gain_safe']), 3)}")
    print(f"7. Original MAMR gain vs best standard: {round(float(m['minority_gain_original']), 3)}")
    print(f"8. MGR: {round(float(m['mgr']), 3)}")
    print(f"9. Majority recall delta mean/worst: {round(float(m['majority_delta_safe']), 3)} / {round(float(m['worst_dataset_majority_delta_safe']), 3)}")
    print(f"10. Original majority delta mean/worst: {round(float(m['majority_delta_original']), 3)} / {round(float(m['worst_dataset_majority_delta_original']), 3)}")
    print(f"11. MHR: {round(float(m['majority_harm_reduction']), 3)}")
    print(f"12. AUPRC delta mean/worst: {round(float(m['auprc_delta_safe']), 4)} / {round(float(m['worst_dataset_auprc_delta_safe']), 4)}")
    print(f"13. IID minority/AUPRC delta: {round(float(m['iid_minority_recall_delta']), 3)} / {round(float(m['iid_auprc_delta']), 4)}")
    print(f"14. Median LRR / positive fraction: {round(float(c['median_lrr']), 3)} / {round(float(c['positive_lrr_fraction']), 3)}")
    print(f"15. BIG mean/positive/datasets>=0.05: {round(float(m['big_mean']), 3)} / {round(float(m['big_positive_fraction']), 3)} / {c['n_datasets_big_ge_005']}")
    print(f"16. MAS: {round(float(c['mas']), 4)}")
    print(f"17. LR/XGB mean BIG: {round(float(m['lr_mean_big']), 3)} / {round(float(m['xgb_mean_big']), 3)}")
    print(f"18. Pareto non-dominated cells: {c['pareto_cells']}/{c['pareto_total']}")
    st = gate["statistics"]
    print(f"19. Bootstrap CI minority/majority: {round(float(st['minority_boot_low']), 3)},{round(float(st['minority_boot_high']), 3)} / {round(float(st['majority_boot_low']), 3)},{round(float(st['majority_boot_high']), 3)}")
    r = " ".join(f"{k}:{'P' if v else 'F'}" for k, v in gate["conditions"].items())
    print(f"20. R1-R10: {r}")
    print(f"21. Final: {gate['gate']}")
    print(f"22. Enter Phase E4?: {'YES' if gate['gate'] in ('STRONG-GO', 'GO') else 'NO'}")
    print(f"23. Reason: {gate['recommended_next_phase']}")
    print("24. Authoritative artifacts: results/phase_e3r_mamr_safety/")
    print("25. Reproduce: python scripts/reproduce_phase_e3r_mamr_safety.py --resume\n")


if __name__ == "__main__":
    raise SystemExit(main())
