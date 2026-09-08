"""Derived-metric formulas and Method Gate decision tests."""

import numpy as np
import pandas as pd

from src.method_gate import (
    _decide,
    compute_best_standard,
    compute_big,
    compute_iid_cost,
    compute_lrr,
    compute_mas,
    evaluate_gate,
    run_statistics,
)
from src.phase_e3_config import PhaseE3Config

ROOT = __file__


def _cfg(config_path: str | None = None):
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "configs" / "phase_e3_mamr.yaml"
    return PhaseE3Config.from_yaml(path)


def _raw():
    """Small 1-dataset scenario with hand-checkable numbers."""
    rows = [
        # dataset, model, seed, method, environment, minority, majority, auprc, balacc, mcc
        ("ds", "m", 42, "erm", "mcar_05", 0.80, 0.99, 0.70, 0.88, 0.50),
        ("ds", "m", 42, "erm", "cc_50", 0.40, 0.98, 0.55, 0.70, 0.30),
        ("ds", "m", 42, "class_weight", "mcar_05", 0.70, 0.95, 0.68, 0.82, 0.45),
        ("ds", "m", 42, "class_weight", "cc_50", 0.50, 0.96, 0.58, 0.76, 0.40),
        ("ds", "m", 42, "mamr_full", "mcar_05", 0.82, 0.90, 0.69, 0.86, 0.42),
        ("ds", "m", 42, "mamr_full", "cc_50", 0.55, 0.93, 0.60, 0.75, 0.35),
    ]
    df = pd.DataFrame(
        rows,
        columns=["dataset", "model", "seed", "method", "environment",
                 "minority_recall", "majority_recall", "AUPRC",
                 "balanced_accuracy", "MCC"],
    )
    return df


def test_lrr_formula():
    raw = _raw()
    lrr = compute_lrr(raw, pd.Series(["mamr_full"]), ["cc_50"], include_filter=True)
    row = lrr[lrr["method"] == "mamr_full"].iloc[0]
    assert np.isclose(row["lrr"], (0.55 - 0.40) / (0.80 - 0.40), atol=1e-9)


def test_big_formula():
    raw = _raw()
    best = compute_best_standard(raw)
    big = compute_big(raw, best, "mamr_full")
    row = big[big["environment"] == "cc_50"].iloc[0]
    assert np.isclose(row["big"], 0.55 - 0.50, atol=1e-9)


def test_mas_formula():
    raw = _raw()
    mas = compute_mas(raw, ["cc_50"], ["mcar_05"])
    # high-risk delta = 0.55-0.40 = 0.15; control delta = 0.82-0.80 = 0.02.
    assert np.isclose(mas["mas"], 0.15 - 0.02, atol=1e-9)


def test_iid_cost_formula():
    raw = _raw()
    iid = compute_iid_cost(raw)
    row = iid.iloc[0]
    assert np.isclose(row["iid_minority_recall_delta"], 0.82 - 0.80, atol=1e-9)
    assert np.isclose(row["iid_auprc_delta"], 0.69 - 0.70, atol=1e-9)


def test_best_standard_picks_highest_minority_recall():
    raw = _raw()
    best = compute_best_standard(raw)
    cc = best[best["environment"] == "cc_50"].iloc[0]
    assert cc["best_standard_method"] == "class_weight"


def test_gate_decision_strong_go():
    conds = {"GO_A": True, "GO_B": True, "GO_C": True, "GO_D": True,
             "GO_E": True, "GO_F": True, "GO_G": True, "GO_H": True}
    assert _decide(conds, 7, [], []) == "STRONG-GO"


def test_gate_decision_go():
    conds = {"GO_A": True, "GO_B": False, "GO_C": True, "GO_D": True,
             "GO_E": True, "GO_F": True, "GO_G": False, "GO_H": True}
    assert _decide(conds, 6, [], []) == "GO"


def test_gate_decision_hold():
    conds = {"GO_A": True, "GO_B": False, "GO_C": True, "GO_D": True,
             "GO_E": True, "GO_F": True, "GO_G": False, "GO_H": False}
    assert _decide(conds, 5, [], []) == "HOLD"


def test_gate_decision_stop_on_signals():
    conds = {"GO_A": True, "GO_B": True, "GO_C": True, "GO_D": True,
             "GO_E": True, "GO_F": True, "GO_G": True, "GO_H": True}
    assert _decide(conds, 7, ["majority collapse"], []) == "STOP"


def test_evaluate_gate_runs():
    raw = _raw()
    cfg = _cfg()
    cfg.environments = ["mcar_05", "cc_50"]
    cfg.high_risk_mechanisms = ["cc_50"]
    cfg.control_mechanisms = ["mcar_05"]
    gate = evaluate_gate(raw, cfg)
    assert gate["gate"] in ("STRONG-GO", "GO", "HOLD", "STOP")
    assert set(gate["conditions"]) == {
        "GO_A", "GO_B", "GO_C", "GO_D", "GO_E", "GO_F", "GO_G", "GO_H"
    }


def test_run_statistics_no_crash():
    raw = _raw()
    cfg = _cfg()
    cfg.environments = ["mcar_05", "cc_50"]
    cfg.high_risk_mechanisms = ["cc_50"]
    cfg.control_mechanisms = ["mcar_05"]
    rows = run_statistics(raw, cfg, bootstrap_iterations=100)
    assert any(r["metric"] == "minority_recall" for r in rows)
