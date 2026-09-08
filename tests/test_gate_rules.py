"""Synthetic GO / HOLD / STOP cases for the automatic gate."""

import pandas as pd

from src.config import Config
from src.gate_decision import decide, evaluate_go


def _cfg() -> Config:
    cfg = Config()
    cfg.gate.update(
        {
            "strong_mvg": 0.10,
            "moderate_mvg": 0.05,
            "strong_recall_drop": 0.10,
            "stable_auroc_drop": 0.03,
            "slope_ratio": 1.5,
            "min_seed_consistency": 2,
            "min_datasets": 2,
        }
    )
    cfg.seeds = [42, 52, 62]
    return cfg


def _evidence(strongest_mvg, seed_consistency, ccep, hidden_frac):
    rows = []
    for ds in ("ds1", "ds2"):
        for model in ("xgb", "lr"):
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "strongest_MVG": strongest_mvg,
                    "seed_consistency": seed_consistency,
                    "class_conditional_seed_consistency": seed_consistency,
                    "CCEP_recall": ccep,
                    "hidden_seed_fraction": hidden_frac,
                    "hidden_failure": hidden_frac > 0,
                    "hidden_failure_count": 1 if hidden_frac > 0 else 0,
                    "indicator_gain_auprc": 0.0,
                }
            )
    return pd.DataFrame(rows)


def _slopes(minority, majority):
    return pd.DataFrame(
        {
            "dataset": ["ds1", "ds2"],
            "model": ["xgb", "lr"],
            "minority_slope": [minority, minority],
            "majority_slope": [majority, majority],
            "slope_gap": [abs(minority) - abs(majority)] * 2,
            "slope_ratio": [abs(minority) / (abs(majority) + 1e-12)] * 2,
        }
    )


def test_go_decision():
    ev = _evidence(0.15, 1.0, 0.15, 1.0)
    sl = _slopes(-0.5, -0.2)
    go, verdict = decide(ev, pd.DataFrame(), sl, pd.DataFrame(), _cfg())
    assert verdict == "GO"
    assert go["go_conditions_hit"]


def test_hold_decision():
    ev = _evidence(0.07, 0.5, 0.0, 0.0)
    sl = _slopes(-0.2, -0.19)
    go, verdict = decide(ev, pd.DataFrame(), sl, pd.DataFrame(), _cfg())
    assert verdict == "HOLD"
    assert go["hold_signals"]


def test_stop_decision():
    ev = _evidence(0.02, 0.0, 0.0, 0.0)
    sl = _slopes(-0.2, -0.15)
    go, verdict = decide(ev, pd.DataFrame(), sl, pd.DataFrame(), _cfg())
    assert verdict == "STOP/PIVOT"


def test_evaluate_go_flags():
    ev = _evidence(0.15, 1.0, 0.15, 1.0)
    sl = _slopes(-0.5, -0.2)
    go = evaluate_go(ev, pd.DataFrame(), sl, pd.DataFrame(), _cfg())
    assert go["verdicts"]["GO_A_minority_vulnerability_gap"] is True
