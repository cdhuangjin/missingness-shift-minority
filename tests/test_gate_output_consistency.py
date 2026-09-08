"""Regression tests for the Phase E3 gate output-consistency bug.

The E3 report/manifest could disagree with the console / GO conditions because
``_decide`` allowed STRONG-GO / GO without enforcing the mandatory majority
safety condition (GO_C). These tests pin the fix so ``method_gate.json``, the
report and the console verdict cannot diverge.
"""

from __future__ import annotations

import pytest

from src.method_gate import (
    _decide,
    evaluate_gate,
    validate_gate_consistency,
)
from src.phase_e3_config import PhaseE3Config
from pathlib import Path


def _cfg():
    path = Path(__file__).resolve().parents[1] / "configs" / "phase_e3_mamr.yaml"
    return PhaseE3Config.from_yaml(path)


def _e3_like_conditions():
    return {
        "GO_A": True,
        "GO_B": True,
        "GO_C": False,  # the only hard failure in Phase E3
        "GO_D": True,
        "GO_E": True,
        "GO_F": True,
        "GO_G": True,
        "GO_H": True,
    }


def _persistent_hold_signals():
    return ["significant majority recall tradeoff", "MAMR behaviour is unstable across datasets"]


def test_synthetic_seven_of_eight_go_c_false_is_hold():
    """§2: 7/8 conditions + GO-C=False + no STOP signal => HOLD."""
    conds = _e3_like_conditions()
    assert _decide(conds, 7, stop_signals=[], hold_signals=[]) == "HOLD"


def test_strong_go_requires_go_c():
    """STRONG-GO must never be returned while GO_C (majority safety) is False."""
    conds = _e3_like_conditions()
    assert _decide(conds, 7, stop_signals=[], hold_signals=[]) != "STRONG-GO"


def test_go_requires_go_c():
    conds = _e3_like_conditions()
    assert _decide(conds, 6, stop_signals=[], hold_signals=[]) != "GO"


def test_e3_hold_verdict_with_hold_signals():
    """Phase E3 regression: real HOLD payload keeps HOLD."""
    conds = _e3_like_conditions()
    assert (
        _decide(conds, 7, stop_signals=[], hold_signals=_persistent_hold_signals())
        == "HOLD"
    )


def test_validate_passes_on_consistent_hold_payload():
    payload = {
        "gate": "HOLD",
        "conditions": _e3_like_conditions(),
        "hold_signals": _persistent_hold_signals(),
        "recommended_next_phase": "Revise MAMR",
    }
    validate_gate_consistency(payload)


def test_validate_raises_on_inconsistent_go_without_go_c():
    payload = {
        "gate": "GO",
        "conditions": _e3_like_conditions(),  # GO_C is False
        "hold_signals": [],
        "recommended_next_phase": "Phase E4",
    }
    with pytest.raises(ValueError):
        validate_gate_consistency(payload)


def test_validate_raises_on_mismatched_next_phase():
    payload = {
        "gate": "HOLD",
        "conditions": _e3_like_conditions(),
        "hold_signals": _persistent_hold_signals(),
        "recommended_next_phase": "Stop MAMR",
    }
    with pytest.raises(ValueError):
        validate_gate_consistency(payload)


def test_evaluate_gate_returns_consistent_hold():
    """End-to-end: a hand-built E3-like raw frame yields HOLD that validates."""
    rows = [
        # dataset, model, seed, method, environment, minority, majority, auprc, balacc, mcc
        ("ds", "m", 42, "erm", "mcar_05", 0.80, 0.99, 0.70, 0.88, 0.50),
        ("ds", "m", 42, "erm", "cc_50", 0.40, 0.98, 0.55, 0.70, 0.30),
        ("ds", "m", 42, "class_weight", "mcar_05", 0.70, 0.95, 0.68, 0.82, 0.45),
        ("ds", "m", 42, "class_weight", "cc_50", 0.50, 0.96, 0.58, 0.76, 0.40),
        ("ds", "m", 42, "mamr_full", "mcar_05", 0.82, 0.80, 0.69, 0.81, 0.42),
        ("ds", "m", 42, "mamr_full", "cc_50", 0.55, 0.75, 0.60, 0.65, 0.35),
    ]
    import pandas as pd

    raw = pd.DataFrame(
        rows,
        columns=[
            "dataset", "model", "seed", "method", "environment",
            "minority_recall", "majority_recall", "AUPRC",
            "balanced_accuracy", "MCC",
        ],
    )
    cfg = _cfg()
    cfg.environments = ["mcar_05", "cc_50"]
    cfg.high_risk_mechanisms = ["cc_50"]
    cfg.control_mechanisms = ["mcar_05"]
    payload = evaluate_gate(raw, cfg)
    # The hand-built frame must exhibit the Phase E3 pattern (GO-C + hold).
    assert payload["conditions"]["GO_C"] is False
    assert payload["gate"] == "HOLD"
    assert payload["recommended_next_phase"] == "Revise MAMR"
