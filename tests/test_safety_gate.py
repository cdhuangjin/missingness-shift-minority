"""Safety Gate R1-R10 evaluation tests (including mandatory R3 / HOLD-STOP)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.phase_e3_config import PhaseE3Config
from src.safety_gate import evaluate_safety_gate

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    cfg = PhaseE3Config.from_yaml(ROOT / "configs" / "phase_e3r_mamr_safety.yaml")
    # Restrict to the envs used by the synthetic frame to keep it exact.
    cfg.environments = ["mcar_05", "feature_specific", "block_severe", "cc_40", "cc_50", "reverse_cc_40"]
    return cfg


def _synthetic_raw(hr_minority: float = 0.90, hr_majority: float = 0.60) -> pd.DataFrame:
    """A 1-dataset / 1-model / 1-seed frame with hand-set operating points.

    ``safe_mamr`` has a large minority gain but a large majority sacrifice on the
    high-risk envs, which is the Phase E3 pattern (R3 fails, benefit significant).
    """
    hr = ["feature_specific", "block_severe", "cc_40", "cc_50"]
    envs = ["mcar_05", "reverse_cc_40"] + hr
    rows = []

    def add(method, env, mr, maj, ap, ba, mcc):
        rows.append(
            {
                "dataset": "ds",
                "model": "m",
                "seed": 42,
                "method": method,
                "environment": env,
                "minority_recall": mr,
                "majority_recall": maj,
                "AUPRC": ap,
                "AUROC": 0.87,
                "balanced_accuracy": ba,
                "MCC": mcc,
                "status": "ok",
                "error": "",
            }
        )

    # Baselines: good majority, moderate minority.
    for method in ["erm", "class_weight", "random_oversampling", "missing_indicator"]:
        for env in envs:
            on_hr = env in hr
            mr = 0.50 if on_hr else 0.78
            add(method, env, mr, 0.96 if on_hr else 0.98, 0.62 if on_hr else 0.72, 0.74, 0.45)
    # Pareto / reference MAMR-like methods.
    for method, base_mr, base_maj in [
        ("augmentation_only", 0.70, 0.88),
        ("exposure_weighting_only", 0.72, 0.86),
        ("uniform_minority_weight", 0.73, 0.85),
        ("original_mamr", 0.85, 0.65),
    ]:
        for env in envs:
            on_hr = env in hr
            add(method, env, base_mr if on_hr else 0.80, base_maj if on_hr else 0.95, 0.68, 0.80, 0.50)
    # Safe MAMR: biggest minority gain, biggest majority sacrifice on high-risk.
    for env in envs:
        on_hr = env in hr
        ap = 0.70 if on_hr else 0.73  # keep IID AUPRC safe on the control envs
        add("safe_mamr", env, hr_minority if on_hr else 0.82, hr_majority if on_hr else 0.94, ap, 0.78, 0.46)
    return pd.DataFrame(rows)


def test_gate_structure():
    cfg = _cfg()
    gate = evaluate_safety_gate(_synthetic_raw(), cfg)
    assert set(gate["conditions"]) == {f"R{i}" for i in range(1, 11)}
    assert gate["gate"] in ("STRONG-GO", "GO", "HOLD-STOP", "STOP")
    assert "recommended_next_phase" in gate


def test_r3_mandatory_blocks_go():
    cfg = _cfg()
    gate = evaluate_safety_gate(_synthetic_raw(), cfg)
    # The synthetic safe MAMR sacrifices majority heavily -> R3 fails.
    assert gate["conditions"]["R3"] is False
    assert gate["gate"] not in ("GO", "STRONG-GO")


def test_hold_stop_terminal_when_benefit_significant():
    cfg = _cfg()
    gate = evaluate_safety_gate(_synthetic_raw(), cfg)
    assert gate["gate"] == "HOLD-STOP"
    assert "no further MAMR repair" in gate["recommended_next_phase"]


def test_stop_when_no_benefit():
    cfg = _cfg()
    # Make safe MAMR worse than the best standard on minority recall everywhere.
    gate = evaluate_safety_gate(_synthetic_raw(hr_minority=0.30, hr_majority=0.96), cfg)
    assert gate["gate"] == "STOP"
