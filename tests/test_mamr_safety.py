"""Unit tests for the E3R conservative safety configuration support."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataset_registry import freeze_hash, list_hash, load_registry
from src.mamr_safety import (
    build_candidate_grid_step_a,
    build_candidate_grid_step_b,
    compute_mgr,
    compute_mhr,
    pareto_non_dominated,
    safe_utility,
    safety_feasible,
    select_global_safe_config,
)
from src.phase_e3_config import PhaseE3Config
from src.resume_manager import ResumeManager

ROOT = Path(__file__).resolve().parents[1]


def _cfg():
    return PhaseE3Config.from_yaml(ROOT / "configs" / "phase_e3r_mamr_safety.yaml")


def test_grid_step_a_pre_registered():
    cfg = _cfg()
    grid = build_candidate_grid_step_a(cfg)
    assert [c["lambda_"] for c in grid] == [0.0, 0.10, 0.25, 0.50, 0.75]
    assert all(c["copy_fraction"] == 0.15 for c in grid)
    assert all(c["corruption_rate"] == 0.10 for c in grid)


def test_grid_step_b_pre_registered():
    cfg = _cfg()
    grid = build_candidate_grid_step_b(cfg, 0.25)
    assert len(grid) == 12  # 4 copy x 3 corruption rates
    assert all(c["lambda_"] == 0.25 for c in grid)
    assert {c["copy_fraction"] for c in grid} == {0.10, 0.15, 0.20, 0.25}
    assert {c["corruption_rate"] for c in grid} == {0.05, 0.10, 0.15}


def test_hard_caps_respected():
    cfg = _cfg()
    caps = cfg.raw["safety_search"]["hard_caps"]
    for cand in build_candidate_grid_step_a(cfg) + build_candidate_grid_step_b(cfg, 0.5):
        assert cand["lambda_"] <= caps["lambda"]
        assert cand["copy_fraction"] <= caps["copy_fraction"]
        assert cand["corruption_rate"] <= caps["corruption_rate"]


def test_safe_utility_formula():
    # U = R_min - alpha*max(0, ref_maj - maj) - beta*max(0, ref_ap - ap)
    u = safe_utility(0.80, 0.80, 0.70, 0.60, 0.90, 0.65, alpha=2.0, beta=1.0)
    assert np.isclose(u, 0.80 - 2.0 * max(0.0, 0.90 - 0.80) - 1.0 * max(0.0, 0.65 - 0.70), atol=1e-9)


def test_safety_feasible_thresholds():
    assert safety_feasible(-0.03, -0.02, -0.03, -0.02) is True
    assert safety_feasible(-0.031, -0.02, -0.03, -0.02) is False
    assert safety_feasible(-0.03, -0.021, -0.03, -0.02) is False


def test_select_global_prefers_feasible_with_best_median():
    rows = []
    # lambda=0.0 fully feasible (3/3) and highest U_safe -> must be chosen.
    for i, (l, feas) in enumerate([(0.0, [True, True, True]), (0.5, [True, True, False]), (0.75, [True, False, False])]):
        for cell in range(3):
            rows.append(
                {
                    "step": "A",
                    "lambda_": l,
                    "copy_fraction": 0.15,
                    "corruption_rate": 0.10,
                    "u_safe": 0.9 - i * 0.1,
                    "majority_delta": -0.02,
                    "auprc_delta": 0.0,
                    "feasible": bool(feas[cell]),
                }
            )
    frame = pd.DataFrame(rows)
    chosen = select_global_safe_config(frame, feasible_fraction_min=0.80)
    assert chosen["lambda_"] == 0.0
    assert chosen["validation_safety_feasible"] is True


def test_select_global_infeasible_fallback():
    rows = [
        {"step": "A", "lambda_": l, "copy_fraction": 0.15, "corruption_rate": 0.10,
         "u_safe": 0.6, "majority_delta": d, "auprc_delta": 0.0, "feasible": False}
        for l, d in [(0.0, -0.02), (0.25, -0.10), (0.75, -0.20)]
    ]
    frame = pd.DataFrame(rows)
    chosen = select_global_safe_config(frame, feasible_fraction_min=0.80)
    assert chosen["validation_safety_feasible"] is False
    assert chosen["majority_delta_mean"] == -0.02  # least majority sacrifice


def test_mgr_mhr():
    assert np.isclose(compute_mgr(0.3, 0.5), 0.6, atol=1e-9)
    # H_safe = max(0, -(-0.05)) = 0.05 ; H_orig = max(0, -(-0.20)) = 0.20 -> 1 - 0.25 = 0.75
    assert np.isclose(compute_mhr(-0.05, -0.20), 0.75, atol=1e-9)
    assert np.isclose(compute_mhr(0.01, -0.20), 1.0, atol=1e-9)  # no safe harm


def test_pareto_detection():
    dominated = pd.DataFrame(
        [
            {"method": "safe_mamr", "minority_recall": 0.8, "majority_recall": 0.7, "AUPRC": 0.6},
            {"method": "original_mamr", "minority_recall": 0.9, "majority_recall": 0.8, "AUPRC": 0.7},
        ]
    )
    assert pareto_non_dominated(dominated) is False
    nd = pd.DataFrame(
        [
            {"method": "safe_mamr", "minority_recall": 0.9, "majority_recall": 0.7, "AUPRC": 0.6},
            {"method": "original_mamr", "minority_recall": 0.8, "majority_recall": 0.75, "AUPRC": 0.7},
        ]
    )
    assert pareto_non_dominated(nd) is True


def test_split_same_as_e3():
    e3 = PhaseE3Config.from_yaml(ROOT / "configs" / "phase_e3_mamr.yaml")
    e3r = _cfg()
    assert e3.split == e3r.split


def test_deterministic_selection():
    rows = [
        {"step": "A", "lambda_": l, "copy_fraction": 0.15, "corruption_rate": 0.10,
         "u_safe": u, "majority_delta": d, "auprc_delta": 0.0, "feasible": f}
        for l, u, d, f in [(0.0, 0.8, -0.02, True), (0.5, 0.7, -0.05, True), (0.75, 0.6, -0.10, False)]
    ]
    frame = pd.DataFrame(rows)
    a = select_global_safe_config(frame, feasible_fraction_min=0.80)
    b = select_global_safe_config(frame, feasible_fraction_min=0.80)
    assert a == b


def test_upstream_artifact_hashes_unchanged():
    reg = load_registry(ROOT / "configs" / "datasets_phase_e12.yaml")
    manifest = json.loads((ROOT / "results" / "phase_e3_mamr" / "manifest.json").read_text())
    assert freeze_hash(reg) == manifest["dataset_hashes"]
    assert list_hash(reg) == manifest["dataset_list_hash"]


def test_resume_persists_and_is_complete():
    import shutil
    import tempfile

    # The pytest tmp_path fixture cannot scan the OS temp dir on this machine, so
    # build a small temporary directory under the project root and clean it up.
    tmp = Path(tempfile.mkdtemp(prefix="e3r_resume_", dir=ROOT))
    rm = ResumeManager(tmp / "res")
    key = "d|42|lr|safe_mamr|test|l0.25|c0.15|r0.10"
    assert rm.is_complete(key) is False
    rm.save(key, {"ok": True, "raw": [{"x": 1}]})
    assert rm.is_complete(key) is True
    assert rm.load(key)["data"]["ok"] is True
    # Only remove the directory we created, and only if it's under the repo root.
    if tmp.resolve().is_relative_to(ROOT.resolve()):
        shutil.rmtree(tmp)
