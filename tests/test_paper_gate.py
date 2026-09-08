"""Paper Gate verdicts for synthetic STRONG-GO / GO / HOLD / STOP cases."""

import numpy as np
import pandas as pd

from src.paper_gate import paper_gate


def _ds(names, mvg):
    return pd.DataFrame({"dataset": names, "mean_MVG": mvg})


def _models(names, mvg):
    return pd.DataFrame({"model": names, "mean_MVG": mvg})


def _slopes(datasets, minor, major):
    return pd.DataFrame({
        "dataset": datasets * 2,
        "model": ["LR"] * len(datasets) + ["XGB"] * len(datasets),
        "minority_slope": minor + minor,
        "majority_slope": major + major,
    })


def _cc(datasets):
    rows = []
    for ds in datasets:
        for lvl in ("cc_20", "cc_30", "cc_40", "cc_50"):
            rows.append({"dataset": ds, "model": "LR", "cc_level": lvl,
                         "CCEP_recall": 0.15})
    return pd.DataFrame(rows)


def _hidden(datasets):
    return pd.DataFrame({"dataset": datasets, "n_hidden": [2] * len(datasets)})


def _reverse(datasets, delta=0.12):
    return pd.DataFrame({"dataset": datasets, "delta_MVG_reverse": [delta] * len(datasets)})


def _stats(fdr_p=0.001, es=0.6):
    return {"fdr_p": fdr_p, "effect_size": es}


def test_strong_go():
    ds = ["a", "b", "c", "d", "e", "f", "g", "h"]
    verdict, meta = paper_gate(
        _ds(ds, [0.2] * 8), _models(["LR", "RF", "XGB", "LGBM"], [0.2] * 4),
        _slopes(ds, [-0.4] * 8, [-0.05] * 8), _cc(ds), _hidden(ds),
        _reverse(ds), _stats(), n_datasets=8, effect_vs_seed_noise_ok=True,
    )
    assert verdict == "STRONG-GO"
    assert meta["n_hit"] >= 7


def test_go():
    ds = ["a", "b", "c", "d", "e", "f", "g", "h"]
    # Stable positive MVG and steeper minority slope, but CC/hidden weak.
    verdict, _ = paper_gate(
        _ds(ds, [0.15] * 8), _models(["LR", "RF", "XGB", "LGBM"], [0.15] * 4),
        _slopes(ds, [-0.4] * 8, [-0.05] * 8),
        pd.DataFrame(columns=["dataset", "model", "cc_level", "CCEP_recall"]),
        pd.DataFrame(columns=["dataset", "n_hidden"]),
        _reverse(ds, delta=0.01), _stats(es=0.2), n_datasets=8, effect_vs_seed_noise_ok=True,
    )
    assert verdict == "GO"


def test_hold_when_model_dependent():
    ds = ["a", "b", "c", "d", "e", "f", "g", "h"]
    # Only two models positive but the MVG is stable across datasets.
    verdict, _ = paper_gate(
        _ds(ds, [0.15, 0.15, 0.12, 0.05, 0.02, 0.0, -0.01, 0.0]),
        _models(["LR", "RF", "XGB", "LGBM"], [0.15, -0.02, 0.01, -0.05]),
        _slopes(ds, [-0.4] * 8, [-0.05] * 8),
        pd.DataFrame(columns=["dataset", "model", "cc_level", "CCEP_recall"]),
        pd.DataFrame(columns=["dataset", "n_hidden"]),
        _reverse(ds, delta=0.01), _stats(), n_datasets=8, effect_vs_seed_noise_ok=True,
    )
    assert verdict == "HOLD"


def test_stop_pivot_when_effect_vanishes():
    ds = ["a", "b", "c", "d", "e", "f", "g", "h"]
    verdict, _ = paper_gate(
        _ds(ds, [0.0, -0.02, 0.01, 0.0, -0.01, 0.02, 0.0, 0.0]),
        _models(["LR", "RF", "XGB", "LGBM"], [0.0, 0.01, -0.01, 0.0]),
        _slopes(ds, [-0.05] * 8, [-0.04] * 8),
        pd.DataFrame(columns=["dataset", "model", "cc_level", "CCEP_recall"]),
        pd.DataFrame(columns=["dataset", "n_hidden"]),
        _reverse(ds, delta=0.01),
        _stats(fdr_p=0.5, es=0.05), n_datasets=8, effect_vs_seed_noise_ok=False,
    )
    assert verdict == "STOP-PIVOT"
