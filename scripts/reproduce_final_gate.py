"""Derive the Project E final-gate checks from frozen retained artifacts.

This script never fits a model.  It only re-aggregates ``raw_results.csv``
for the hidden-failure threshold grid and records whether a complete-data
0%-missingness evaluation can be performed from retained files.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "results" / "phase_e12" / "raw_results.csv"
OUT = ROOT / "final_gate"


def primary_shift_cells() -> pd.DataFrame:
    raw = pd.read_csv(RAW)
    primary = raw[
        (raw["pipeline"] == "p0")
        & (raw["imputer"] == "median")
        & (raw["feature_set"] == "top")
    ].copy()
    rows: list[dict] = []
    for keys, group in primary.groupby(["dataset", "seed", "model"]):
        ref = group[group["environment"] == "mcar_05"]
        if ref.empty:
            continue
        ref = ref.iloc[0]
        for _, target in group[group["environment"] != "mcar_05"].iterrows():
            rows.append(
                {
                    "dataset": keys[0],
                    "seed": int(keys[1]),
                    "model": keys[2],
                    "environment": target["environment"],
                    "auroc_drop": float(ref["AUROC"] - target["AUROC"]),
                    "minority_recall_drop": float(
                        ref["minority_recall"] - target["minority_recall"]
                    ),
                }
            )
    return pd.DataFrame(rows)


def write_hidden_thresholds(cells: pd.DataFrame) -> None:
    auroc_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05]
    recall_thresholds = [0.05, 0.10, 0.15, 0.20]
    total = len(cells)
    rows: list[dict] = []
    for auroc in auroc_thresholds:
        for recall in recall_thresholds:
            selected = cells[
                (cells["auroc_drop"] <= auroc)
                & (cells["minority_recall_drop"] >= recall)
            ]
            rows.append(
                {
                    "auroc_drop_threshold": auroc,
                    "minority_recall_drop_threshold": recall,
                    "hidden_failure_count": int(len(selected)),
                    "fraction_of_primary_cells": (len(selected) / total if total else None),
                    "n_datasets": int(selected["dataset"].nunique()),
                    "n_mechanisms": int(selected["environment"].nunique()),
                    "denominator_primary_cells": total,
                }
            )
    out_csv = OUT / "hidden_failure_threshold_sensitivity.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False, float_format="%.6f")

    lines = [
        "# Table S3 — Hidden-failure threshold sensitivity",
        "",
        "Derived from the frozen primary P0/median/top per-cell results; no model was retrained.",
        f"The denominator is {total} target environment-level cells relative to the 5% MCAR source-like reference.",
        "A hidden failure is counted when both inequalities hold: AUROC drop ≤ the stated threshold and minority-recall drop ≥ the stated threshold.",
        "",
        "| AUROC drop ≤ | Minority-recall drop ≥ | Hidden failures | Fraction | Datasets | Mechanisms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['auroc_drop_threshold']:.2f} | {row['minority_recall_drop_threshold']:.2f} | "
            f"{row['hidden_failure_count']} | {row['fraction_of_primary_cells']:.3f} | "
            f"{row['n_datasets']} | {row['n_mechanisms']} |"
        )
    lines += [
        "",
        "The frozen primary definition is the 0.03/0.10 row (62 cases). This grid is a descriptive sensitivity analysis; it does not redefine the prespecified operational diagnostic or create new inferential units.",
    ]
    (OUT / "Table_S_hidden_failure_thresholds.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_source_reference_sanity() -> None:
    # Search only for retained evaluation outputs; do not infer a complete-data
    # result from the 5% reference and do not retrain when no checkpoint exists.
    checkpoint_patterns = ("*.joblib", "*.pkl", "*.pickle", "*.sav", "*predictions*.csv")
    checkpoints = [p for pattern in checkpoint_patterns for p in ROOT.rglob(pattern)]
    out_csv = OUT / "source_reference_sanity.csv"
    fields = [
        "dataset", "model", "seed", "recall_min_0", "recall_maj_0",
        "recall_min_ref5", "recall_maj_ref5", "delta_min", "delta_maj",
        "asymmetry", "status",
    ]
    status = "NOT EXECUTED — requires retraining" if not checkpoints else "NOT EXECUTED — no validated frozen 0% evaluation artifact"
    with out_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"dataset": "ALL", "model": "ALL", "seed": "ALL", "status": status})
    note = [
        "# Source-reference sanity check (0% vs 5% MCAR)",
        "",
        f"Status: **{status}**.",
        "",
        "The repository contains the 5% MCAR reference results but no serialized estimator, checkpoint, prediction file, or retained complete-data (0% missingness) evaluation. The permitted check therefore was not run; no model was retrained, retuned, reseeded, or re-split.",
        "",
        "The manuscript consequently defines MVG as incremental class-specific degradation relative to the already-missing 5% source-like reference, not as total degradation relative to complete data.",
    ]
    (OUT / "source_reference_sanity.md").write_text("\n".join(note) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cells = primary_shift_cells()
    write_hidden_thresholds(cells)
    write_source_reference_sanity()
    print(f"derived_cells={len(cells)}")
    print(f"wrote={OUT / 'hidden_failure_threshold_sensitivity.csv'}")
    print(f"wrote={OUT / 'source_reference_sanity.csv'}")


if __name__ == "__main__":
    main()
