# Missingness-Shift Minority Robustness

Reproducible study of minority-class robustness under missingness distribution
shift in imbalanced tabular classification.

## Overview

Imbalanced tabular classifiers are usually trained assuming the training
missingness pattern persists at deployment. This pipeline tests whether a shift
in the deployment-time missingness **rate** or **mechanism** degrades the
**minority class** more than the majority, and whether the degradation is
repeatable, even when conventional aggregate metrics (e.g. AUROC) remain
relatively stable. It is a diagnostic / robustness study and does not propose a
new classifier or imputer.

Three reproducible phases:

| Phase | Scope | Purpose |
| --- | --- | --- |
| Gate A | 3 datasets, 2 models | Foundational check of the class-asymmetric phenomenon |
| Phase E1/E2 | 8 frozen datasets, 4 models | Paper-level phenomenon verification, statistics, reverse control, mechanism analysis, automatic Paper Gate |
| Phase E-Final-RC | frozen artifacts | Additional reviewer-style validation checks (RC1-RC12) |

## Requirements

```bash
pip install -r requirements.txt
```

Python 3.12 recommended. Packages: `numpy`, `pandas`, `scikit-learn`, `xgboost`,
`matplotlib`, `seaborn`, `pyyaml`, `ucimlrepo`, `pytest`.

## Repository layout

```text
configs/   dataset + phase configuration (yaml)
src/       pipeline modules
scripts/   one-shot reproduction entry points
tests/     unit tests
```

## Datasets

The frozen experiment list is stored in `configs/datasets_phase_e12.yaml` and
recorded before any run (via `dataset_freeze_timestamp` and
`dataset_list_hash`). A dataset is only replaced for a documented,
non-contractual cause; it is never removed because the results are unattractive.

| dataset | source | n | numeric | imbalance | natural missing |
|---|---|---|---:|---:|---:|
| default-of-credit-card-clients | UCI 350 | 30000 | 23 | 1:3.5 | 0% |
| bank-additional-full | UCI 00222 | 41188 | 10 | 1:7.9 | 0% |
| HTRU2 | UCI 372 | 17898 | 8 | 1:9.9 | 0% |
| spambase | UCI 94 | 4601 | 57 | 1:1.5 | 0% |
| Wisconsin Diagnostic Breast Cancer | UCI 17 | 569 | 30 | 1:1.7 | 0% |
| Magic Gamma Telescope | UCI 159 | 19020 | 10 | 1:1.8 | 0% |
| Polish Bankruptcy | UCI 365 | 43405 | 64 | 1:19.8 | 1.5% |
| Ozone | UCI 172 | 5070 | 72 | 1:20.8 | 8.2% |

Gate A uses the first three datasets with a 60/20/20 stratified split.

## Missingness environments

- MCAR: `mcar_05`, `mcar_10`, `mcar_20`, `mcar_30`, `mcar_40`.
- MAR-like: `mar_weak`, `mar_medium`, `mar_strong` (never label-dependent).
- Class-conditional stress: `cc_20`, `cc_30`, `cc_40`, `cc_50` (majority 10%,
  minority 20/30/40/50%).
- Comparable-rate MCAR: `matched_mcar_cc20/30/40/50` (CCEP control).
- Reverse class-conditional: `reverse_cc_30`, `reverse_cc_40` (minority 10%,
  majority 30/40%).
- Correlated / block: `block_mild` (~15%), `block_severe` (~30%).
- Feature-specific: `feature_specific` (declining per-feature rates).

Only numeric features are exposed to the artificial missingness protocol.

## Models & pipelines

Logistic Regression, Random Forest, XGBoost, LightGBM. Primary pipeline `P0`
(median / mode imputation); `P1` (imputation + missingness indicators) and
`native_missing` baselines for XGBoost / LightGBM; KNN / iterative imputation
robustness sub-matrix on three representative datasets. Fixed default
hyperparameters; no per-dataset HPO; no neural / transformer / GAN models.

## Metrics & vulnerability definitions

- Primary: AUPRC, minority recall / F1, `MVG_recall`, `MVG_f1`.
- Secondary: AUROC, minority precision, majority recall / F1, balanced accuracy,
  MCC, G-mean.
- `MiRD` / `MaRD`: relative degradation of minority / majority metric vs the
  `mcar_05` baseline.
- `MVG` = `MiRD - MaRD` (positive ⇒ minority harmed more than majority).
- `MRR`: minority retention ratio.
- `CCEP`: comparable-rate MCAR metric minus class-conditional minority metric
  (matched rate ⇒ same overall missingness ≠ same minority risk).
- `MIG`: missingness-induced gap; the slope gap comes from the
  missingness-sensitivity slope.
- Hidden failure: `AUROC drop <= 0.03` **and** `minority_recall drop >= 0.10`
  (plus an AUPRC variant).

## Statistical validation

Wilcoxon signed-rank (primary, after aggregation to 96 dataset × model × seed
blocks), paired t-test, rank-biserial and Cohen's dz effect sizes,
Benjamini-Hochberg FDR, and 95% block-bootstrap CIs on `MVG`, `CCEP`, slopes
and the reverse-exposure effect. The 1,296-cell result table is descriptive.

## Reproduce

```bash
# Gate A (foundation)
python scripts/reproduce_gate_a.py
python scripts/reproduce_gate_a.py --datasets htr2 --models xgboost --seeds 42

# Phase E1/E2 (main phenomenon)
python scripts/reproduce_phase_e12.py                       # full run
python scripts/reproduce_phase_e12.py --resume              # resume skipped keys
python scripts/reproduce_phase_e12.py --smoke               # 2 datasets x 2 models x 1 seed
python scripts/reproduce_phase_e12.py --datasets htr2 spambase --models logistic_regression xgboost

# Phase E-Final-RC (additional validation)
python scripts/reproduce_phase_e_final_rc.py
```

Datasets are downloaded and cached automatically. Results are written under
`results/`, which is gitignored and regenerated by the scripts above.

## Tests

```bash
python -m pytest -q
```

The suite enforces the leakage contract (ranking / imputer fitted on source
train only, no test labels used for parameter selection), checks actual
missingness rates and class-conditional exposure, verifies the vulnerability
formulas, and exercises the GO / HOLD / STOP synthetic gate cases.

## Paper Gate logic

`STRONG-GO` requires roughly 7 of 10 conditions: `>=5/8` datasets with positive
mean MVG, `>=4/8` datasets with `MVG >= 0.10`, `>=3` models positive,
FDR-adjusted Wilcoxon significant, moderate effect size, `>=5/8` steeper
minority slopes, `>=3` datasets with `CCEP >= 0.10`, `>=3` datasets with hidden
failure, `>=4` datasets where reverse class-conditional materially reduces MVG,
and effect greater than seed noise. `GO` holds when MVG is broadly positive and
minority slopes are steeper but the class-conditional / hidden-failure conditions
are only partially met; `HOLD` applies when the phenomenon is concentrated in a
few datasets or is model-dependent; `STOP-PIVOT` applies when MVG is near zero
across most datasets or the effect is at / below seed noise. The decision is
produced by `src/paper_gate.py`.

## Configuration

`configs/*.yaml` control seeds, datasets, models, missingness environments,
thresholds and run scope. `configs/gate_a.yaml`, `configs/phase_e12.yaml` and
`configs/datasets_phase_e12.yaml` are used by the corresponding reproduction
scripts.
