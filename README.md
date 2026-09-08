# Project E Gate A

**Minority-Class Robustness under Missingness Shift in Imbalanced Tabular
Classification** (中文工作题目：缺失机制变化下不平衡表格分类的少数类鲁棒性研究)

This is a **standalone** project. It does not share data, code, caches, models or
Gate conclusions with Project A/B/C/D.

## Goal

Gate A verifies whether a *shift in the test-time missingness rate or pattern*
degrades the **minority class** more than the majority class, and whether the
degradation is repeatable. The pipeline evaluates this on three independent,
publicly downloadable binary imbalanced UCI datasets.

## Datasets

| dataset | source | n | numeric | imbalance | natural missing |
|---|---|---|---:|---:|---:|
| default-of-credit-card-clients | UCI id 350 | 30000 | 23 | 1:3.5 | 0 |
| bank-additional-full | UCI 00222 (`bank-additional.zip`) | 41188 | 10 | 1:7.9 | 0 |
| HTRU2 | UCI id 372 | 17898 | 8 | 1:9.9 | 0 |

Only numeric features are subjected to the artificial missingness protocol; the
datasets have zero natural missingness so the protocol is not masked.

## Quick start

```bash
pip install -r requirements.txt
python scripts/reproduce_gate_a.py
```

The script downloads/caches the datasets, splits 60/20/20, injects source MCAR
missingness (5%) on train/validation, fits the imputer on source train, trains
XGBoost and Logistic Regression, evaluates five target missingness environments
(IID, mild MCAR, severe MCAR, MAR-like, class-conditional) plus a matched-rate
MCAR comparator, computes vulnerability metrics, fits the missingness-sensitivity
slope, renders the six figures, and runs the automatic Gate A decision
(GO / HOLD / STOP-PIVOT).

## Outputs

Everything is written under `results/gate_a/`:

- `raw_results.csv`, `missingness_profiles.csv`, `vulnerability_results.csv`,
  `slope_results.csv`
- `gate_decision.json`, `gate_report.md`, `manifest.json`
- `figures/` (six PNG figures)

## Tests

```bash
python -m pytest -q
```

The suite enforces the leakage contract (ranking / imputer on source train only,
no test labels used for parameter selection), verifies actual missingness rates
and class-conditional exposure, hand-checks the vulnerability formulas, and
exercises GO / HOLD / STOP synthetic gate cases.

## Configuration

`configs/gate_a.yaml` controls seeds, datasets, models, the missingness
environments and Gate thresholds. To limit a run:

```bash
python scripts/reproduce_gate_a.py --datasets htr2 --models xgboost --seeds 42
```

## Method

- Imputation: median for numeric, most-frequent for categorical, fitted only on
  source train, reused verbatim for every target environment.
- Pipelines: `P0` = imputation; `P1` = imputation + missingness indicators
  (XGBoost only).
- Metrics: AUPRC, AUROC, minority/majority recall, F1, precision, balanced
  accuracy, MCC, G-mean.
- Vulnerability: `MRR`, `MiRD`, `MaRD`, `MVG`, `CCEP`, `MIG`, slope gap.
- Gate A is a *diagnostic*, not a claim that missingness shift always hurts the
  minority class; it asks whether the shift produces a repeatable
  class-asymmetric failure.

---

## Phase E1/E2: Paper-Level Phenomenon Verification

Gate A established the core phenomenon on three datasets. Phase E1/E2 expands to
**eight frozen datasets**, **four model families**, and a richer set of
missingness mechanisms, then runs the statistical validation, reverse control,
mechanism analysis and the **Paper Gate** (STRONG-GO / GO / HOLD / STOP-PIVOT).
Only if the Paper Gate returns STRONG-GO or a stable GO does the project move to
Phase E3/E4 method design; **no method is implemented in this phase**.

### Scientific question

> Does missingness distribution shift cause disproportionate minority-class
> degradation in imbalanced tabular classification, even when conventional
> aggregate metrics remain relatively stable?

### Frozen dataset list

The list is frozen in `configs/datasets_phase_e12.yaml` *before* any experiment
runs (`dataset_freeze_timestamp` and `dataset_list_hash` are recorded). One
dataset may only be replaced for a documented, non-contractual cause (see
`results/phase_e12/dataset_exclusion_log.md`); it is never removed because the
results are unattractive.

| dataset | source | n | numeric | imbalance | natural missing |
|---|---|---|---:|---:|---:|
| default-of-credit-card-clients | UCI 350 | 30000 | 23 | 1:3.5 | 0 |
| bank-additional-full | UCI 00222 | 41188 | 10 | 1:7.9 | 0 |
| HTRU2 | UCI 372 | 17898 | 8 | 1:9.9 | 0 |
| spambase | UCI 94 | 4601 | 57 | 1:1.5 | 0 |
| Wisconsin Diagnostic Breast Cancer | UCI 17 | 569 | 30 | 1:1.7 | 0 |
| Magic Gamma Telescope | UCI 159 | 19020 | 10 | 1:1.8 | 0 |
| Polish Bankruptcy | UCI 365 | 43405 | 64 | 1:19.8 | 1.5% |
| Ozone | UCI 172 | 5070 | 72 | 1:20.8 | 8.2% |

### Missingness mechanisms

- MCAR: `mcar_05`, `mcar_10`, `mcar_20`, `mcar_30`, `mcar_40`.
- MAR-like: `mar_weak`, `mar_medium`, `mar_strong` (never label-dependent).
- Class-conditional stress: `cc_20`, `cc_30`, `cc_40`, `cc_50`
  (majority 10%, minority 20/30/40/50%).
- Comparable-rate MCAR: `matched_mcar_cc20/30/40/50` (CCEP control).
- Reverse class-conditional: `reverse_cc_30`, `reverse_cc_40`
  (minority 10%, majority 30/40%).
- Correlated/block: `block_mild` (~15%), `block_severe` (~30%).
- Feature-specific: `feature_specific` (declining per-feature rates).

### Models & pipelines

Logistic Regression, Random Forest, XGBoost, LightGBM. Primary pipeline `P0`
(median/mode imputation); `P1` (imputation + missingness indicators) and
`native_missing` baselines for XGBoost/LightGBM; KNN/Iterative imputation
robustness sub-matrix on three representative datasets. Fixed, sensible
default hyperparameters; no per-dataset HPO, no neural/transformer/GAN models.

### Metrics & statistical tests

Primary: AUPRC, minority recall/F1, `MVG_recall`, `MVG_f1`. Secondary: AUROC,
minority precision, majority recall/F1, balanced accuracy, MCC, G-mean.
`MiRD`, `MaRD`, `MVG`, `MRR` follow the same definitions as Gate A; `CCEP` is
the comparable-rate MCAR minus class-conditional minority metric; hidden
failure = `AUROC_drop <= 0.03` and `minority_recall_drop >= 0.10` (plus the
AUPRC variant). Statistical validation: Wilcoxon signed-rank (primary), paired
t-test, rank-biserial + Cohen's dz effect size, Benjamini-Hochberg FDR, and 95%
bootstrap CIs on `MVG`, `CCEP`, slopes and `ReverseExposureEffect`.

### Reproduce

```bash
python scripts/reproduce_phase_e12.py                      # full run
python scripts/reproduce_phase_e12.py --resume             # resume, skipping complete keys
python scripts/reproduce_phase_e12.py --smoke              # 2 datasets x 2 models x 1 seed
python scripts/reproduce_phase_e12.py --datasets htr2 spambase --models logistic_regression xgboost
```

Each unit result is checkpointed immediately under `results/phase_e12/raw/` with
a unique run key, so an interrupted run resumes without recomputation.

### Outputs

Written to `results/phase_e12/`: `raw_results.csv`, `missingness_profiles.csv`,
`vulnerability_results.csv`, `slope_results.csv`, `class_conditional_results.csv`,
`reverse_control_results.csv`, `hidden_failure_results.csv`,
`statistics_results.csv`, `dataset_summary.csv`, `model_summary.csv`,
`mechanism_summary.csv`, `imbalance_correlation.csv`, `ambg_results.csv`,
`paper_gate.json`, `phase_e12_report.md`, `manifest.json`, `progress.json`,
`failed_runs.csv`, `dataset_exclusion_log.md`, `figures/` (10 figures) and
`tables/` (8 tables).

### Paper Gate logic

`STRONG-GO` requires roughly 7 of 10 conditions: `>=5/8` datasets with positive
mean MVG, `>=4/8` datasets with `MVG >= 0.10`, `>=3` models positive, FDR-adjusted
Wilcoxon significant, moderate effect size, `>=5/8` steeper minority slopes,
`>=3` datasets with `CCEP >= 0.10`, `>=3` datasets with hidden failure, `>=4`
datasets where reverse CC materially reduces MVG, and effect > seed noise.
`GO` holds when MVG is broadly positive and minority slopes are steeper but the
class-conditional / hidden-failure conditions are only partially met. `HOLD`
applies when the phenomenon is concentrated in a few datasets or model-dependent.
`STOP-PIVOT` applies when MVG is near zero across most datasets or the effect is
at or below seed noise. **No method is proposed unless the gate is at least a
stable GO.**

---

## Phase E3: MAMR Method Gate

Phase E3 asks whether a lightweight, model-agnostic robustification can mitigate
the exposure-driven minority recall collapse under missingness shift **beyond
standard imbalance treatments**, without materially sacrificing majority or IID
performance.

### Method

**MAMR** = Minority-Aware Missingness Robustification, a training-side injection
with two components:

1. **Exposure-aware minority weighting**: `w_i = w_class(y_i) * (1 + lambda * e_i)`
   for the minority class, where `e_i` is a train-side missingness exposure
   proxy built from mutual-information feature importance and the clipped,
   normalised standardised feature magnitude (`r_ij`). The proxy is normalised to
   unit mean over the minority so its average weight intensity matches a uniform
   `e_i = 1` control.
2. **Minority-focused mild missingness augmentation**: a corrupted copy of a
   fraction of the minority training rows (default `copy_fraction=0.5`,
   `corruption_rate=0.15`, mechanism mix 40% MCAR / 30% feature-specific / 30%
   block) is appended to the source train, which is always 100% retained.

### Methods & matrix

8 methods (`erm`, `class_weight`, `random_oversampling`, `missing_indicator`,
`augmentation_only`, `exposure_weighting_only`, `uniform_minority_weight`,
`mamr_full`) x 3 datasets (`default_credit_card`, `bank_additional`, `htr2`) x 2
models (LR, XGBoost) x 3 seeds x 8 test environments = **1152 runs**. The split
is 60/20/20 stratified and identical across methods; source missingness is MCAR
5% on train/validation; model hyperparameters are fixed (no MAMR-specific HPO).

### Hyperparameter gate

Only MAMR parameters are tuned, on validation only, in two steps: fix
`copy_fraction=0.5`, `corruption_rate=0.15` and select
`lambda in {0.25, 0.5, 1.0, 2.0}`; then fix the chosen lambda and select
`copy_fraction in {0.25, 0.5}` x `corruption_rate in {0.10, 0.15, 0.20}`. The
selection objective is
`Utility = Recall_minority - 0.5 max(0, Recall_majority_base - Recall_majority)
- 0.5 max(0, AUPRC_base - AUPRC)` (base = ERM on validation).

### Core metrics

Primary: minority recall, majority recall, AUPRC. Secondary: minority F1, AUROC,
balanced accuracy, MCC. Robustness: MiRD/MaRD/MVG recall, recall & AUPRC
retention. Plus **LRR** (Lost Recall Recovery), **MTO** (Majority Trade-off),
**IID Cost**, **BIG** (Beyond-Imbalance Gain vs the best of `class_weight` /
`random_oversampling` / `missing_indicator`) and **MAS** (Mechanism Alignment
Score).

### Method Gate and result

The automatic Method Gate answers the 8 GO conditions (minority robustness gain,
collapse recovery, no majority sacrifice, no AUPRC collapse, IID safety, beyond
standard imbalance treatment, mechanism alignment, seed consistency).

**Result: `HOLD`** (7/8 conditions, `GO-C` false). MAMR gives large, significant
minority recall gains (`+0.151` vs best standard, Wilcoxon / FDR `p < 0.001`,
rank-biserial 1.0) and improves AUPRC (`+0.013`) and IID minority recall on all
three datasets (BIG positive fraction 1.0). However, it achieves this by
substantially sacrificing majority recall (`-0.158` mean, `-0.348` worst on
`default_credit_card`), which is a large majority tradeoff and is unstable across
datasets. The gate therefore does not admit Phase E4 and instead recommends
revising the MAMR weighting / augmentation strength.

### Reproduce

```bash
python scripts/run_phase_e3_mamr.py --smoke            # smoke
python scripts/reproduce_phase_e3_mamr.py              # full Method Gate
python scripts/reproduce_phase_e3_mamr.py --resume     # resume
```

Outputs are written to `results/phase_e3_mamr/`: `raw_results.csv`,
`collapse_recovery.csv`, `beyond_imbalance_gain.csv`, `iid_cost.csv`,
`majority_tradeoff.csv`, `mechanism_alignment.csv`, `ablation_results.csv`,
`statistics_results.csv`, `exposure_audit.csv`, `seed_consistency.csv`,
`hyperparameter_results.csv`, `method_gate.json`, `method_gate_report.md`,
`manifest.json`, `selected_mamr_params.json`, `figures/` (9 figures) and other
summaries. Each unit is checkpointed under `results/phase_e3_mamr/raw/`.

The frozen config is `configs/phase_e3_mamr.yaml`. This phase does **not** modify
`results/gate_a/` or `results/phase_e12/`.

## Phase E3R: MAMR Safety Repair Gate

Phase E3 returned `HOLD` (7/8, `GO-C` false) because the MAMR minority gain was
paid for with a large majority-recall sacrifice. Phase E3R is the single
pre-registered **conservative repair**: it keeps the MAMR components but moves to
a weaker operating point (smaller `lambda`, `copy_fraction`, `corruption_rate`)
so the majority cost drops while most of the minority gain is retained.

### Search and safety utility

Only weaker settings are explored: Step A
`lambda in {0.0, 0.10, 0.25, 0.50, 0.75}` at `copy=0.15, corr=0.10`; Step B
`copy in {0.10, 0.15, 0.20, 0.25}` x `corr in {0.05, 0.10, 0.15}`. Hard caps:
`lambda<=0.75`, `copy<=0.25`, `corr<=0.15`. Selection is validation-only with
`U_safe = R_min - 2.0*max(0,R_maj_ref-R_maj) - 1.0*max(0,AUPRC_ref-AUPRC)`,
reference = best_standard on a source-validation **stress suite** (mcar 20/30,
feature-specific mild, block mild, class-conditional mild), plus hard
validation constraints `majority delta >= -0.03`, `AUPRC delta >= -0.02`.

### Result

The global conservative config selected `lambda=0.0, copy_fraction=0.10,
corruption_rate=0.05` (the most conservative corner, validation feasibility
0.833). On the frozen test environments it reduced the majority-recall cost from
`-0.158` (mean) to `-0.027`, i.e. **Majority Harm Reduction ≈ 0.83**; but it also
dropped the minority-recall gain vs best_standard from `+0.151` to `+0.040`
(**Minority Gain Retention ≈ 0.265**). R1 / R2 / R3 all fail (BIG mean 0.035,
MGR 0.265, majority worst `-0.062`), so the Safety Gate returns **`STOP`**: the
conservative repair cannot keep the MAMR minority benefit while clearing the
pre-registered majority-safety constraint. This is a terminal decision for the
MAMR method line; the E1/E2 phenomenon / mechanism conclusions are retained.

### Reproduce

```bash
python scripts/run_phase_e3r_mamr_safety.py --smoke            # smoke
python scripts/reproduce_phase_e3r_mamr_safety.py              # full Safety Gate
python scripts/reproduce_phase_e3r_mamr_safety.py --resume     # resume
```

Outputs are written to `results/phase_e3r_mamr_safety/`: `pre_registration.md`,
`validation_safety_search.csv`, `selected_safe_params.json`,
`global_safe_config.json`, `raw_safe_results.csv`,
`safe_method_comparison.csv`, `minority_gain_retention.csv`,
`majority_harm_reduction.csv`, `pareto_results.csv`, `collapse_recovery.csv`,
`beyond_imbalance_gain.csv`, `mechanism_alignment.csv`, `iid_cost.csv`,
`statistics_results.csv`, `default_credit_card_safety_diagnostic.csv`,
`gate_conditions.csv`, `safety_gate.json`, `safety_gate_report.md`,
`manifest.json`, `progress.json`, `figures/` (6 figures) and other summaries.

The frozen config is `configs/phase_e3r_mamr_safety.yaml`. This phase does **not**
modify `results/gate_a/`, `results/phase_e12/`, or `results/phase_e3_mamr/`.

## Phase E-Final-RC: Reviewer-Proof Supplementary Validation

Phase E-Final-RC is the post-submission reviewer-proof supplement. It **purely
re-aggregates** the frozen E1/E2, E3 and E3R artifacts; no model is trained and
no new method is introduced. It runs 12 reviewer checks (RC1-RC12) against the
authoritative E1/E2 definitions and writes a `FROZEN` manifest. After this phase
the experiment is locked: `EXPERIMENT STATUS = FROZEN`.

### Frozen study design

- 8 datasets, 4 models (logistic-regression, random-forest, xgboost, lightgbm),
  seeds 42/52/62.
- Primary analysis matrix: `pipeline == p0`, `imputer == median`,
  `feature_set == top` (1,392 cells in `results/phase_e12/raw_results.csv`).
- Definitions unchanged from E1/E2: MiRD/MaRD = relative degradation vs the
  `mcar_05` baseline; MVG = `MiRD - MaRD`; CCEP = matched-MCAR recall - CC recall;
  reverse-exposure effect = `normal_cc_MVG - reverse_cc_MVG`; hidden failure =
  `AUROC drop <= 0.03 AND minority recall drop >= 0.10`.

### Reviewer checks and verdicts

| Check | Verdict | Basis |
| --- | --- | --- |
| RC1 metric sensitivity | PASS | AUPRC relative drop (0.036) > AUROC relative drop (0.011): aggregate metrics understate minority harm |
| RC2 hidden-failure threshold | PASS | >= 3 definitions with hidden failures in >= 2 datasets |
| RC3 leave-one-dataset-out | PASS | pooled MVG > 0 and slope gap > 0 for every exclusion |
| RC4 leave-one-model-out | PASS | pooled MVG > 0 and slope gap > 0 for every exclusion |
| RC5 seed sensitivity | PASS | single-seed and pooled-seed mean MVG all > 0 |
| RC6 severity exclusion | PASS | slope gap > 0 for `moderate_5_30` |
| RC7 class-conditional exposure (CCEP) | PASS | positive fraction 0.84, mean CCEP 0.031 |
| RC8 reverse-exposure control | PASS | > 60% of cells have positive reverse-exposure effect |
| RC9 natural missingness | DESCRIPTIVE | natural-missing vs zero-natural-missing groups reported |
| RC10 imputation sensitivity | PASS | all non-P0 imputation configs keep positive mean MVG |
| RC11 aggregate masking | QUANTIFIED | Spearman recall-vs-AUROC 0.58; fraction recall>=0.10 & AUROC<=0.03 = 0.048 |
| RC12 MAMR negative result | NEGATIVE_RESULT | E3 `HOLD` and E3R `STOP` retained as honest negatives |

### Reproduce

```bash
python scripts/reproduce_phase_e_final_rc.py
```

Outputs are written to `results/phase_e_final_rc/`: `metric_sensitivity.csv`,
`hidden_failure_threshold_sensitivity.csv`, `lodo_dataset_influence.csv`,
`lomo_model_influence.csv`, `seed_sensitivity.csv`, `severity_exclusion.csv`,
`ccep_reviewer_check.csv`, `reverse_cc_reviewer_check.csv`,
`natural_missingness_sensitivity.csv`, `imputation_sensitivity.csv`,
`aggregate_masking_quantification.csv`, `mamr_negative_result_summary.csv`,
`phase_e_final_rc_report.md`, `manifest.json`, `progress.json`, and `figures/`
(5 reviewer figures).

This phase does **not** modify `results/gate_a/`, `results/phase_e12/`,
`results/phase_e3_mamr/`, or `results/phase_e3r_mamr_safety/`. The MAMR method
line remains terminated at E3R (`STOP`); no MAMR-v2/v3, no new data, no new
feature weighting/augmentation, and no Phase E4 are run.
