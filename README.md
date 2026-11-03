# Feature Selection and Class Imbalance Handling for Software Defect Prediction

Code, data and results for the paper

> **An Empirical Comparison of Feature Selection and Class Imbalance Handling Techniques for Machine Learning-Based Software Defect Prediction** (Zhang Haoyu and Wang Mei; revised version submitted to *IET Software*).

## Overview

A controlled full-factorial study of three software-defect-prediction (SDP) pipeline decisions:

| Factor | Levels |
|---|---|
| Feature selection (FS) | none, Information Gain (IG), ANOVA F-test (AF) — 50% of the features retained |
| Imbalance handling (IMB) | none, SMOTE, ADASYN, cost-sensitive weighting (CS), threshold moving (THR) |
| Classifier (CLF) | Logistic Regression, RBF-SVM, Random Forest, XGBoost |

3 × 5 × 4 = **60 pipelines**, evaluated on **24 datasets** with **10 × 10 repeated stratified cross-validation** (1,440 dataset–pipeline configurations, 144,000 fold-level results). All data-dependent steps (standardisation, feature selection, resampling, class-weight computation, threshold selection) are fitted inside the training folds only.

Additional analyses: decision-theoretic mechanism analysis of imbalance handling, nested cross-validation for configuration selection, hyperparameter tuning, sensitivity analysis (SMOTE/ADASYN k and sampling ratio, cost-sensitive weight, five feature selectors × three retention levels) and leave-one-project-out cross-project prediction.

## Data

| Folder | Datasets | Source |
|---|---|---|
| `data/` | 12 NASA MDP datasets (CM1, JM1, KC1, KC3, MC1, MC2, MW1, PC1–PC5), D'' cleaned version | PROMISE repository; Shepperd et al. (2013), *IEEE TSE* 39(9):1208–1215 |
| `data_ext/AEEEM/` | 5 AEEEM systems (EQ, JDT, Lucene, Mylyn, PDE) | D'Ambros et al. (2012), *Empirical Software Engineering* 17:531–577 |
| `data_ext/JIRA/` | 7 JIRA releases (ActiveMQ 5.0.0, Derby 10.5.1.1, Groovy 1.6 beta 1, HBase 0.94.0, Hive 0.9.0, JRuby 1.1, Wicket 1.3.0 beta 2) | Yatish et al. (2019), ICSE 2019 |

The AEEEM and JIRA ARFF files are redistributed from Tong, H. (2019) *AEEEM-JIRA-PROMISE* (version 1.0), Zenodo, https://doi.org/10.5281/zenodo.3362613 (CC BY 4.0).

Preprocessing (done in the scripts, not in the files): the label is binarised (`buggy`/`Y` or defect count > 0 → 1); for the AEEEM and JIRA datasets exact duplicate rows are removed and constant columns are dropped. Dataset characteristics after preprocessing are in `results_v3/datasets_summary.csv`.

## Code

| Script | Purpose | Output | Runtime* |
|---|---|---|---|
| `run_experiment_v3.py` | Main factorial experiment (60 pipelines × 24 datasets, 10 × 10 CV); saves fold-level metrics and out-of-fold probabilities | `results_v3/all_folds.csv`, `results_v3/oof/`, `results_v3/datasets_summary.csv`, `results_v3/run_info.json` | ~50 min |
| `analyze_v3.py` | Marginal means, Wilcoxon + Holm, matched-pairs rank-biserial r, bootstrap CIs, Bayesian signed-rank test with ROPE, blocked factorial ANOVA (η²), efficiency | `results_v3/analysis/` | ~2 min |
| `analyze_mechanism.py` | Mechanism of imbalance handling: prior-threshold rule, decision agreement, rank correlation, calibration (uses the saved OOF probabilities) | `results_v3/mechanism/` | ~3 min |
| `run_sensitivity.py` | Sensitivity analysis (part A: SMOTE/ADASYN k × sampling ratio, CS weight; part B: IG/AF/mRMR/L1/RF-importance × 25/50/75% retention), 5 × 10 CV | `results_sensitivity/` | ~1.5 h |
| `run_nested.py --part select` | Nested CV selection of the best of 60 pipelines vs post-hoc selection | `results_nested/select/`, `results_nested/select_summary.csv` | ~40 min |
| `run_nested.py --part tune` | Nested randomized-search hyperparameter tuning vs defaults | `results_nested/tune/`, `results_nested/tune_*.csv` | ~30 min |
| `run_cpdp.py` | Leave-one-project-out cross-project prediction within AEEEM and within JIRA | `results_cpdp/` | ~20 min |
| `make_revision_assets.py` | Builds all tables and figures reported in the paper from the result files | `revision_assets/tables/`, `revision_assets/figures/` | <1 min |

\* Apple M1 Ultra, 18 worker processes.

All experiment scripts run in parallel (`--jobs`, default = CPU cores − 2), save results per dataset and skip completed datasets when re-run (`--overwrite` forces a re-run). Each script has a `--quick` smoke-test mode.

## Usage

```bash
pip install -r requirements.txt

python run_experiment_v3.py            # main experiment
python analyze_v3.py                   # statistics for RQ1–RQ3
python analyze_mechanism.py            # mechanism analysis (RQ4)
python run_cpdp.py                     # cross-project prediction (RQ4)
python run_sensitivity.py              # sensitivity analysis
python run_nested.py --part select     # nested configuration selection
python run_nested.py --part tune       # hyperparameter tuning
python make_revision_assets.py         # tables and figures
```

## Results included

All result files used in the paper are included, so the statistics, tables and figures can be regenerated without re-running the experiments:

- `results_v3/all_folds.csv` — 144,000 fold-level results (dataset, pipeline, repeat, fold, precision, recall, F1, MCC, AUC-ROC, AUC-PR, G-mean, PF, balanced accuracy, Brier score, threshold, fit/predict time, model size)
- `results_v3/oof/*.npz` — out-of-fold predicted probabilities and decisions for every pipeline and repeat, with fold assignments and labels
- `results_v3/analysis/`, `results_v3/mechanism/` — analysis outputs
- `results_sensitivity/folds/`, `results_nested/`, `results_cpdp/` — additional experiments
- `revision_assets/` — tables and figures as reported in the paper

`results_sensitivity/all_folds.csv` is not included (size); it is rebuilt from `results_sensitivity/folds/` when `run_sensitivity.py` is run.

The `legacy_v2/` folder contains the scripts and results of the originally submitted version (48 pipelines, 12 NASA datasets, single 10-fold CV) for traceability.

## Reproducibility notes

- Seeds: models use `42 + r` and CV splits use `1000 + r` for repetition `r`; the sensitivity analysis uses the same splits as repetitions 0–4 of the main experiment.
- Software versions used for the reported results are recorded in `results_v3/run_info.json` (Python 3.12.0, scikit-learn 1.9.0, imbalanced-learn 0.14.2, XGBoost 3.4.1, NumPy 2.5.2, pandas 3.0.5, SciPy 1.18.0; macOS 26.6.2).
- ADASYN cannot generate samples for the nearly balanced EQ dataset; in that case the scripts fall back to SMOTE and record this in the `note` column.

## Citation

Full citation to be added upon publication.

## License

Code: MIT. Datasets: original licences of their sources (PROMISE; AEEEM-JIRA-PROMISE Zenodo record, CC BY 4.0).
