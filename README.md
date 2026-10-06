# Triage ML Model

AI-based emergency department (ED) triage prioritization — an explainable, rigorously evaluated machine learning pipeline developed as an academic research project.

This repository contains the full source code, datasets, and generated results for a study comparing machine learning-based triage prediction against existing clinical triage baselines (ESI and nurse-assigned KTAS scores), using nested cross-validation, bootstrap significance testing, and SHAP-based explainability across two independent datasets.

## Overview

Manual ED triage is subject to inconsistency arising from clinical workload, time pressure, and the subjectivity of rapid assessment. This project evaluates whether conventional machine learning models — Logistic Regression, Random Forest, and XGBoost — can predict triage urgency from structured vitals and chief-complaint text, and whether any such model can match or exceed existing clinical judgment.

Two independent, publicly available datasets were used:

* **MIMIC-IV-ED Demo** (n = 222) — binary critical-illness prediction, benchmarked against the Emergency Severity Index (ESI).
* **KTAS** (n = 1,267) — five-class urgency prediction (Korean Triage and Acuity Scale), incorporating chief-complaint text via TF-IDF, benchmarked against the nurse's own initial triage score.

In both pipelines, the existing clinical triage score is **deliberately excluded from model inputs** and used only as a comparison baseline, to avoid the model trivially reproducing an existing clinical judgment rather than learning from raw patient data.

## Key Results

| Dataset     | Best Model          | Mean Performance (nested 5-fold CV) | Clinical Baseline      | Statistically Significant?                       |
| ----------- | ------------------- | ----------------------------------- | ---------------------- | ------------------------------------------------ |
| MIMIC-IV-ED | Logistic Regression | AUC 0.619 ± 0.084                   | ESI: AUC 0.683         | No (p = 0.272)                                   |
| KTAS        | Random Forest       | Accuracy 0.681 ± 0.023              | Nurse (KTAS_RN): 0.853 | Yes (p < 0.0001) — baseline significantly better |

Full results, confidence intervals, confusion matrices, ROC curves, and SHAP explainability plots are included in this repository.

## Methodology Highlights

* **Nested cross-validation** (5 outer folds for performance estimation, 3 inner folds for hyperparameter search) to avoid information leakage between model selection and performance evaluation.
* **Equal hyperparameter tuning** applied to all three candidate models, for a fair comparison.
* **Bootstrap resampling** (2,000 iterations) to compute 95% confidence intervals and test statistical significance against each dataset's clinical baseline.
* **SHAP (TreeExplainer)** for both global feature importance and per-patient explanations, applied to the best-performing model on each dataset.
* **Structured error analysis** of undertriaged cases, identifying specific clinical patterns associated with model failure.
* **Fold-wise preprocessing** (median imputation, TF-IDF vectorization) to prevent any leakage from test data into training.

Full methodological detail is provided in the accompanying research paper.

## Repository Structure

```text
├── data.csv                              # KTAS dataset (semicolon-delimited, Latin-1 encoded)
├── mimic_ed_demo_prepared.csv            # MIMIC-IV-ED Demo dataset
│
├── mimic_iv_ed_triage_v5_rigorous.py     # MIMIC-IV-ED pipeline: nested CV, bootstrap testing,
│                                          #   SHAP, error analysis, figure generation
├── ktas_triage_v4_rigorous.py            # KTAS pipeline: same methodology, 5-class target,
│                                          #   vitals + TF-IDF text features
│
├── mimic_nested_cv_results.csv           # Per-model nested CV scores (MIMIC-IV-ED)
├── ktas_nested_cv_results.csv            # Per-model nested CV scores (KTAS)
├── mimic_all_oof_predictions.csv         # Raw out-of-fold predictions, all models (MIMIC-IV-ED)
├── ktas_all_oof_predictions.csv          # Raw out-of-fold predictions, all models (KTAS)
├── mimic_undertriaged_cases.csv          # Full list of undertriaged patients (MIMIC-IV-ED)
├── ktas_undertriaged_cases.csv           # Full list of undertriaged patients (KTAS)
│
├── mimic_confusion_matrix.png
├── mimic_roc_curve.png
├── mimic_model_comparison_barchart.png
├── mimic_shap_summary_rigorous.png
├── ktas_confusion_matrix.png
├── ktas_confusion_matrix_nurse.png
├── ktas_model_comparison_barchart.png
├── ktas_shap_summary_rigorous.png
│
└── requirements.txt                      # Pinned dependency versions
```

## Requirements

```text
pandas
numpy
scikit-learn
xgboost
shap
matplotlib
scipy
```

Install via:

```bash
pip install -r requirements.txt
```

**Note:** TensorFlow is not required. An earlier version of this pipeline included an optional neural network comparison, which is skipped automatically if TensorFlow is unavailable (e.g., on newer Python versions without current TensorFlow support).

## Running the Pipelines

```bash
python mimic_iv_ed_triage_v5_rigorous.py
python ktas_triage_v4_rigorous.py
```

Each script is self-contained: it loads its respective dataset from the current directory, runs the full nested cross-validation and evaluation procedure, and saves all result tables and figures listed above.

The KTAS pipeline takes noticeably longer to run (approximately 15–20 minutes) due to per-fold TF-IDF refitting and multi-class hyperparameter search on a larger sample size.

All random seeds are fixed (seed = 42) for reproducibility — running either script should reproduce the results reported in this repository and in the accompanying paper exactly, given the same library versions.

## Datasets

* **MIMIC-IV-ED Demo** — a publicly accessible demonstration subset of the MIMIC-IV-ED database, requiring no credentialing. Full-scale access to MIMIC-IV-ED requires PhysioNet credentialing and was outside the scope of this study.
* **KTAS** — derived from the original Korean Triage and Acuity Scale accuracy study, obtained via a public Kaggle dataset.

Neither dataset is redistributed here under modification beyond standard cleaning for pipeline compatibility; original sources are credited in the research documentation.

## Project Status

This repository contains the implementation, evaluation results, and supporting figures for the academic study.

The accompanying research paper is currently being finalized.
