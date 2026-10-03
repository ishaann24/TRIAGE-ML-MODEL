# -*- coding: utf-8 -*-
"""
KTAS Triage Pipeline (v3 — FINAL/TUNED)
==========================================

Adds hyperparameter tuning + 5-fold cross-validation to the best-performing
setup from v2 (Random Forest / XGBoost with vitals + chief-complaint text),
for the same reason as the MIMIC v4 script: a single train/test split can
be misleadingly optimistic or pessimistic, especially to check whether
this holds up better than it did on the smaller MIMIC dataset.
"""

import os
import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    print("[WARNING] xgboost not installed.")

# ---------------------------------------------------------------
# 1. LOAD + CLEAN (same as v2)
# ---------------------------------------------------------------
DATA_PATH = "data.csv"
if not os.path.exists(DATA_PATH):
    sys.exit(f"[ERROR] Could not find '{DATA_PATH}'.")

df = pd.read_csv(DATA_PATH, sep=";", encoding="latin1")
print("Shape:", df.shape)

NUMERIC_COLS = ["Age", "Sex", "Arrival mode", "Injury", "Mental", "Pain",
                "NRS_pain", "SBP", "DBP", "HR", "RR", "BT", "Saturation"]
BASELINE_COL = "KTAS_RN"
TARGET_COL = "KTAS_expert"
TEXT_COL = "Chief_complain"

from sklearn.impute import SimpleImputer

X_num_raw = df[NUMERIC_COLS].apply(
    lambda col: pd.to_numeric(col.astype(str).str.replace(",", ".", regex=False), errors="coerce")
)
imputer = SimpleImputer(strategy="median")
X_num = pd.DataFrame(imputer.fit_transform(X_num_raw), columns=NUMERIC_COLS)

y_raw = pd.to_numeric(df[TARGET_COL], errors="coerce")
nurse_baseline_raw = pd.to_numeric(df[BASELINE_COL], errors="coerce")
chief_complaint_raw = df[TEXT_COL].fillna("").astype(str).str.lower()

valid_rows = ~y_raw.isna()
X_num = X_num[valid_rows.values].reset_index(drop=True)
y_raw = y_raw[valid_rows].reset_index(drop=True)
nurse_baseline_raw = nurse_baseline_raw[valid_rows].reset_index(drop=True)
chief_complaint_raw = chief_complaint_raw[valid_rows.values].reset_index(drop=True)

y = (y_raw - 1).astype(int).values  # 0-indexed for XGBoost

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

tfidf = TfidfVectorizer(max_features=200, stop_words="english", min_df=2)
text_all_tfidf = tfidf.fit_transform(chief_complaint_raw)
X_combined_all = hstack([csr_matrix(X_num.values), text_all_tfidf]).tocsr()

print(f"Combined feature matrix shape (vitals + text): {X_combined_all.shape}")

# Nurse baseline (whole-dataset, for reference)
valid_nurse = ~np.isnan(nurse_baseline_raw.values)
nurse_exact_match = (nurse_baseline_raw.values[valid_nurse] == y_raw.values[valid_nurse]).mean()
print(f"\nNurse's own score exact-match accuracy (whole dataset): {nurse_exact_match:.4f}")

# ---------------------------------------------------------------
# 2. HYPERPARAMETER TUNING (XGBoost, multi-class, vitals+text)
# ---------------------------------------------------------------
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score
)

idx_train, idx_test = train_test_split(
    np.arange(len(y)), test_size=0.25, random_state=SEED, stratify=y
)
X_train, X_test = X_combined_all[idx_train], X_combined_all[idx_test]
y_train, y_test = y[idx_train], y[idx_test]

results_summary = []

if XGB_AVAILABLE:
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING — XGBoost (vitals + text, RandomizedSearchCV)")
    print("=" * 70)

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    base_xgb = XGBClassifier(
        objective="multi:softprob", num_class=5,
        random_state=SEED, eval_metric="mlogloss"
    )
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        base_xgb, param_distributions=param_dist, n_iter=40,
        scoring="accuracy", cv=cv_strategy, random_state=SEED, n_jobs=-1, verbose=0
    )
    search.fit(X_train, y_train)

    print(f"\nBest cross-validated accuracy during search: {search.best_score_:.4f}")
    print("Best hyperparameters:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    xgb_tuned = search.best_estimator_
    test_pred = xgb_tuned.predict(X_test)
    test_acc = (test_pred == y_test).mean()
    diff = (test_pred + 1) - (y_test + 1)
    undertriage = (diff > 0).mean()
    overtriage = (diff < 0).mean()

    print(f"\n===== XGBoost (TUNED, vitals + text) — held-out test set =====")
    print(f"Exact-match accuracy: {test_acc:.4f}")
    print(f"Undertriage rate: {undertriage:.4f}   Overtriage rate: {overtriage:.4f}")

    print("\n" + "=" * 70)
    print("FINAL CROSS-VALIDATED PERFORMANCE (tuned model, full dataset)")
    print("=" * 70)
    cv_scores = cross_val_score(
        xgb_tuned, X_combined_all, y,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="accuracy"
    )
    print(f"Exact-match accuracy per fold: {np.round(cv_scores, 4)}")
    print(f"Mean accuracy: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")
    print(f"\nFor comparison — nurse's own score exact-match accuracy: {nurse_exact_match:.4f}")

    results_summary.append({
        "name": "XGBoost (tuned, vitals+text) — single split", "accuracy": test_acc
    })
    results_summary.append({
        "name": "XGBoost (tuned, vitals+text) — 5-fold CV mean", "accuracy": cv_scores.mean(),
        "std": cv_scores.std()
    })
    results_summary.append({
        "name": "Nurse's own score (whole dataset)", "accuracy": nurse_exact_match
    })

results_df = pd.DataFrame(results_summary)
print("\n===== FINAL SUMMARY (KTAS) =====")
print(results_df.to_string(index=False))
results_df.to_csv("ktas_final_tuned_results.csv", index=False)
print("\nSaved: ktas_final_tuned_results.csv")
