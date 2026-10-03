# -*- coding: utf-8 -*-
"""
MIMIC-IV-ED Triage Pipeline (v4 — FINAL/TUNED)
=================================================

WHAT CHANGED FROM v3:
v3 used default hyperparameters and a single train/test split. This is
fine for an early check, but not defensible as a "final" result — default
settings are rarely the best a model can do, and a single split means the
reported number could be somewhat lucky or unlucky.

This version adds:
1. HYPERPARAMETER TUNING for XGBoost (the best-performing model) using
   RandomizedSearchCV — this tries many combinations of settings (tree
   depth, learning rate, number of trees, etc.) and keeps the
   combination that performs best under cross-validation, instead of
   guessing reasonable-sounding defaults.
2. 5-FOLD STRATIFIED CROSS-VALIDATION for the final reported AUC — instead
   of one train/test split, the data is split 5 different ways and the
   model is trained/tested 5 times. The final reported score is the
   average across all 5, with a +/- range, which is a much more
   defensible "this is genuinely how good the model is" claim than a
   single lucky/unlucky split.
3. All earlier v3 baselines (untuned LogReg/RF/XGBoost, single split) are
   KEPT and still printed, so you can see the improvement tuning + CV
   provides, not just the final number in isolation.

Everything else (ESI-as-baseline-only, undertriage/overtriage reporting,
SHAP, reproducibility seeds) is unchanged from v3.
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

TF_AVAILABLE = False
try:
    import tensorflow as tf
    tf.random.set_seed(SEED)
    TF_AVAILABLE = True
except ImportError:
    print("[INFO] TensorFlow not installed — neural network comparison skipped.")

XGB_AVAILABLE = False
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    print("[WARNING] xgboost not installed — run `pip install xgboost`.")

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("[WARNING] shap not installed — run `pip install shap`.")

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
DATA_PATH = "mimic_ed_demo_prepared.csv"
if not os.path.exists(DATA_PATH):
    sys.exit(f"[ERROR] Could not find '{DATA_PATH}' in the current folder.")

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

BASE_NUMERIC_COLS = ["Sex", "Tr Temp", "Tr HR", "Tr SBP", "Tr DBP", "Tr RR", "Tr O2"]
OPTIONAL_COLS = ["Age", "Pain"]
ESI_COL = "ESI"
TEXT_COL = "CC"
TARGET_COL = "Positive"

REQUIRED_COLS = BASE_NUMERIC_COLS + [ESI_COL, TEXT_COL, TARGET_COL]
missing_required = [c for c in REQUIRED_COLS if c not in df.columns]
if missing_required:
    sys.exit(f"[ERROR] Missing required columns: {missing_required}\n"
              f"Columns found: {list(df.columns)}")

NUMERIC_COLS = BASE_NUMERIC_COLS.copy()
for col in OPTIONAL_COLS:
    if col in df.columns:
        NUMERIC_COLS.append(col)

print("Final feature set:", NUMERIC_COLS)

from sklearn.impute import SimpleImputer

X_num_raw = df[NUMERIC_COLS].apply(pd.to_numeric, errors="coerce")
imputer = SimpleImputer(strategy="median")
X_num = pd.DataFrame(imputer.fit_transform(X_num_raw), columns=NUMERIC_COLS)

y = df[TARGET_COL].astype(int).values
esi_baseline = pd.to_numeric(df[ESI_COL], errors="coerce")

if len(np.unique(y)) < 2:
    sys.exit("[ERROR] Target has only one class — check the 'Positive' column.")

# ---------------------------------------------------------------
# 2. SINGLE SPLIT (kept for the v3-style baseline comparison + SHAP)
# ---------------------------------------------------------------
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV, cross_val_score

idx_all = np.arange(len(df))
idx_train, idx_test = train_test_split(idx_all, test_size=0.25, random_state=SEED, stratify=y)

Xn_train, Xn_test = X_num.values[idx_train], X_num.values[idx_test]
y_train, y_test = y[idx_train], y[idx_test]
esi_test = esi_baseline.values[idx_test]

print(f"\nSingle split — Train: {len(y_train)}  Test: {len(y_test)}")

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

def evaluate(name, y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    undertriage_rate = fn / (fn + tp) if (fn + tp) > 0 else float("nan")
    overtriage_rate = fp / (fp + tn) if (fp + tn) > 0 else float("nan")
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = float("nan")

    print(f"\n===== {name} =====")
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}  AUC: {auc:.4f}  "
          f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}  "
          f"Recall: {recall_score(y_true, y_pred, zero_division=0):.4f}  "
          f"F1: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Undertriage rate: {undertriage_rate:.4f}   Overtriage rate: {overtriage_rate:.4f}")
    print("Confusion matrix [[TN, FP], [FN, TP]]:")
    print(cm)
    return {"name": name, "auc": auc, "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "undertriage_rate": undertriage_rate, "overtriage_rate": overtriage_rate}

results = []

# ESI baseline
esi_valid = ~np.isnan(esi_test)
if esi_valid.sum() > 0 and len(np.unique(y_test[esi_valid])) > 1:
    esi_score_inverted = (6 - esi_test[esi_valid]) / 5.0
    esi_auc = roc_auc_score(y_test[esi_valid], esi_score_inverted)
    print(f"\n===== ESI (clinical baseline) =====\nAUC: {esi_auc:.4f}")
    results.append({"name": "ESI (baseline)", "auc": esi_auc})

# Untuned baselines (same as v3, for before/after comparison)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
Xn_train_scaled = scaler.fit_transform(Xn_train)
Xn_test_scaled = scaler.transform(Xn_test)

logreg = LogisticRegression(max_iter=1000, random_state=SEED, class_weight="balanced")
logreg.fit(Xn_train_scaled, y_train)
results.append(evaluate("Logistic Regression", y_test, logreg.predict_proba(Xn_test_scaled)[:, 1]))

rf = RandomForestClassifier(n_estimators=300, random_state=SEED, class_weight="balanced", n_jobs=-1)
rf.fit(Xn_train, y_train)
results.append(evaluate("Random Forest (untuned)", y_test, rf.predict_proba(Xn_test)[:, 1]))

xgb_untuned = None
if XGB_AVAILABLE:
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_untuned = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_pos_weight,
        random_state=SEED, eval_metric="auc"
    )
    xgb_untuned.fit(Xn_train, y_train)
    results.append(evaluate("XGBoost (untuned, v3 baseline)", y_test, xgb_untuned.predict_proba(Xn_test)[:, 1]))

# ---------------------------------------------------------------
# 3. HYPERPARAMETER TUNING (XGBoost) — this is the "make it the best" step
# ---------------------------------------------------------------
xgb_tuned = None
if XGB_AVAILABLE:
    print("\n" + "=" * 70)
    print("HYPERPARAMETER TUNING — XGBoost (RandomizedSearchCV, 5-fold)")
    print("=" * 70)
    print("This tries many combinations of settings and keeps the best one, "
          "instead of using guessed default values. May take a minute.")

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    }

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    base_xgb = XGBClassifier(
        objective="binary:logistic", scale_pos_weight=scale_pos_weight,
        random_state=SEED, eval_metric="auc"
    )

    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    search = RandomizedSearchCV(
        base_xgb, param_distributions=param_dist, n_iter=40,
        scoring="roc_auc", cv=cv_strategy, random_state=SEED, n_jobs=-1, verbose=0
    )
    search.fit(Xn_train, y_train)

    print(f"\nBest cross-validated AUC during search: {search.best_score_:.4f}")
    print("Best hyperparameters found:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    xgb_tuned = search.best_estimator_
    results.append(evaluate("XGBoost (TUNED — final model)", y_test,
                             xgb_tuned.predict_proba(Xn_test)[:, 1]))

    # ---------------------------------------------------------------
    # 4. 5-FOLD CROSS-VALIDATED AUC on the TUNED model — the defensible
    #    "final" number, not dependent on one lucky/unlucky split
    # ---------------------------------------------------------------
    print("\n" + "=" * 70)
    print("FINAL CROSS-VALIDATED PERFORMANCE (tuned XGBoost, full dataset)")
    print("=" * 70)
    cv_scores = cross_val_score(
        xgb_tuned, X_num.values, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        scoring="roc_auc"
    )
    print(f"AUC per fold: {np.round(cv_scores, 4)}")
    print(f"Mean AUC: {cv_scores.mean():.4f}  (+/- {cv_scores.std():.4f})")
    print("\nThis is the number to report as your FINAL model performance — "
          "it reflects average performance across 5 different train/test "
          "splits, not a single split that could be lucky or unlucky.")

# ---------------------------------------------------------------
# 5. SHAP on the TUNED model
# ---------------------------------------------------------------
shap_target_model = xgb_tuned if xgb_tuned is not None else xgb_untuned
if SHAP_AVAILABLE and shap_target_model is not None:
    try:
        explainer = shap.TreeExplainer(shap_target_model)
        shap_values = explainer.shap_values(Xn_test)
        shap.summary_plot(shap_values, pd.DataFrame(Xn_test, columns=NUMERIC_COLS), show=False)
        plt.tight_layout()
        plt.savefig("shap_summary_plot_final.png", dpi=150)
        plt.close()
        print("\nSaved: shap_summary_plot_final.png")
    except Exception as e:
        print(f"[WARNING] SHAP failed: {e}")

# ---------------------------------------------------------------
# 6. FINAL COMPARISON TABLE
# ---------------------------------------------------------------
results_df = pd.DataFrame(results)
print("\n===== FINAL MODEL COMPARISON (MIMIC-IV-ED) =====")
print(results_df.to_string(index=False))
results_df.to_csv("model_comparison_results_final.csv", index=False)
print("\nSaved: model_comparison_results_final.csv")
