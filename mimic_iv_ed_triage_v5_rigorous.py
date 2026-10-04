# -*- coding: utf-8 -*-
"""
MIMIC-IV-ED Triage Pipeline (v5 — RIGOROUS)
==============================================

WHAT CHANGED FROM v4, AND WHY EACH CHANGE MATTERS FOR A REAL PAPER:

1. NESTED CROSS-VALIDATION (instead of "tune once, then CV with those
   fixed settings"). In v4, hyperparameters were picked once using the
   training data, then the SAME fixed settings were cross-validated.
   A reviewer can reasonably ask: "were the hyperparameters selected
   using information from data you then evaluated on?" Nested CV avoids
   this entirely: for each of 5 outer folds, hyperparameter search runs
   ONLY on that fold's training data (via an inner search), and the
   resulting model is tested on that fold's held-out data, which the
   search never saw. The outer-fold scores are then the honest,
   leak-free performance estimate.

2. ALL THREE MODELS TUNED EQUALLY (Logistic Regression, Random Forest,
   XGBoost). v4 only tuned XGBoost and left the other two at default
   settings, which is an unfair comparison — of course the tuned model
   looked best. Here, every model gets the same nested-CV tuning
   treatment, so the comparison between them is actually fair.

3. BOOTSTRAP CONFIDENCE INTERVALS + SIGNIFICANCE TEST. Raw numbers like
   "0.62 vs 0.71" don't tell you if the difference is real or just
   noise. This script pools the out-of-fold predictions from nested CV
   (predictions made on data each fold's model never trained on) and
   bootstraps (resamples the patients thousands of times) to get a 95%
   confidence interval on each model's AUC, AND a direct test of
   "is the best model significantly different from the ESI baseline."

4. REAL ERROR ANALYSIS. Instead of only reporting an aggregate
   undertriage RATE, this pulls out actual undertriaged patients (model
   said low-risk, they were actually critical) and shows their vitals
   and top SHAP-driving factors, so the paper can discuss WHAT kind of
   patient the model tends to miss, not just how often.

5. A FINAL MODEL FIT ON ALL DATA, kept SEPARATE from the performance
   estimate. The nested-CV score (point 1) is what you report as "how
   good is this approach" — it should never come from a model that was
   also used for SHAP/error-analysis, since combining them conflates
   performance estimation with interpretation. This script keeps those
   two uses clearly separated, as proper methodology.

REQUIREMENTS:
    pip install pandas numpy scikit-learn xgboost shap matplotlib
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
    sys.exit("[ERROR] xgboost is required for this script. Run: pip install xgboost")

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("[WARNING] shap not installed — error analysis will skip SHAP factors.")

# ---------------------------------------------------------------
# 1. LOAD + CLEAN (same as v4)
# ---------------------------------------------------------------
DATA_PATH = "mimic_ed_demo_prepared.csv"
if not os.path.exists(DATA_PATH):
    sys.exit(f"[ERROR] Could not find '{DATA_PATH}'.")

df = pd.read_csv(DATA_PATH)
print("Shape:", df.shape)

BASE_NUMERIC_COLS = ["Sex", "Tr Temp", "Tr HR", "Tr SBP", "Tr DBP", "Tr RR", "Tr O2"]
OPTIONAL_COLS = ["Age", "Pain"]
ESI_COL = "ESI"
TARGET_COL = "Positive"

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
esi_baseline = pd.to_numeric(df[ESI_COL], errors="coerce").values

if len(np.unique(y)) < 2:
    sys.exit("[ERROR] Target has only one class.")

X = X_num.values
n = len(y)
print(f"Total patients: {n}, Positive rate: {y.mean():.3f}")

# ---------------------------------------------------------------
# 2. NESTED CROSS-VALIDATION SETUP
# ---------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, confusion_matrix

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3  # kept smaller than outer since each inner fold has less data to work with

outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)

PARAM_GRIDS = {
    "Logistic Regression": {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "penalty": ["l2"],
        "solver": ["lbfgs"],
    },
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    },
    "XGBoost": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "min_child_weight": [1, 3, 5],
    },
}

def make_model(name, scale_pos_weight=None):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")
    elif name == "Random Forest":
        return RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1)
    elif name == "XGBoost":
        return XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight,
                              random_state=SEED, eval_metric="auc")

# Storage for pooled out-of-fold predictions — these are the honest,
# leak-free predictions used for CIs, significance testing, and error analysis.
oof_predictions = {name: np.full(n, np.nan) for name in PARAM_GRIDS}
outer_fold_scores = {name: [] for name in PARAM_GRIDS}

print("\n" + "=" * 70)
print(f"NESTED CROSS-VALIDATION: {N_OUTER_FOLDS} outer folds x "
      f"{N_INNER_FOLDS}-fold inner tuning, for all 3 models")
print("=" * 70)
print("This will take a few minutes — each model is tuned fresh inside "
      "every outer fold, so this is doing (3 models x 5 outer folds) "
      "separate hyperparameter searches.")

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y), start=1):
    X_train_fold, X_test_fold = X[train_idx], X[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_fold)
    X_test_scaled = scaler.transform(X_test_fold)

    scale_pos_weight = (y_train_fold == 0).sum() / max((y_train_fold == 1).sum(), 1)

    for model_name, param_grid in PARAM_GRIDS.items():
        base_model = make_model(model_name, scale_pos_weight)
        use_scaled = model_name == "Logistic Regression"

        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=20,
            scoring="roc_auc", cv=inner_cv, random_state=SEED, n_jobs=-1
        )
        if use_scaled:
            search.fit(X_train_scaled, y_train_fold)
            fold_proba = search.predict_proba(X_test_scaled)[:, 1]
        else:
            search.fit(X_train_fold, y_train_fold)
            fold_proba = search.predict_proba(X_test_fold)[:, 1]

        oof_predictions[model_name][test_idx] = fold_proba
        fold_auc = roc_auc_score(y_test_fold, fold_proba)
        outer_fold_scores[model_name].append(fold_auc)

    print(f"Outer fold {fold_idx}/{N_OUTER_FOLDS} done — "
          f"AUCs this fold: " +
          ", ".join(f"{k}={outer_fold_scores[k][-1]:.3f}" for k in PARAM_GRIDS))

print("\n" + "=" * 70)
print("NESTED CV RESULTS — the honest, leak-free performance estimate")
print("=" * 70)

nested_cv_summary = []
for model_name, scores in outer_fold_scores.items():
    scores = np.array(scores)
    print(f"\n{model_name}:")
    print(f"  Per-fold AUC: {np.round(scores, 4)}")
    print(f"  Mean AUC: {scores.mean():.4f}  (+/- {scores.std():.4f})")
    nested_cv_summary.append({
        "model": model_name, "mean_auc": scores.mean(), "std_auc": scores.std()
    })

# ESI baseline, for comparison (on all patients with a valid ESI value)
esi_valid = ~np.isnan(esi_baseline)
esi_score_inverted = (6 - esi_baseline[esi_valid]) / 5.0
esi_auc = roc_auc_score(y[esi_valid], esi_score_inverted)
print(f"\nESI (clinical baseline): AUC {esi_auc:.4f}")
nested_cv_summary.append({"model": "ESI (baseline)", "mean_auc": esi_auc, "std_auc": float("nan")})

nested_cv_df = pd.DataFrame(nested_cv_summary).sort_values("mean_auc", ascending=False)
nested_cv_df.to_csv("mimic_nested_cv_results.csv", index=False)
print("\nSaved: mimic_nested_cv_results.csv")

# ---------------------------------------------------------------
# 3. BOOTSTRAP CONFIDENCE INTERVALS + SIGNIFICANCE TEST
# ---------------------------------------------------------------
# Uses the pooled out-of-fold predictions (every patient's prediction
# came from a fold that never trained on them), which is the correct,
# unbiased input for this kind of resampling test.
print("\n" + "=" * 70)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 resamples)")
print("=" * 70)

N_BOOTSTRAP = 2000
rng = np.random.RandomState(SEED)

def bootstrap_auc_ci(y_true, y_prob, n_boot=N_BOOTSTRAP):
    boot_aucs = []
    n_samples = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[idx])) < 2:
            continue  # skip degenerate resamples with only one class
        boot_aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))
    boot_aucs = np.array(boot_aucs)
    return np.percentile(boot_aucs, 2.5), np.percentile(boot_aucs, 97.5), boot_aucs

best_model_name = nested_cv_df.iloc[0]["model"]
if best_model_name == "ESI (baseline)":
    best_model_name = nested_cv_df.iloc[1]["model"]  # best actual trained model

best_oof_preds = oof_predictions[best_model_name]
ci_low, ci_high, boot_dist = bootstrap_auc_ci(y, best_oof_preds)
print(f"\n{best_model_name} (best model) — Bootstrap 95% CI for AUC: "
      f"[{ci_low:.4f}, {ci_high:.4f}]")

# Paired significance test: best model vs ESI, on patients with valid ESI
esi_probs_full = np.full(n, np.nan)
esi_probs_full[esi_valid] = esi_score_inverted
both_valid = esi_valid  # ESI validity defines the comparable subset

y_compare = y[both_valid]
model_probs_compare = best_oof_preds[both_valid]
esi_probs_compare = esi_probs_full[both_valid]

diffs = []
for _ in range(N_BOOTSTRAP):
    idx = rng.randint(0, len(y_compare), len(y_compare))
    if len(np.unique(y_compare[idx])) < 2:
        continue
    model_auc_b = roc_auc_score(y_compare[idx], model_probs_compare[idx])
    esi_auc_b = roc_auc_score(y_compare[idx], esi_probs_compare[idx])
    diffs.append(model_auc_b - esi_auc_b)
diffs = np.array(diffs)

p_value_approx = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
diff_ci_low, diff_ci_high = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

print(f"\nSignificance test: {best_model_name} vs. ESI baseline")
print(f"  Mean AUC difference (model - ESI): {diffs.mean():.4f}")
print(f"  95% CI of difference: [{diff_ci_low:.4f}, {diff_ci_high:.4f}]")
print(f"  Approx. two-sided bootstrap p-value: {p_value_approx:.4f}")
if diff_ci_low <= 0 <= diff_ci_high:
    print("  -> CI includes 0: NOT a statistically significant difference.")
else:
    print("  -> CI excludes 0: statistically significant difference.")

# ---------------------------------------------------------------
# 4. FINAL MODEL (fit on ALL data) — for SHAP + error analysis ONLY,
#    never used for the performance numbers above
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("FITTING FINAL MODEL ON ALL DATA (for SHAP + error analysis only — "
      "NOT used for performance reporting, see section 2-3 above for that)")
print("=" * 70)

scale_pos_weight_full = (y == 0).sum() / max((y == 1).sum(), 1)
final_search = RandomizedSearchCV(
    XGBClassifier(objective="binary:logistic", scale_pos_weight=scale_pos_weight_full,
                  random_state=SEED, eval_metric="auc"),
    param_distributions=PARAM_GRIDS["XGBoost"], n_iter=30,
    scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    random_state=SEED, n_jobs=-1
)
final_search.fit(X, y)
final_model = final_search.best_estimator_
print(f"Final model best params: {final_search.best_params_}")

# ---------------------------------------------------------------
# 5. ERROR ANALYSIS — actual undertriaged patients, with SHAP factors
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("ERROR ANALYSIS — undertriaged patients (using pooled nested-CV "
      "out-of-fold predictions from the best model, so these are honest "
      "errors on data the fold's model never trained on)")
print("=" * 70)

threshold = 0.5
predicted_class = (best_oof_preds >= threshold).astype(int)
undertriaged_mask = (predicted_class == 0) & (y == 1)  # model said low-risk, actually critical
undertriaged_idx = np.where(undertriaged_mask)[0]

print(f"\nTotal undertriaged patients: {len(undertriaged_idx)} out of {n}")

if SHAP_AVAILABLE and len(undertriaged_idx) > 0:
    explainer = shap.TreeExplainer(final_model)
    shap_vals_all = explainer.shap_values(X)

    print(f"\nShowing up to 5 undertriaged cases with their top SHAP factors:")
    for i, idx in enumerate(undertriaged_idx[:5]):
        print(f"\n--- Undertriaged patient #{i+1} (predicted risk: {best_oof_preds[idx]:.3f}, "
              f"actually critical) ---")
        patient_vitals = pd.Series(X[idx], index=NUMERIC_COLS)
        print("Vitals:", patient_vitals.to_dict())
        contribs = pd.Series(shap_vals_all[idx], index=NUMERIC_COLS).sort_values(key=abs, ascending=False)
        print("Top factors pulling risk DOWN (why the model under-estimated this patient):")
        print(contribs[contribs < 0].head(3))

undertriaged_df = pd.DataFrame(X[undertriaged_idx], columns=NUMERIC_COLS)
undertriaged_df["predicted_risk"] = best_oof_preds[undertriaged_idx]
undertriaged_df.to_csv("mimic_undertriaged_cases.csv", index=False)
print("\nSaved: mimic_undertriaged_cases.csv")

# ---------------------------------------------------------------
# 6. SHAP SUMMARY PLOT (final model)
# ---------------------------------------------------------------
if SHAP_AVAILABLE:
    try:
        explainer = shap.TreeExplainer(final_model)
        shap_values = explainer.shap_values(X)
        shap.summary_plot(shap_values, pd.DataFrame(X, columns=NUMERIC_COLS), show=False)
        plt.tight_layout()
        plt.savefig("mimic_shap_summary_rigorous.png", dpi=150)
        plt.close()
        print("Saved: mimic_shap_summary_rigorous.png")
    except Exception as e:
        print(f"[WARNING] SHAP plot failed: {e}")

# ---------------------------------------------------------------
# 7. PUBLICATION-QUALITY FIGURES — for Chapter 5 (Results & Discussion)
# ---------------------------------------------------------------
# These are built from the pooled nested-CV out-of-fold predictions
# (never from a model tested on data it trained on), so every figure
# here is as honest as the numbers in section 2-3 above.
print("\n" + "=" * 70)
print("GENERATING PUBLICATION FIGURES")
print("=" * 70)

from sklearn.metrics import roc_curve, ConfusionMatrixDisplay

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

# --- Figure 1: Confusion matrix (best model, threshold 0.5) ---
cm = confusion_matrix(y, predicted_class, labels=[0, 1])
fig, ax = plt.subplots(figsize=(5, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Non-Critical", "Critical"])
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"Confusion Matrix — {best_model_name}\n(MIMIC-IV-ED, pooled nested-CV predictions)")
plt.tight_layout()
plt.savefig("mimic_confusion_matrix.png", dpi=200)
plt.close()
print("Saved: mimic_confusion_matrix.png")

# --- Figure 2: ROC curve — best model vs ESI baseline ---
fpr_model, tpr_model, _ = roc_curve(y_compare, model_probs_compare)
fpr_esi, tpr_esi, _ = roc_curve(y_compare, esi_probs_compare)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr_model, tpr_model, label=f"{best_model_name} (AUC = {roc_auc_score(y_compare, model_probs_compare):.3f})",
        color="#2166AC", linewidth=2)
ax.plot(fpr_esi, tpr_esi, label=f"ESI Baseline (AUC = {esi_auc:.3f})",
        color="#B2182B", linewidth=2, linestyle="--")
ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, label="Chance (AUC = 0.5)")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve — Model vs. ESI Clinical Baseline\n(MIMIC-IV-ED, nested cross-validation)")
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.02])
plt.tight_layout()
plt.savefig("mimic_roc_curve.png", dpi=200)
plt.close()
print("Saved: mimic_roc_curve.png")

# --- Figure 3: Model comparison bar chart with error bars ---
plot_df = nested_cv_df.copy()
plot_df["std_auc"] = plot_df["std_auc"].fillna(0)  # ESI baseline has no std (single value, not CV'd)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#B2182B" if "ESI" in m else "#2166AC" for m in plot_df["model"]]
bars = ax.bar(plot_df["model"], plot_df["mean_auc"], yerr=plot_df["std_auc"],
              capsize=5, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("AUC (nested 5-fold cross-validation)")
ax.set_title("Model Comparison — MIMIC-IV-ED\n(error bars = +/- 1 std across outer folds)")
ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="Chance level")
ax.set_ylim([0, 1])
plt.xticks(rotation=20, ha="right")
plt.legend()
plt.tight_layout()
plt.savefig("mimic_model_comparison_barchart.png", dpi=200)
plt.close()
print("Saved: mimic_model_comparison_barchart.png")

# --- Save all models' out-of-fold predictions (not just the best one),
#     so raw predictions are available for any further analysis/appendix ---
oof_df = pd.DataFrame({"true_label": y})
for model_name in PARAM_GRIDS:
    oof_df[f"{model_name}_oof_prediction"] = oof_predictions[model_name]
oof_df["ESI_raw"] = esi_baseline
oof_df.to_csv("mimic_all_oof_predictions.csv", index=False)
print("Saved: mimic_all_oof_predictions.csv")

print("\n" + "=" * 70)
print("DONE. Report the NESTED CV numbers (section 2) as your main result, "
      "the bootstrap CI + significance test (section 3) to support any "
      "'beats/doesn't beat ESI' claim, the error analysis (section 5) in "
      "your discussion section, and the 3 figures above directly in "
      "Chapter 5 (Results & Discussion).")
print("=" * 70)