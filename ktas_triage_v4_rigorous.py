# -*- coding: utf-8 -*-
"""
KTAS Triage Pipeline (v4 — RIGOROUS)
========================================
Same upgrades as the MIMIC v5 script — see that file's header for the full
reasoning. Applied here to the 5-class KTAS problem (vitals + chief
complaint text vs. KTAS_expert, nurse's KTAS_RN score as baseline).

REQUIREMENTS:
    pip install pandas numpy scikit-learn xgboost shap matplotlib scipy
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

try:
    from xgboost import XGBClassifier
except ImportError:
    sys.exit("[ERROR] xgboost is required. Run: pip install xgboost")

SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("[WARNING] shap not installed — error analysis will skip SHAP factors.")

# ---------------------------------------------------------------
# 1. LOAD + CLEAN (same as v3)
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
nurse_raw = pd.to_numeric(df[BASELINE_COL], errors="coerce")
chief_complaint_raw = df[TEXT_COL].fillna("").astype(str).str.lower()

valid_rows = ~y_raw.isna()
X_num = X_num[valid_rows.values].reset_index(drop=True)
y_raw = y_raw[valid_rows].reset_index(drop=True)
nurse_raw = nurse_raw[valid_rows].reset_index(drop=True)
chief_complaint_raw = chief_complaint_raw[valid_rows.values].reset_index(drop=True)

y = (y_raw - 1).astype(int).values  # 0-indexed for XGBoost
n = len(y)
print(f"Total patients: {n}")
print("Target class balance (KTAS level : count):")
print(y_raw.value_counts().sort_index())

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack, csr_matrix

# NOTE: for proper nested CV, TF-IDF must be fit ONLY on each fold's
# training data, never on the full dataset ahead of time — fitting it on
# everything first would leak test-set vocabulary into training, which is
# a subtle but real form of data leakage. This is handled per-fold below.

# ---------------------------------------------------------------
# 2. NESTED CROSS-VALIDATION SETUP
# ---------------------------------------------------------------
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics import confusion_matrix

N_OUTER_FOLDS = 5
N_INNER_FOLDS = 3

outer_cv = StratifiedKFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=SEED)
inner_cv = StratifiedKFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=SEED)

PARAM_GRIDS = {
    "Logistic Regression": {"C": [0.001, 0.01, 0.1, 1, 10, 100]},
    "Random Forest": {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [5, 8, 12, 20, None],
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

def make_model(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=2000, random_state=SEED, class_weight="balanced")
    elif name == "Random Forest":
        return RandomForestClassifier(random_state=SEED, class_weight="balanced", n_jobs=-1)
    elif name == "XGBoost":
        return XGBClassifier(objective="multi:softprob", num_class=5,
                              random_state=SEED, eval_metric="mlogloss")

oof_predictions = {name: np.full(n, -1, dtype=int) for name in PARAM_GRIDS}
outer_fold_scores = {name: [] for name in PARAM_GRIDS}

print("\n" + "=" * 70)
print(f"NESTED CROSS-VALIDATION: {N_OUTER_FOLDS} outer folds x "
      f"{N_INNER_FOLDS}-fold inner tuning, for all 3 models (vitals + text)")
print("=" * 70)
print("This will take several minutes given the dataset size and TF-IDF "
      "refitting per fold — let it run fully.")

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_num.values, y), start=1):
    Xn_train_fold, Xn_test_fold = X_num.values[train_idx], X_num.values[test_idx]
    y_train_fold, y_test_fold = y[train_idx], y[test_idx]
    text_train_fold = chief_complaint_raw.values[train_idx]
    text_test_fold = chief_complaint_raw.values[test_idx]

    # TF-IDF fit ONLY on this fold's training text — avoids leakage
    tfidf_fold = TfidfVectorizer(max_features=200, stop_words="english", min_df=2)
    text_train_tfidf = tfidf_fold.fit_transform(text_train_fold)
    text_test_tfidf = tfidf_fold.transform(text_test_fold)

    X_train_combined = hstack([csr_matrix(Xn_train_fold), text_train_tfidf]).tocsr()
    X_test_combined = hstack([csr_matrix(Xn_test_fold), text_test_tfidf]).tocsr()

    scaler = MaxAbsScaler()
    X_train_scaled = scaler.fit_transform(X_train_combined)
    X_test_scaled = scaler.transform(X_test_combined)

    for model_name, param_grid in PARAM_GRIDS.items():
        base_model = make_model(model_name)
        use_scaled = model_name == "Logistic Regression"

        search = RandomizedSearchCV(
            base_model, param_distributions=param_grid, n_iter=15,
            scoring="accuracy", cv=inner_cv, random_state=SEED, n_jobs=-1
        )
        if use_scaled:
            search.fit(X_train_scaled, y_train_fold)
            fold_pred = search.predict(X_test_scaled)
        else:
            search.fit(X_train_combined, y_train_fold)
            fold_pred = search.predict(X_test_combined)

        oof_predictions[model_name][test_idx] = fold_pred
        fold_acc = (fold_pred == y_test_fold).mean()
        outer_fold_scores[model_name].append(fold_acc)

    print(f"Outer fold {fold_idx}/{N_OUTER_FOLDS} done — "
          f"Accuracy this fold: " +
          ", ".join(f"{k}={outer_fold_scores[k][-1]:.3f}" for k in PARAM_GRIDS))

print("\n" + "=" * 70)
print("NESTED CV RESULTS — the honest, leak-free performance estimate")
print("=" * 70)

nested_cv_summary = []
for model_name, scores in outer_fold_scores.items():
    scores = np.array(scores)
    print(f"\n{model_name}:")
    print(f"  Per-fold accuracy: {np.round(scores, 4)}")
    print(f"  Mean accuracy: {scores.mean():.4f}  (+/- {scores.std():.4f})")
    nested_cv_summary.append({"model": model_name, "mean_accuracy": scores.mean(),
                               "std_accuracy": scores.std()})

valid_nurse = ~np.isnan(nurse_raw.values)
nurse_acc = (nurse_raw.values[valid_nurse] == y_raw.values[valid_nurse]).mean()
print(f"\nNurse's own score (whole dataset): accuracy {nurse_acc:.4f}")
nested_cv_summary.append({"model": "Nurse's Own Score (baseline)",
                           "mean_accuracy": nurse_acc, "std_accuracy": float("nan")})

nested_cv_df = pd.DataFrame(nested_cv_summary).sort_values("mean_accuracy", ascending=False)
nested_cv_df.to_csv("ktas_nested_cv_results.csv", index=False)
print("\nSaved: ktas_nested_cv_results.csv")

# ---------------------------------------------------------------
# 3. BOOTSTRAP CONFIDENCE INTERVALS + SIGNIFICANCE TEST
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("BOOTSTRAP 95% CONFIDENCE INTERVALS (2000 resamples)")
print("=" * 70)

N_BOOTSTRAP = 2000
rng = np.random.RandomState(SEED)

def bootstrap_acc_ci(y_true, y_pred, n_boot=N_BOOTSTRAP):
    boot_accs = []
    n_samples = len(y_true)
    for _ in range(n_boot):
        idx = rng.randint(0, n_samples, n_samples)
        boot_accs.append((y_true[idx] == y_pred[idx]).mean())
    boot_accs = np.array(boot_accs)
    return np.percentile(boot_accs, 2.5), np.percentile(boot_accs, 97.5)

best_model_name = nested_cv_df.iloc[0]["model"]
if "Nurse" in best_model_name:
    best_model_name = nested_cv_df.iloc[1]["model"]

best_oof_preds = oof_predictions[best_model_name]
ci_low, ci_high = bootstrap_acc_ci(y, best_oof_preds)
print(f"\n{best_model_name} (best trained model) — Bootstrap 95% CI for accuracy: "
      f"[{ci_low:.4f}, {ci_high:.4f}]")

# Paired significance test: best model vs nurse
compare_mask = valid_nurse
y_compare = y_raw.values[compare_mask] - 1
model_pred_compare = best_oof_preds[compare_mask]
nurse_pred_compare = nurse_raw.values[compare_mask] - 1

diffs = []
for _ in range(N_BOOTSTRAP):
    idx = rng.randint(0, len(y_compare), len(y_compare))
    model_acc_b = (y_compare[idx] == model_pred_compare[idx]).mean()
    nurse_acc_b = (y_compare[idx] == nurse_pred_compare[idx]).mean()
    diffs.append(model_acc_b - nurse_acc_b)
diffs = np.array(diffs)

p_value_approx = 2 * min((diffs > 0).mean(), (diffs < 0).mean())
diff_ci_low, diff_ci_high = np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

print(f"\nSignificance test: {best_model_name} vs. Nurse's own score")
print(f"  Mean accuracy difference (model - nurse): {diffs.mean():.4f}")
print(f"  95% CI of difference: [{diff_ci_low:.4f}, {diff_ci_high:.4f}]")
print(f"  Approx. two-sided bootstrap p-value: {p_value_approx:.4f}")
if diff_ci_low <= 0 <= diff_ci_high:
    print("  -> CI includes 0: NOT a statistically significant difference.")
else:
    print("  -> CI excludes 0: statistically significant difference "
          f"({'model better' if diffs.mean() > 0 else 'nurse better'}).")

# ---------------------------------------------------------------
# 4. FINAL MODEL (fit on ALL data) — for SHAP + error analysis ONLY
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("FITTING FINAL MODEL ON ALL DATA (for SHAP + error analysis only)")
print("=" * 70)

tfidf_full = TfidfVectorizer(max_features=200, stop_words="english", min_df=2)
text_full_tfidf = tfidf_full.fit_transform(chief_complaint_raw)
X_combined_full = hstack([csr_matrix(X_num.values), text_full_tfidf]).tocsr()

final_search = RandomizedSearchCV(
    XGBClassifier(objective="multi:softprob", num_class=5, random_state=SEED, eval_metric="mlogloss"),
    param_distributions=PARAM_GRIDS["XGBoost"], n_iter=30,
    scoring="accuracy", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    random_state=SEED, n_jobs=-1
)
final_search.fit(X_combined_full, y)
final_model = final_search.best_estimator_
print(f"Final model best params: {final_search.best_params_}")

# ---------------------------------------------------------------
# 5. ERROR ANALYSIS — undertriaged patients
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("ERROR ANALYSIS — undertriaged patients (predicted level HIGHER "
      "number = less urgent than true expert level)")
print("=" * 70)

diff_levels = (best_oof_preds + 1) - (y + 1)
undertriaged_idx = np.where(diff_levels > 0)[0]
print(f"\nTotal undertriaged patients: {len(undertriaged_idx)} out of {n}")

worst_cases_idx = undertriaged_idx[np.argsort(-diff_levels[undertriaged_idx])][:5]
print(f"\nShowing the 5 WORST undertriage cases (largest gap between "
      f"predicted and true urgency):")
for i, idx in enumerate(worst_cases_idx):
    print(f"\n--- Worst undertriage case #{i+1} ---")
    print(f"  True KTAS level: {y[idx]+1}   Predicted level: {best_oof_preds[idx]+1}   "
          f"Gap: {diff_levels[idx]} levels")
    print(f"  Vitals: {pd.Series(X_num.values[idx], index=NUMERIC_COLS).to_dict()}")
    print(f"  Chief complaint: \"{chief_complaint_raw.values[idx]}\"")

undertriaged_df = pd.DataFrame(X_num.values[undertriaged_idx], columns=NUMERIC_COLS)
undertriaged_df["true_level"] = y[undertriaged_idx] + 1
undertriaged_df["predicted_level"] = best_oof_preds[undertriaged_idx] + 1
undertriaged_df["chief_complaint"] = chief_complaint_raw.values[undertriaged_idx]
undertriaged_df.to_csv("ktas_undertriaged_cases.csv", index=False)
print("\nSaved: ktas_undertriaged_cases.csv")

# ---------------------------------------------------------------
# 6. SHAP SUMMARY (final model, vitals only for readability — same
#    approach as the earlier cleaned-up KTAS SHAP bar chart)
# ---------------------------------------------------------------
if SHAP_AVAILABLE:
    try:
        # Refit a vitals-only version for an interpretable SHAP plot
        vitals_only_search = RandomizedSearchCV(
            XGBClassifier(objective="multi:softprob", num_class=5, random_state=SEED, eval_metric="mlogloss"),
            param_distributions=PARAM_GRIDS["XGBoost"], n_iter=30,
            scoring="accuracy", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
            random_state=SEED, n_jobs=-1
        )
        vitals_only_search.fit(X_num.values, y)
        vitals_model = vitals_only_search.best_estimator_

        explainer = shap.TreeExplainer(vitals_model)
        shap_values = np.array(explainer.shap_values(X_num.values))

        if shap_values.ndim == 3:
            if shap_values.shape[-1] == 5:
                overall_importance = np.abs(shap_values).mean(axis=(0, 2))
            else:
                overall_importance = np.abs(shap_values).mean(axis=(0, 1))
        else:
            overall_importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": NUMERIC_COLS, "mean_abs_shap": overall_importance
        }).sort_values("mean_abs_shap", ascending=True)

        plt.figure(figsize=(8, 6))
        plt.barh(importance_df["feature"], importance_df["mean_abs_shap"], color="#4C72B0")
        plt.xlabel("Mean |SHAP value| (average impact across all 5 KTAS classes)")
        plt.title("KTAS Model (Rigorous, Vitals-Only) — Global Feature Importance")
        plt.tight_layout()
        plt.savefig("ktas_shap_summary_rigorous.png", dpi=150)
        plt.close()
        print("\nSaved: ktas_shap_summary_rigorous.png")
    except Exception as e:
        print(f"[WARNING] SHAP step failed: {e}")

# ---------------------------------------------------------------
# 7. PUBLICATION-QUALITY FIGURES — for Chapter 5 (Results & Discussion)
# ---------------------------------------------------------------
print("\n" + "=" * 70)
print("GENERATING PUBLICATION FIGURES")
print("=" * 70)

from sklearn.metrics import ConfusionMatrixDisplay

plt.rcParams.update({"font.size": 11, "font.family": "serif"})

# --- Figure 1: Confusion matrix (best model, 5-class, KTAS 1-5 labels) ---
cm = confusion_matrix(y, best_oof_preds, labels=list(range(5)))
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[f"KTAS {c}" for c in range(1, 6)])
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
ax.set_title(f"Confusion Matrix — {best_model_name}\n(KTAS, pooled nested-CV predictions, vitals+text)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("ktas_confusion_matrix.png", dpi=200)
plt.close()
print("Saved: ktas_confusion_matrix.png")

# --- Figure 2: Nurse's own confusion matrix, for direct visual comparison ---
nurse_pred_0idx = (nurse_raw.values[valid_nurse] - 1).astype(int)
y_nurse_compare = (y_raw.values[valid_nurse] - 1).astype(int)
cm_nurse = confusion_matrix(y_nurse_compare, nurse_pred_0idx, labels=list(range(5)))
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm_nurse, display_labels=[f"KTAS {c}" for c in range(1, 6)])
disp.plot(ax=ax, cmap="Greens", colorbar=False, values_format="d")
ax.set_title("Confusion Matrix — Nurse's Own Score (KTAS_RN)\n(whole dataset, for comparison)")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("ktas_confusion_matrix_nurse.png", dpi=200)
plt.close()
print("Saved: ktas_confusion_matrix_nurse.png")

# --- Figure 3: Model comparison bar chart with error bars ---
plot_df = nested_cv_df.copy()
plot_df["std_accuracy"] = plot_df["std_accuracy"].fillna(0)

fig, ax = plt.subplots(figsize=(8, 5))
colors = ["#1B7837" if "Nurse" in m else "#2166AC" for m in plot_df["model"]]
bars = ax.bar(plot_df["model"], plot_df["mean_accuracy"], yerr=plot_df["std_accuracy"],
              capsize=5, color=colors, edgecolor="black", linewidth=0.5)
ax.set_ylabel("Exact-Match Accuracy (nested 5-fold cross-validation)")
ax.set_title("Model Comparison — KTAS\n(error bars = +/- 1 std across outer folds)")
ax.set_ylim([0, 1])
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig("ktas_model_comparison_barchart.png", dpi=200)
plt.close()
print("Saved: ktas_model_comparison_barchart.png")

# --- Save all models' out-of-fold predictions ---
oof_df = pd.DataFrame({"true_level": y + 1})
for model_name in PARAM_GRIDS:
    oof_df[f"{model_name}_oof_prediction"] = oof_predictions[model_name] + 1
oof_df["nurse_KTAS_RN"] = nurse_raw.values
oof_df.to_csv("ktas_all_oof_predictions.csv", index=False)
print("Saved: ktas_all_oof_predictions.csv")

print("\n" + "=" * 70)
print("DONE. Report the NESTED CV numbers (section 2) as your main result, "
      "the bootstrap CI + significance test (section 3) to support any "
      "'beats/doesn't beat the nurse' claim, the error analysis (section 5) "
      "in your discussion section, and the figures above directly in "
      "Chapter 5 (Results & Discussion).")
print("=" * 70)