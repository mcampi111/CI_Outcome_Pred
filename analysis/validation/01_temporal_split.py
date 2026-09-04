#!/usr/bin/env python3
"""
run_tabpfn_over_time.py
=======================
Replicates the new RQ2 longitudinal analyses for the CI paper using TabPFN.

Three outputs:
  1) Prediction R² over time (regression):
       - Mono1 at each post-op timepoint (0-6, 6-12, 12-24, 24+ mo)
       - With pre-op features only vs pre-op + previous-timepoint Mono1
  2) Risk classification AUC over time (binary, Mono1 < 40%):
       - Same comparisons as above
  3) Saves CSVs ready to plot.

Setup
-----
  source ~/venvs/general/bin/activate
  cd ~/Desktop/Postdoc_Zurich/CI_Outcome_Pred/code/notebooks/Unified_Analysis
  python run_tabpfn_over_time.py

Data path defaults to:
  ~/Desktop/Postdoc_Zurich/CI_Outcome_Pred/code/data/CI_UNIFIED_DATASET.xlsx

Override by passing path as first arg:
  python run_tabpfn_over_time.py /path/to/dataset.xlsx
"""

import os
import sys
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
DEFAULT_DATA = Path.home() / "Desktop" / "Postdoc_Zurich" / "CI_Outcome_Pred" / "code" / "data" / "CI_UNIFIED_DATASET.xlsx"
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
OUT_DIR = Path.cwd() / "tabpfn_over_time_results"
OUT_DIR.mkdir(exist_ok=True)

CV_FOLDS = 10
RANDOM_STATE = 42
POOR_THRESHOLD = 40.0  # Mono1 < 40% = poor outcome

# --------------------------------------------------------------------------
# Imports — TabPFN local + sklearn pipeline
# --------------------------------------------------------------------------
try:
    from tabpfn import TabPFNRegressor, TabPFNClassifier
    print("✓ TabPFN local imported")
except ImportError:
    print("ERROR: TabPFN not installed. Run:  pip install tabpfn")
    sys.exit(1)

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------
# Load data
# --------------------------------------------------------------------------
print(f"\nLoading: {DATA_PATH}")
if not DATA_PATH.exists():
    print(f"ERROR: data file not found at {DATA_PATH}")
    sys.exit(1)

df = pd.read_excel(DATA_PATH)
df = df[df['Age_at_OP'] >= 18].copy()
print(f"  Adults: N = {len(df)}")

# Build feature set: 60ish pre-op features (matches paper)
audiom = [c for c in df.columns
          if (c.startswith('prPTA_') or c.startswith('prFF_'))
          and (c.endswith('_CI') or c.endswith('_Co'))]
categorical_vars = ['EVA_DEAF_ONSET', 'COM_ARTICULATION', 'COM_MULTILANG_HOME',
                    'COM_SIGN_LANGUAGE', 'COM_FATHER_HACI_USER', 'COM_MOTHER_HACI_USER',
                    'SES_PROFESSION_LEARNED', 'SES_EDUCATION']
numeric_vars = ['Age_at_OP', 'duration_HL_total']
categorical_vars = [v for v in categorical_vars if v in df.columns]
numeric_vars = [v for v in numeric_vars if v in df.columns]

features_df = df[audiom + numeric_vars].apply(pd.to_numeric, errors='coerce').astype(float)
cat_encoded = pd.get_dummies(df[categorical_vars], drop_first=False, dummy_na=False).astype(float)
features_df = pd.concat([features_df, cat_encoded], axis=1)
print(f"  Pre-op features: {features_df.shape[1]}")

# Timepoint columns
tp_cols = {
    '0-6 mo':  'SQL_Freiburger_0_6mo',
    '6-12 mo': 'SQL_Freiburger_6_12mo',
    '12-24 mo':'SQL_Freiburger_12_24mo',
    '24+ mo':  'SQL_Freiburger_24plus',
}

# --------------------------------------------------------------------------
# Helpers — TabPFN with manual CV (TabPFN doesn't play well with sklearn pipeline)
# --------------------------------------------------------------------------
def cv_tabpfn_regression(X, y, n_splits=CV_FOLDS):
    """Run TabPFN regressor with manual k-fold CV. Returns mean R² across folds."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        # Preprocess inside fold (no leak)
        imp = SimpleImputer(strategy='median').fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        X_tr = sc.transform(imp.transform(X[tr]))
        X_te = sc.transform(imp.transform(X[te]))
        try:
            reg = TabPFNRegressor(random_state=RANDOM_STATE)
            reg.fit(X_tr, y[tr])
            pred = reg.predict(X_te)
            r2_scores.append(r2_score(y[te], pred))
        except Exception as e:
            print(f"    fold {fold} failed: {e}")
            r2_scores.append(np.nan)
    return float(np.nanmean(r2_scores)), float(np.nanstd(r2_scores))

def cv_tabpfn_classification(X, y, n_splits=CV_FOLDS):
    """Run TabPFN classifier with manual stratified k-fold CV. Returns mean AUC across folds."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        imp = SimpleImputer(strategy='median').fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        X_tr = sc.transform(imp.transform(X[tr]))
        X_te = sc.transform(imp.transform(X[te]))
        try:
            clf = TabPFNClassifier(random_state=RANDOM_STATE)
            clf.fit(X_tr, y[tr])
            proba = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y[te], proba))
        except Exception as e:
            print(f"    fold {fold} failed: {e}")
            aucs.append(np.nan)
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))

# --------------------------------------------------------------------------
# Define all conditions to run
# --------------------------------------------------------------------------
prev_pairs = [
    ('6-12 mo', '0-6 mo',  'SQL_Freiburger_0_6mo'),
    ('12-24 mo','6-12 mo', 'SQL_Freiburger_6_12mo'),
    ('24+ mo',  '12-24 mo','SQL_Freiburger_12_24mo'),
]

# --------------------------------------------------------------------------
# REGRESSION
# --------------------------------------------------------------------------
print("\n" + "="*70)
print("REGRESSION — Prediction R² over time (TabPFN)")
print("="*70)

reg_results = []

# (a) pre-op only at each timepoint
for tp_name, tp_col in tp_cols.items():
    mask = df[tp_col].notna()
    X = features_df[mask].values
    y = df.loc[mask, tp_col].values.astype(float)
    print(f"\n{tp_name}: pre-op only (N={mask.sum()})")
    t0 = time.time()
    r2_mean, r2_std = cv_tabpfn_regression(X, y)
    elapsed = time.time() - t0
    print(f"  R² = {r2_mean:.3f} ± {r2_std:.3f}  (time: {elapsed:.0f}s)")
    reg_results.append({
        'timepoint': tp_name, 'features': 'pre-op only',
        'N': int(mask.sum()), 'R2_mean': r2_mean, 'R2_std': r2_std,
    })

# (b) pre-op + previous timepoint Mono1
for tp_name, prev_name, prev_col in prev_pairs:
    tp_col = tp_cols[tp_name]
    mask = df[tp_col].notna() & df[prev_col].notna()
    X = np.column_stack([
        features_df[mask].values,
        df.loc[mask, prev_col].values.reshape(-1, 1)
    ])
    y = df.loc[mask, tp_col].values.astype(float)
    print(f"\n{tp_name}: pre-op + {prev_name} (N={mask.sum()})")
    t0 = time.time()
    r2_mean, r2_std = cv_tabpfn_regression(X, y)
    elapsed = time.time() - t0
    print(f"  R² = {r2_mean:.3f} ± {r2_std:.3f}  (time: {elapsed:.0f}s)")
    reg_results.append({
        'timepoint': tp_name, 'features': f'pre-op + {prev_name}',
        'N': int(mask.sum()), 'R2_mean': r2_mean, 'R2_std': r2_std,
    })

reg_df = pd.DataFrame(reg_results)
out_reg = OUT_DIR / 'tabpfn_prediction_over_time.csv'
reg_df.to_csv(out_reg, index=False, float_format='%.4f')
print(f"\n✓ Saved: {out_reg}")

# --------------------------------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------------------------------
print("\n" + "="*70)
print(f"CLASSIFICATION — Risk AUC (Mono1 < {POOR_THRESHOLD}%) over time (TabPFN)")
print("="*70)

clf_results = []

# (a) pre-op only at each timepoint
for tp_name, tp_col in tp_cols.items():
    mask = df[tp_col].notna()
    y = (df.loc[mask, tp_col] < POOR_THRESHOLD).astype(int).values
    if y.sum() < 10 or (1-y).sum() < 10:
        print(f"\n{tp_name}: too few cases, skipping")
        continue
    X = features_df[mask].values
    print(f"\n{tp_name}: pre-op only (N={mask.sum()}, poor={y.sum()})")
    t0 = time.time()
    auc_mean, auc_std = cv_tabpfn_classification(X, y)
    elapsed = time.time() - t0
    print(f"  AUC = {auc_mean:.3f} ± {auc_std:.3f}  (time: {elapsed:.0f}s)")
    clf_results.append({
        'timepoint': tp_name, 'features': 'pre-op only',
        'N': int(mask.sum()), 'N_poor': int(y.sum()),
        'AUC_mean': auc_mean, 'AUC_std': auc_std,
    })

# (b) pre-op + previous timepoint Mono1
for tp_name, prev_name, prev_col in prev_pairs:
    tp_col = tp_cols[tp_name]
    mask = df[tp_col].notna() & df[prev_col].notna()
    y = (df.loc[mask, tp_col] < POOR_THRESHOLD).astype(int).values
    if y.sum() < 10 or (1-y).sum() < 10:
        print(f"\n{tp_name}: too few cases, skipping")
        continue
    X = np.column_stack([
        features_df[mask].values,
        df.loc[mask, prev_col].values.reshape(-1, 1)
    ])
    print(f"\n{tp_name}: pre-op + {prev_name} (N={mask.sum()}, poor={y.sum()})")
    t0 = time.time()
    auc_mean, auc_std = cv_tabpfn_classification(X, y)
    elapsed = time.time() - t0
    print(f"  AUC = {auc_mean:.3f} ± {auc_std:.3f}  (time: {elapsed:.0f}s)")
    clf_results.append({
        'timepoint': tp_name, 'features': f'pre-op + {prev_name}',
        'N': int(mask.sum()), 'N_poor': int(y.sum()),
        'AUC_mean': auc_mean, 'AUC_std': auc_std,
    })

clf_df = pd.DataFrame(clf_results)
out_clf = OUT_DIR / 'tabpfn_classification_over_time.csv'
clf_df.to_csv(out_clf, index=False, float_format='%.4f')
print(f"\n✓ Saved: {out_clf}")

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY")
print("="*70)
print("\nRegression (R²):")
print(reg_df.to_string(index=False))
print("\nClassification (AUC):")
print(clf_df.to_string(index=False))
print(f"\nResults saved in: {OUT_DIR}")
print("\nDone. Send the two CSVs back to Claude for plotting.\n")
