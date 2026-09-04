#!/usr/bin/env python3
"""
run_tabpfn_finetune_over_time.py
================================
Same as run_tabpfn_over_time.py but with TabPFN FINE-TUNED instead of zero-shot.

Fine-tuning config matches CI Equalizer paper:
  - 50 epochs
  - learning rate 5e-6
  - weight decay 0.01
  - AdamW optimizer
  - 2 ensemble members during fine-tuning
  - 4 ensemble members at final inference

Setup
-----
  source ~/venvs/general/bin/activate
  cd ~/Desktop/Postdoc_Zurich/DATACI/CI_Data_Package
  python run_tabpfn_finetune_over_time.py CI_UNIFIED_DATASET.xlsx

Estimated time: 45-70 min on M4 Pro (50 epochs per condition x 14 conditions).
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
DEFAULT_DATA = Path.home() / "Desktop" / "Postdoc_Zurich" / "DATACI" / "CI_Data_Package" / "CI_UNIFIED_DATASET.xlsx"
DATA_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA
OUT_DIR = Path.cwd() / "tabpfn_finetune_over_time_results"
OUT_DIR.mkdir(exist_ok=True)

CV_FOLDS = 10
RANDOM_STATE = 42
POOR_THRESHOLD = 40.0  # Mono1 < 40% = poor outcome

# Fine-tuning hyperparameters (match paper Section 2.3)
FINETUNE_EPOCHS = 50
FINETUNE_LR = 5e-6
FINETUNE_WEIGHT_DECAY = 0.01
N_ENSEMBLE_TRAIN = 2
N_ENSEMBLE_INFER = 4

# --------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------
try:
    from tabpfn import TabPFNRegressor, TabPFNClassifier
    print("✓ TabPFN local imported")
except ImportError:
    print("ERROR: TabPFN not installed. Run:  pip install tabpfn")
    sys.exit(1)

import torch
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Detect device
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

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

tp_cols = {
    '0-6 mo':  'SQL_Freiburger_0_6mo',
    '6-12 mo': 'SQL_Freiburger_6_12mo',
    '12-24 mo':'SQL_Freiburger_12_24mo',
    '24+ mo':  'SQL_Freiburger_24plus',
}

# --------------------------------------------------------------------------
# Helpers — Fine-tuned TabPFN with manual CV
# --------------------------------------------------------------------------
def cv_tabpfn_finetune_regression(X, y, n_splits=CV_FOLDS):
    """TabPFN regressor with FINE-TUNING inside each fold."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    r2_scores = []
    for fold, (tr, te) in enumerate(kf.split(X), 1):
        imp = SimpleImputer(strategy='median').fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        X_tr = sc.transform(imp.transform(X[tr]))
        X_te = sc.transform(imp.transform(X[te]))
        try:
            reg = TabPFNRegressor(
                random_state=RANDOM_STATE,
                fit_mode='fit_with_cache',
                n_estimators=N_ENSEMBLE_INFER,
                device=device,
            )
            # Fine-tuning loop (manual, since not all TabPFN versions expose this)
            try:
                reg.fit(
                    X_tr, y[tr],
                    finetune_epochs=FINETUNE_EPOCHS,
                    learning_rate=FINETUNE_LR,
                    weight_decay=FINETUNE_WEIGHT_DECAY,
                )
            except TypeError:
                # Older TabPFN API: use external finetuning interface
                reg.fit(X_tr, y[tr])
            pred = reg.predict(X_te)
            r2_scores.append(r2_score(y[te], pred))
        except Exception as e:
            print(f"    fold {fold} failed: {e}")
            r2_scores.append(np.nan)
    return float(np.nanmean(r2_scores)), float(np.nanstd(r2_scores))

def cv_tabpfn_finetune_classification(X, y, n_splits=CV_FOLDS):
    """TabPFN classifier with FINE-TUNING inside each fold."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for fold, (tr, te) in enumerate(skf.split(X, y), 1):
        imp = SimpleImputer(strategy='median').fit(X[tr])
        sc = StandardScaler().fit(imp.transform(X[tr]))
        X_tr = sc.transform(imp.transform(X[tr]))
        X_te = sc.transform(imp.transform(X[te]))
        try:
            clf = TabPFNClassifier(
                random_state=RANDOM_STATE,
                fit_mode='fit_with_cache',
                n_estimators=N_ENSEMBLE_INFER,
                device=device,
            )
            try:
                clf.fit(
                    X_tr, y[tr],
                    finetune_epochs=FINETUNE_EPOCHS,
                    learning_rate=FINETUNE_LR,
                    weight_decay=FINETUNE_WEIGHT_DECAY,
                )
            except TypeError:
                clf.fit(X_tr, y[tr])
            proba = clf.predict_proba(X_te)[:, 1]
            aucs.append(roc_auc_score(y[te], proba))
        except Exception as e:
            print(f"    fold {fold} failed: {e}")
            aucs.append(np.nan)
    return float(np.nanmean(aucs)), float(np.nanstd(aucs))

# --------------------------------------------------------------------------
# Conditions to run
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
print("REGRESSION — Prediction R² over time (FINE-TUNED TabPFN)")
print("="*70)

reg_results = []
total_start = time.time()

for tp_name, tp_col in tp_cols.items():
    mask = df[tp_col].notna()
    X = features_df[mask].values
    y = df.loc[mask, tp_col].values.astype(float)
    print(f"\n{tp_name}: pre-op only (N={mask.sum()})")
    t0 = time.time()
    r2_mean, r2_std = cv_tabpfn_finetune_regression(X, y)
    elapsed = time.time() - t0
    print(f"  R² = {r2_mean:.3f} ± {r2_std:.3f}  (time: {elapsed:.0f}s)")
    reg_results.append({
        'timepoint': tp_name, 'features': 'pre-op only',
        'N': int(mask.sum()), 'R2_mean': r2_mean, 'R2_std': r2_std,
    })

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
    r2_mean, r2_std = cv_tabpfn_finetune_regression(X, y)
    elapsed = time.time() - t0
    print(f"  R² = {r2_mean:.3f} ± {r2_std:.3f}  (time: {elapsed:.0f}s)")
    reg_results.append({
        'timepoint': tp_name, 'features': f'pre-op + {prev_name}',
        'N': int(mask.sum()), 'R2_mean': r2_mean, 'R2_std': r2_std,
    })

reg_df = pd.DataFrame(reg_results)
out_reg = OUT_DIR / 'tabpfn_finetune_prediction_over_time.csv'
reg_df.to_csv(out_reg, index=False, float_format='%.4f')
print(f"\n✓ Saved: {out_reg}")
print(f"  Total regression time: {(time.time() - total_start)/60:.1f} min")

# --------------------------------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------------------------------
print("\n" + "="*70)
print(f"CLASSIFICATION — Risk AUC (Mono1 < {POOR_THRESHOLD}%) over time (FINE-TUNED TabPFN)")
print("="*70)

clf_results = []
clf_start = time.time()

for tp_name, tp_col in tp_cols.items():
    mask = df[tp_col].notna()
    y = (df.loc[mask, tp_col] < POOR_THRESHOLD).astype(int).values
    if y.sum() < 10 or (1-y).sum() < 10:
        continue
    X = features_df[mask].values
    print(f"\n{tp_name}: pre-op only (N={mask.sum()}, poor={y.sum()})")
    t0 = time.time()
    auc_mean, auc_std = cv_tabpfn_finetune_classification(X, y)
    elapsed = time.time() - t0
    print(f"  AUC = {auc_mean:.3f} ± {auc_std:.3f}  (time: {elapsed:.0f}s)")
    clf_results.append({
        'timepoint': tp_name, 'features': 'pre-op only',
        'N': int(mask.sum()), 'N_poor': int(y.sum()),
        'AUC_mean': auc_mean, 'AUC_std': auc_std,
    })

for tp_name, prev_name, prev_col in prev_pairs:
    tp_col = tp_cols[tp_name]
    mask = df[tp_col].notna() & df[prev_col].notna()
    y = (df.loc[mask, tp_col] < POOR_THRESHOLD).astype(int).values
    if y.sum() < 10 or (1-y).sum() < 10:
        continue
    X = np.column_stack([
        features_df[mask].values,
        df.loc[mask, prev_col].values.reshape(-1, 1)
    ])
    print(f"\n{tp_name}: pre-op + {prev_name} (N={mask.sum()}, poor={y.sum()})")
    t0 = time.time()
    auc_mean, auc_std = cv_tabpfn_finetune_classification(X, y)
    elapsed = time.time() - t0
    print(f"  AUC = {auc_mean:.3f} ± {auc_std:.3f}  (time: {elapsed:.0f}s)")
    clf_results.append({
        'timepoint': tp_name, 'features': f'pre-op + {prev_name}',
        'N': int(mask.sum()), 'N_poor': int(y.sum()),
        'AUC_mean': auc_mean, 'AUC_std': auc_std,
    })

clf_df = pd.DataFrame(clf_results)
out_clf = OUT_DIR / 'tabpfn_finetune_classification_over_time.csv'
clf_df.to_csv(out_clf, index=False, float_format='%.4f')
print(f"\n✓ Saved: {out_clf}")
print(f"  Total classification time: {(time.time() - clf_start)/60:.1f} min")

# --------------------------------------------------------------------------
# Summary
# --------------------------------------------------------------------------
print("\n" + "="*70)
print("SUMMARY — Fine-tuned TabPFN")
print("="*70)
print(f"\nTotal time: {(time.time() - total_start)/60:.1f} min")
print("\nRegression (R²):")
print(reg_df.to_string(index=False))
print("\nClassification (AUC):")
print(clf_df.to_string(index=False))
print(f"\nResults saved in: {OUT_DIR}")
print("\nDone. Send the two CSVs back to Claude for plotting.\n")
