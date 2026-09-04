#!/usr/bin/env python3
"""
run_tabpfn_ci.py — TabPFN analysis for CI Equalizer paper (poor outcome classification)

Runs:
  1. Cohort filtering (N=378 adults, SSD-excluded, post-op Mono1 available)
  2. TabPFN zero-shot AUC on 4 feature subsets:
       - Equalizer-only (audiometric + aided + pre-op speech)
       - Discriminator-only (9 vars from Wasserstein analysis)
       - Persistence-only (demographic + clinical + communication)
       - All combined (49 features)
  3. TabPFN permutation importance on full 49-feature set
  4. Outputs: tabpfn_AUC_by_category.csv + tabpfn_importance.csv

Requirements:
  - Python 3.9+
  - tabpfn (with valid TABPFN_TOKEN)
  - pandas, numpy, scikit-learn, openpyxl

Usage:
  export TABPFN_TOKEN="..."
  python3 run_tabpfn_ci.py /path/to/CI_UNIFIED_DATASET.xlsx
"""

import sys
import time
import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.inspection import permutation_importance

if 'TABPFN_TOKEN' not in os.environ or not os.environ['TABPFN_TOKEN']:
    print("ERROR: TABPFN_TOKEN environment variable not set.")
    print("Run: export TABPFN_TOKEN=\"...\"")
    sys.exit(1)

from tabpfn import TabPFNClassifier

# ============ CONFIG ============
if len(sys.argv) < 2:
    DATA_PATH = "CI_UNIFIED_DATASET.xlsx"
else:
    DATA_PATH = sys.argv[1]

OUTPUT_DIR = os.path.dirname(os.path.abspath(DATA_PATH))
RANDOM_SEED = 42
N_FOLDS = 10

print(f"Data: {DATA_PATH}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Seed: {RANDOM_SEED}, folds: {N_FOLDS}")
print("=" * 60)

# ============ LOAD + COHORT FILTERING ============
print("\n[1/5] Loading data and applying cohort filter...")
t0 = time.time()
df = pd.read_excel(DATA_PATH, sheet_name='Sheet1')
df['Age_at_OP'] = pd.to_numeric(df['Age_at_OP'], errors='coerce')
adults = df[df['Age_at_OP'] >= 18].copy()

co_freqs = ['prPTA_500_Co', 'prPTA_1000_Co', 'prPTA_2000_Co', 'prPTA_4000_Co']
for c in co_freqs:
    adults[c] = pd.to_numeric(adults[c], errors='coerce')
adults['Co_PTA4'] = adults[co_freqs].mean(axis=1)

non_ssd = adults[(adults['Co_PTA4'] > 30) | (adults['Co_PTA4'].isna())].copy()
non_ssd['Mono1_post'] = pd.to_numeric(non_ssd['Mono1_post'], errors='coerce')
cohort = non_ssd[non_ssd['Mono1_post'].notna()].copy()
cohort['poor_outcome'] = (cohort['Mono1_post'] < 40).astype(int)

print(f"   Cohort: N={len(cohort)}, poor outcome={cohort['poor_outcome'].sum()} ({cohort['poor_outcome'].mean()*100:.1f}%)")
print(f"   Done in {time.time()-t0:.1f}s")

# ============ FEATURE SETS ============
print("\n[2/5] Defining feature sets (Equalizer / Discriminator / Persistence)...")

# CI audiogram
ci_freqs = ['prPTA_125_CI','prPTA_250_CI','prPTA_500_CI','prPTA_1000_CI',
            'prPTA_2000_CI','prPTA_4000_CI','prPTA_8000_CI']
# Co audiogram
co_freqs_a = ['prPTA_125_Co','prPTA_250_Co','prPTA_500_Co','prPTA_1000_Co',
              'prPTA_2000_Co','prPTA_4000_Co','prPTA_8000_Co']
# Aided field
aided = sorted([c for c in cohort.columns if 'FF_' in c])
# Pre-op speech tests
speech_pre = [c for c in ['Mono1_pre','Mono2_pre','V08_pre','C12_pre','FM_pre','Num_pre']
              if c in cohort.columns]

# EQUALIZER: audiometric + aided + pre-op speech
equalizer_cols = ci_freqs + co_freqs_a + aided + speech_pre
equalizer_cols = [c for c in equalizer_cols if c in cohort.columns]

# DISCRIMINATOR: 9 vars from Wasserstein analysis
discriminator_cols = ['EVA_DEAF_ONSET', 'EVA_DEAF_ONSET_L', 'Sur_surg_access',
                      'Sur_insertion_techn', 'Sur_incision_type', 'COM_Gebaerden',
                      'COM_MULTILANG_HOME', 'SES_PROFESSION_LEARNED', 'COM_FATHER_HACI_USER']
discriminator_cols = [c for c in discriminator_cols if c in cohort.columns]

# PERSISTENCE: demographic + clinical + communication
persistence_cols = ['Age_at_OP', 'Geschlecht', 'EVA_DURATION_HEARING_LOSS',
                    'EVA_PROGRESSIVE_HL', 'EVA_BILATERAL_CI', 'COM_LIP_READING',
                    'COM_NATIVE_LANGUAGE', 'COM_PHONE_USE', 'COM_ARTICULATION',
                    'EDU_EDUCATION', 'EDU_LIVING_SITUATION', 'INS_INSURANCE',
                    'EVA_HL_TYPE']
persistence_cols = [c for c in persistence_cols if c in cohort.columns]

all_cols = equalizer_cols + discriminator_cols + persistence_cols

print(f"   Equalizer: {len(equalizer_cols)} features")
print(f"   Discriminator: {len(discriminator_cols)} features")
print(f"   Persistence: {len(persistence_cols)} features")
print(f"   All combined: {len(all_cols)} features")

# ============ HELPER: encode + impute ============
def encode(s):
    s = pd.Series(s)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors='coerce')
    mask = s.notna()
    out = pd.Series(np.nan, index=s.index)
    if mask.sum() > 0:
        le = LabelEncoder()
        out[mask] = le.fit_transform(s[mask].astype(str))
    return out

def build_X(cols):
    return pd.DataFrame({c: encode(cohort[c]) for c in cols if c in cohort.columns})

y = cohort['poor_outcome'].astype(int).values
imputer = SimpleImputer(strategy='median')

# ============ TabPFN AUC by CATEGORY ============
print("\n[3/5] Running TabPFN zero-shot AUC by category (10-fold CV)...")
print("   This may take 5-15 minutes per category on CPU.")

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

cat_results = []
for cat_name, cols in [('Equalizer', equalizer_cols),
                        ('Discriminator', discriminator_cols),
                        ('Persistence', persistence_cols),
                        ('All', all_cols)]:
    print(f"\n   --- {cat_name} ({len(cols)} features) ---")
    t1 = time.time()
    X_cat = build_X(cols)
    X_imp = imputer.fit_transform(X_cat)
    
    tabpfn = TabPFNClassifier(device='cpu', n_estimators=4, ignore_pretraining_limits=True)
    auc = cross_val_score(tabpfn, X_imp, y, cv=skf, scoring='roc_auc', n_jobs=1)
    
    cat_results.append({
        'Category': cat_name,
        'N_features': len(cols),
        'TabPFN_AUC_mean': auc.mean(),
        'TabPFN_AUC_SD': auc.std(),
    })
    print(f"      AUC = {auc.mean():.3f} ± {auc.std():.3f}  ({time.time()-t1:.0f}s)")

cat_df = pd.DataFrame(cat_results)
out_path = os.path.join(OUTPUT_DIR, 'tabpfn_AUC_by_category.csv')
cat_df.to_csv(out_path, index=False, float_format='%.4f')
print(f"\n   Saved: {out_path}")

# ============ TabPFN PERMUTATION IMPORTANCE ============
print("\n[4/5] Running TabPFN permutation importance (n_repeats=10)...")
print("   This may take 30-60 minutes on CPU.")
t2 = time.time()

X_all = build_X(all_cols)
X_all_imp = imputer.fit_transform(X_all)

tabpfn_full = TabPFNClassifier(device='cpu', n_estimators=4, ignore_pretraining_limits=True)
tabpfn_full.fit(X_all_imp, y)
print(f"   Trained in {time.time()-t2:.0f}s. Computing permutation importance...")

t3 = time.time()
perm = permutation_importance(tabpfn_full, X_all_imp, y,
                               n_repeats=10, random_state=RANDOM_SEED,
                               n_jobs=1, scoring='roc_auc')
print(f"   Permutation importance done in {time.time()-t3:.0f}s")

# Trichotomy mapping
def trichotomy(v):
    if v in equalizer_cols: return 'Equalizer'
    if v in discriminator_cols: return 'Discriminator'
    return 'Persistence'

imp_df = pd.DataFrame({
    'Variable': all_cols,
    'Category': [trichotomy(v) for v in all_cols],
    'TabPFN_perm_importance': perm.importances_mean,
    'TabPFN_perm_SD': perm.importances_std,
})
imp_df['TabPFN_rank'] = imp_df['TabPFN_perm_importance'].rank(ascending=False, method='min').astype(int)
imp_df = imp_df.sort_values('TabPFN_rank').reset_index(drop=True)

out_imp_path = os.path.join(OUTPUT_DIR, 'tabpfn_importance.csv')
imp_df.to_csv(out_imp_path, index=False, float_format='%.5f')
print(f"   Saved: {out_imp_path}")

# ============ SUMMARY ============
print("\n[5/5] === FINAL SUMMARY ===\n")

print("AUC by category (10-fold CV):")
print(cat_df.to_string(index=False))

print("\n\nTOP 15 PREDICTORS by TabPFN permutation importance:")
print(imp_df.head(15)[['Variable','Category','TabPFN_perm_importance','TabPFN_rank']].to_string(index=False))

print("\n\nDONE.")
print(f"Total runtime: {(time.time()-t0)/60:.1f} minutes")
print(f"\nOutput files:")
print(f"  - {out_path}")
print(f"  - {out_imp_path}")
print(f"\nSend these to Claude tomorrow to integrate into the manuscript SI table.")
