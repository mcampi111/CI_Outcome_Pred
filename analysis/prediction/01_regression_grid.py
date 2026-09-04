#!/usr/bin/env python3
"""
run_tabpfn_FINAL.py — TabPFN regression grid + permutation importance for the paper.

Runs TabPFN on the new primary outcome (Mono1 ≥12 months, N=365):
  1. Regression grid: TabPFN (zero-shot) on 4 feature sets (F3, F7, F17, F60)
     under 10-fold CV → CV R² and MAE for each
  2. Permutation importance: TabPFN on F60 → ranks all 60 variables

Output:
  - tabpfn_regression_grid_FINAL.csv (4 rows: F3, F7, F17, F60)
  - tabpfn_importance_F60_FINAL.csv (60 rows ranked)

Usage on Mac:
  cd ~/Desktop/Postdoc_Zurich/DATACI/CI_Data_Package
  export TABPFN_TOKEN="<your token from https://ux.priorlabs.ai/account>"
  python3 run_tabpfn_FINAL.py CI_UNIFIED_DATASET.xlsx

Estimated runtime: ~15-25 minutes on CPU (10 folds × 4 feature sets + permutation).
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

if 'TABPFN_TOKEN' not in os.environ or not os.environ['TABPFN_TOKEN']:
    print("ERROR: TABPFN_TOKEN environment variable not set.")
    print("Run first: export TABPFN_TOKEN=\"<your token>\"")
    print("Get it from https://ux.priorlabs.ai/account")
    sys.exit(1)

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import permutation_importance
from tabpfn import TabPFNRegressor

# =============================================================================
# CONFIG
# =============================================================================
DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else 'CI_UNIFIED_DATASET.xlsx'
RANDOM_SEED = 42
N_FOLDS = 10
OUTPUT_DIR = os.path.dirname(os.path.abspath(DATA_PATH))

REG_OUT = os.path.join(OUTPUT_DIR, 'tabpfn_regression_grid_FINAL.csv')
IMP_OUT = os.path.join(OUTPUT_DIR, 'tabpfn_importance_F60_FINAL.csv')

print(f"Data: {DATA_PATH}")
print(f"Output dir: {OUTPUT_DIR}")
print(f"Seed: {RANDOM_SEED}, folds: {N_FOLDS}")
print("=" * 70)

# =============================================================================
# LOAD AND BUILD COHORT (N=365 primary)
# =============================================================================
print("\n[1/5] Loading and filtering cohort...")
t0 = time.time()
df = pd.read_excel(DATA_PATH)
df = df[df['Age_at_OP'] >= 18].copy()

# Exclude SSD: contralateral 4-freq PTA <= 30 dB HL
co_freqs = ['prPTA_500_Co', 'prPTA_1000_Co', 'prPTA_2000_Co', 'prPTA_4000_Co']
df['_co_pta'] = df[co_freqs].mean(axis=1)
df = df[~(df['_co_pta'] <= 30)].copy()
print(f"  Adults after SSD exclusion: {len(df)}")

# Build primary outcome: Mono1 longest available ≥12 months
def longest_followup_12plus(row):
    for col in ['SQL_Freiburger_24plus', 'SQL_Freiburger_12_24mo']:
        v = row[col]
        if pd.notna(v):
            return v
    return np.nan

df['Mono1_post_12plus'] = df.apply(longest_followup_12plus, axis=1)
df = df[df['Mono1_post_12plus'].notna()].copy()
print(f"  Primary outcome (Mono1 ≥12 mo): N = {len(df)}")
y = df['Mono1_post_12plus'].values
print(f"  Mean = {y.mean():.1f}%, SD = {y.std():.1f}%")

# =============================================================================
# DERIVED FEATURES (so derivations match the paper's definitions exactly)
# =============================================================================
print("\n[2/5] Building derived features...")

# 4-freq PTAs and FFs
df['PTA_CI_4freq'] = df[['prPTA_500_CI', 'prPTA_1000_CI', 'prPTA_2000_CI', 'prPTA_4000_CI']].mean(axis=1)
df['PTA_Co_4freq'] = df[['prPTA_500_Co', 'prPTA_1000_Co', 'prPTA_2000_Co', 'prPTA_4000_Co']].mean(axis=1)
df['FF_CI_4freq']  = df[['prFF_500_CI', 'prFF_1000_CI', 'prFF_2000_CI', 'prFF_4000_CI']].mean(axis=1)
df['FF_Co_4freq']  = df[['prFF_500_Co', 'prFF_1000_Co', 'prFF_2000_Co', 'prFF_4000_Co']].mean(axis=1)

# Sex
df['Sex_F'] = (df['Geschlecht'] == 'F').astype(int)

# Onset of deafness (ordinal: post=0, peri=1, pre=2)
df['onset_ord'] = df['EVA_DEAF_ONSET'].map({'post-ling': 0, 'peri-ling': 1, 'pre-ling': 2})

# Articulation (ordinal: normal=0, slightdist=1, strongdist=2)
df['articulation_ord'] = df['COM_ARTICULATION'].map({'normal': 0, 'slightdist': 1, 'strongdist': 2})

# Etiology (ordinal grouping)
def map_etio(e):
    if pd.isna(e): return np.nan
    e = str(e).lower()
    if 'unknown' in e: return 0
    if 'con' in e and 'syn' in e: return 1
    if 'acq' in e: return 2
    if 'sudden' in e: return 3
    if 'mening' in e or 'infect' in e: return 4
    return 5
df['etiology_ord'] = df['EVA_ETIOLOGY'].map(map_etio)

# Progressive HL
df['progressive_HL'] = (df['EVA_HL_PROGREDIENT'] == 'Y').astype(int)

# Duration of HL (years from total deafness date to OP date)
op_date = pd.to_datetime(df['Date_PTA_preOP_CI'], errors='coerce')
df['duration_HL'] = (op_date - df['EVA_DATE_TOTAL_DEAF']).dt.days / 365.25

# =============================================================================
# DEFINE 4 FEATURE SETS
# =============================================================================
print("\n[3/5] Defining feature sets (F3, F7, F17, F60)...")

F3 = ['Age_at_OP', 'Sex_F', 'PTA_CI_4freq']
F7 = F3 + ['onset_ord', 'duration_HL', 'etiology_ord', 'articulation_ord']

ci_indiv_freqs = ['prPTA_125_CI', 'prPTA_250_CI', 'prPTA_500_CI', 'prPTA_1000_CI',
                   'prPTA_2000_CI', 'prPTA_4000_CI', 'prPTA_6000_CI', 'prPTA_8000_CI']
F17 = F7 + ci_indiv_freqs + ['PTA_Co_4freq', 'FF_CI_4freq']

# F60: full pool — assemble exactly as in the paper Methods table
ci_pta = ['prPTA_125_CI', 'prPTA_250_CI', 'prPTA_500_CI', 'prPTA_1000_CI',
          'prPTA_2000_CI', 'prPTA_4000_CI', 'prPTA_6000_CI', 'prPTA_8000_CI']
co_pta = ['prPTA_125_Co', 'prPTA_250_Co', 'prPTA_500_Co', 'prPTA_1000_Co',
          'prPTA_2000_Co', 'prPTA_4000_Co', 'prPTA_6000_Co', 'prPTA_8000_Co']
ci_ff = ['prFF_250_CI', 'prFF_500_CI', 'prFF_1000_CI', 'prFF_2000_CI',
         'prFF_4000_CI', 'prFF_6000_CI', 'prFF_8000_CI']
co_ff = ['prFF_250_Co', 'prFF_500_Co', 'prFF_1000_Co', 'prFF_2000_Co',
         'prFF_4000_Co', 'prFF_6000_Co', 'prFF_8000_Co']
preop_speech = ['Mono1_pre', 'Mono2_pre', 'V08_pre', 'C12_pre', 'FM_pre', 'Num_pre']
demo = ['Age_at_OP', 'Sex_F']
hl_hist = ['onset_ord', 'duration_HL', 'etiology_ord', 'progressive_HL']
com_cols = [c for c in df.columns if c.startswith('COM_')][:8]
sur_cols = [c for c in df.columns if c.startswith('Sur_')]
sur_pick = [c for c in sur_cols if df[c].isna().mean() < 0.5][:3]
ses_cols = [c for c in df.columns if c.startswith('SES_')]
ses_pick = [c for c in ses_cols if df[c].isna().mean() < 0.5][:7]

F60 = ci_pta + co_pta + ci_ff + co_ff + preop_speech + demo + hl_hist + com_cols + sur_pick + ses_pick
F60 = list(dict.fromkeys(F60))  # remove dups, preserve order
F60 = F60[:60]  # ensure exactly 60

feature_sets = {'F3': F3, 'F7': F7, 'F17': F17, 'F60': F60}
for name, fs in feature_sets.items():
    print(f"  {name}: {len(fs)} features")

# =============================================================================
# HELPER: encode mixed-type features
# =============================================================================
def encode_features(df, cols):
    X = pd.DataFrame()
    for c in cols:
        if c not in df.columns:
            continue
        s = df[c]
        if pd.api.types.is_datetime64_any_dtype(s):
            X[c] = s.dt.year.astype(float)
        elif s.dtype == 'O':
            codes, _ = pd.factorize(s, use_na_sentinel=True)
            codes = codes.astype(float)
            codes[codes == -1] = np.nan
            X[c] = codes
        else:
            X[c] = s.astype(float)
    return X

# =============================================================================
# REGRESSION GRID — 4 feature sets × 10-fold CV
# =============================================================================
print(f"\n[4/5] TabPFN regression grid (10-fold CV)...")
print(f"  This takes ~3-5 minutes per feature set on CPU.")

reg_results = []
for fs_name, fs_cols in feature_sets.items():
    t1 = time.time()
    print(f"\n  Running TabPFN @ {fs_name} ({len(fs_cols)} features)...")
    
    X = encode_features(df, fs_cols).values
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    y_pred_cv = np.zeros_like(y, dtype=float)
    
    for fold_i, (tr, te) in enumerate(kf.split(X)):
        imp = SimpleImputer(strategy='median')
        sc = StandardScaler()
        X_tr = sc.fit_transform(imp.fit_transform(X[tr]))
        X_te = sc.transform(imp.transform(X[te]))
        
        model = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True, n_estimators=4)
        model.fit(X_tr, y[tr])
        y_pred_cv[te] = model.predict(X_te)
        print(f"    fold {fold_i+1}/{N_FOLDS} done", end='\r', flush=True)
    
    r2_cv = r2_score(y, y_pred_cv)
    mae_cv = mean_absolute_error(y, y_pred_cv)
    elapsed = time.time() - t1
    
    print(f"\n    {fs_name}: CV R² = {r2_cv:+.3f}, MAE = {mae_cv:.2f} pp  ({elapsed:.0f}s)")
    reg_results.append({
        'feature_set': fs_name,
        'n_features': len(fs_cols),
        'algorithm': 'TabPFN (zero-shot)',
        'r2_cv': r2_cv,
        'mae_cv': mae_cv,
    })

reg_df = pd.DataFrame(reg_results)
reg_df.to_csv(REG_OUT, index=False, float_format='%.4f')
print(f"\n  Saved regression grid: {REG_OUT}")

# =============================================================================
# PERMUTATION IMPORTANCE on F60
# =============================================================================
print(f"\n[5/5] TabPFN permutation importance (F60, n_repeats=10)...")
print(f"  This may take 10-20 minutes on CPU.")
t2 = time.time()

X_60 = encode_features(df, F60)
feature_names = X_60.columns.tolist()
imp = SimpleImputer(strategy='median')
sc = StandardScaler()
X_60_proc = sc.fit_transform(imp.fit_transform(X_60.values))

print(f"  Fitting TabPFN on full data ({len(y)} samples, {X_60_proc.shape[1]} features)...")
model_full = TabPFNRegressor(device='cpu', ignore_pretraining_limits=True, n_estimators=4)
model_full.fit(X_60_proc, y)
print(f"  Fitted in {time.time()-t2:.0f}s. Computing permutation importance...")

t3 = time.time()
perm = permutation_importance(model_full, X_60_proc, y,
                              n_repeats=10, random_state=RANDOM_SEED,
                              n_jobs=1, scoring='r2')
print(f"  Done in {time.time()-t3:.0f}s")

imp_df = pd.DataFrame({
    'Variable': feature_names,
    'TabPFN_perm_importance': perm.importances_mean,
    'TabPFN_perm_SD': perm.importances_std,
})
imp_df['TabPFN_rank'] = imp_df['TabPFN_perm_importance'].rank(ascending=False, method='min').astype(int)
imp_df = imp_df.sort_values('TabPFN_rank').reset_index(drop=True)
imp_df.to_csv(IMP_OUT, index=False, float_format='%.5f')
print(f"  Saved importance: {IMP_OUT}")

# =============================================================================
# SUMMARY
# =============================================================================
print("\n" + "="*70)
print("FINAL SUMMARY")
print("="*70)

print("\n>>> Regression grid (CV R²):")
for _, row in reg_df.iterrows():
    print(f"  {row['feature_set']:<5} ({row['n_features']:>2} features): R² = {row['r2_cv']:+.3f}, MAE = {row['mae_cv']:.2f} pp")

print("\n>>> Top 15 predictors (TabPFN permutation importance, F60):")
for _, row in imp_df.head(15).iterrows():
    print(f"  #{row['TabPFN_rank']:<3} {row['Variable']:<28}  imp = {row['TabPFN_perm_importance']:+.4f}")

total_time = (time.time() - t0) / 60
print(f"\nTotal runtime: {total_time:.1f} minutes")
print(f"\nFiles to send back to Claude:")
print(f"  1. {REG_OUT}")
print(f"  2. {IMP_OUT}")
