"""
TabPFN binary classification of poor outcome (Mono1 < 40%) at 3 post-op
timepoints under 2 feature configurations, parallel to the regression
analysis of §3.3.2 (Table tab:r2_over_time).

Configurations:
  (i)  pre-op only (41 features)
  (ii) pre-op + 4 contemporaneous post-op speech tests (45 features)

Timepoints: 0-6, 6-12, 12-24 months

Run on Mac:
    python run_tabpfn_classifier_over_time.py CI_UNIFIED_DATASET.xlsx

Output: tabpfn_classifier_tp/auc_over_time.csv
Expected runtime: ~15-20 min on Mac mps.
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

try:
    from tabpfn import TabPFNClassifier
    print("TabPFN imported")
except ImportError:
    print("pip install tabpfn")
    sys.exit(1)

DEVICE = 'mps' if sys.platform == 'darwin' else 'cuda'
print(f"Device: {DEVICE}")

XLSX = sys.argv[1] if len(sys.argv) > 1 else 'CI_UNIFIED_DATASET.xlsx'
OUT = Path('tabpfn_classifier_tp')
OUT.mkdir(exist_ok=True)

# ---------- 1. Load + cohort ----------
df = pd.read_excel(XLSX, sheet_name='Sheet1')
df['Age_at_OP'] = pd.to_numeric(df['Age_at_OP'], errors='coerce')
adults = df[df['Age_at_OP'] >= 18].copy().reset_index(drop=True)
adults['months_since_OP'] = (adults['Mono1_poDat'] - adults['Datum_ErstesCI']).dt.days / 30.44

# ---------- 2. Build feature lists (matching §3.3.2 regression analysis) ----------
ci_freqs = [c for c in adults.columns if c.startswith('prPTA_') and c.endswith('_CI')]
co_freqs = [c for c in adults.columns if c.startswith('prPTA_') and c.endswith('_Co')]
aided = sorted([c for c in adults.columns if c.startswith('prFF_')])
speech_pre = ['Mono1_pre','Mono2_pre','V08_pre','C12_pre','FM_pre','Num_pre']
demo = ['Age_at_OP']
cat_cols_raw = ['Geschlecht','EVA_DEAF_ONSET','COM_ARTICULATION','EVA_HL_PROGREDIENT']
for c in cat_cols_raw:
    if c in adults.columns:
        adults[c+'_enc'] = LabelEncoder().fit_transform(adults[c].astype(str).fillna('NA'))
cat_cols = [c+'_enc' for c in cat_cols_raw if c+'_enc' in adults.columns]
preop_features = demo + ci_freqs + co_freqs + aided + speech_pre + cat_cols
preop_features = [f for f in preop_features if f in adults.columns]

earlier_post = ['FM_post','Num_post','V08_post','C12_post']
for c in earlier_post:
    if c in adults.columns:
        adults[c] = pd.to_numeric(adults[c], errors='coerce')
earlier_post = [c for c in earlier_post if c in adults.columns]
all_features = preop_features + earlier_post

print(f"Pre-op features: {len(preop_features)}")
print(f"Contemporaneous post-op tests: {len(earlier_post)}")
print(f"Total (pre-op + contemporaneous): {len(all_features)}")

# Encode categoricals (StringArray-aware)
for c in preop_features + earlier_post:
    if c not in adults.columns: continue
    s = adults[c]
    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.notna().sum() == s.notna().sum():
        adults[c] = s_num
    else:
        s_obj = s.astype('object')
        mask = s_obj.notna()
        unique_vals = sorted(s_obj[mask].unique().tolist(), key=str)
        mapping = {v: i for i, v in enumerate(unique_vals)}
        adults[c] = s_obj.map(mapping).astype(float)

# ---------- 3. Bin timepoints ----------
def bin_tp(m):
    if m <= 6: return '0-6'
    if m <= 12: return '6-12'
    if m <= 24: return '12-24'
    return None
adults['tp'] = adults['months_since_OP'].apply(bin_tp)

# ---------- 4. CV AUC ----------
def auc_cv_tabpfn(X, y, n_splits=5, n_seeds=3):
    aucs_per_seed = []
    for seed in range(n_seeds):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aucs = []
        for tr, te in cv.split(X, y):
            imp = SimpleImputer(strategy='median')
            X_tr = imp.fit_transform(X[tr])
            X_te = imp.transform(X[te])
            clf = TabPFNClassifier(device=DEVICE, ignore_pretraining_limits=True)
            clf.fit(X_tr, y[tr])
            try:
                proba = clf.predict_proba(X_te)[:, 1]
                fold_aucs.append(roc_auc_score(y[te], proba))
            except Exception as e:
                print(f"    fold failed: {e}")
        if fold_aucs:
            aucs_per_seed.append(np.mean(fold_aucs))
    if not aucs_per_seed:
        return np.nan, np.nan
    return np.mean(aucs_per_seed), np.std(aucs_per_seed)

# ---------- 5. Loop over timepoints × configs ----------
results = []
configs = {
    'preop_only':            preop_features,
    'preop+contemporaneous': all_features,
}

print(f"\n{'Timepoint':<10} {'Config':<25} {'N':>5} {'poor':>5} {'AUC':>15} {'sec':>5}")
print('-' * 70)

for tp in ['0-6','6-12','12-24']:
    sub = adults[(adults['tp']==tp) & adults['Mono1_post'].notna()].copy().reset_index(drop=True)
    # Require all 4 contemporaneous tests for fair comparison across configs
    for c in earlier_post:
        sub = sub[sub[c].notna()]
    sub = sub.reset_index(drop=True)

    if len(sub) < 20:
        print(f"{tp:<10} skipped (N={len(sub)} too small)")
        continue
    y = (sub['Mono1_post'] < 40).astype(int).values
    n_poor = int(y.sum())
    if n_poor < 3:
        print(f"{tp:<10} skipped (only {n_poor} poor outcomes)")
        continue

    # Adapt n_splits to small n_poor
    n_splits = min(5, n_poor)

    for cfg_name, feats in configs.items():
        feats_avail = [f for f in feats if f in sub.columns]
        X = sub[feats_avail].values
        t0 = time.time()
        auc_m, auc_s = auc_cv_tabpfn(X, y, n_splits=n_splits, n_seeds=3)
        elapsed = time.time() - t0
        results.append({
            'timepoint': tp, 'config': cfg_name,
            'N': len(sub), 'n_poor': n_poor,
            'n_features': len(feats_avail),
            'auc_mean': auc_m, 'auc_std': auc_s,
            'sec': round(elapsed, 1),
        })
        print(f"{tp:<10} {cfg_name:<25} {len(sub):>5} {n_poor:>5} {auc_m:.3f} ± {auc_s:.3f}  {elapsed:>4.0f}")

R = pd.DataFrame(results)
R.to_csv(OUT / 'auc_over_time.csv', index=False)
print(f"\nSaved {OUT}/auc_over_time.csv")
print("\n=== FINAL TABLE ===")
print(R.to_string(index=False))
