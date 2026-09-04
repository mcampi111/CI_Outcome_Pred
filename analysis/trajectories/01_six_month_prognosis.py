"""
Two analyses to complete §3.3.2:

ANALYSIS A: Classification permutation importance at 3 timepoints
   For each (timepoint, config), compute permutation importance of TabPFN
   classifier (poor outcome Mono1 < 40%). Output: per-feature delta-AUC.
   Used to build the variable importance figure for the classification task.

ANALYSIS B: Sample size check for longitudinal analysis
   How many patients have BOTH a Mono1 at 0-6 or 6-12 mo (early post-op)
   AND a Mono1 at 12-24 mo (late)? If >= 50, we can add a new analysis:
   "do early post-op measurements predict long-term outcome?".

Run on Mac:
    python run_classification_imp_and_longitudinal_check.py CI_UNIFIED_DATASET.xlsx

Outputs:
    classification_imp/perm_imp_classification.csv
    classification_imp/longitudinal_sample_check.txt
    classification_imp/perm_imp_classification.pkl
"""

import sys
import os
import time
import warnings
warnings.filterwarnings('ignore')
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
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
OUT = Path('classification_imp')
OUT.mkdir(exist_ok=True)

# ============================================================
# 1. Load + cohort (same as before)
# ============================================================
df = pd.read_excel(XLSX, sheet_name='Sheet1')
df['Age_at_OP'] = pd.to_numeric(df['Age_at_OP'], errors='coerce')
adults = df[df['Age_at_OP'] >= 18].copy().reset_index(drop=True)
adults['months_since_OP'] = (adults['Mono1_poDat'] - adults['Datum_ErstesCI']).dt.days / 30.44

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

# Encode categoricals
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

def bin_tp(m):
    if m <= 6: return '0-6'
    if m <= 12: return '6-12'
    if m <= 24: return '12-24'
    return None
adults['tp'] = adults['months_since_OP'].apply(bin_tp)

# ============================================================
# ANALYSIS A: Permutation importance for classification
# ============================================================
print("\n" + "="*60)
print("ANALYSIS A: Classification permutation importance over time")
print("="*60)

def perm_imp_clf(X_arr, y_arr, feature_names, n_repeats=20, random_state=42):
    """Permutation importance based on AUC drop."""
    Xtr, Xte, ytr, yte = train_test_split(X_arr, y_arr, test_size=0.3,
                                           random_state=random_state, stratify=y_arr)
    imp = SimpleImputer(strategy='median')
    Xtr_i = imp.fit_transform(Xtr)
    Xte_i = imp.transform(Xte)
    
    clf = TabPFNClassifier(device=DEVICE, ignore_pretraining_limits=True)
    clf.fit(Xtr_i, ytr)
    base_proba = clf.predict_proba(Xte_i)[:, 1]
    base_auc = roc_auc_score(yte, base_proba)
    
    rng = np.random.default_rng(random_state)
    deltas = np.zeros(X_arr.shape[1])
    for j, fname in enumerate(feature_names):
        d_per_rep = []
        for r in range(n_repeats):
            Xte_perm = Xte_i.copy()
            Xte_perm[:, j] = rng.permutation(Xte_perm[:, j])
            proba_perm = clf.predict_proba(Xte_perm)[:, 1]
            try:
                auc_perm = roc_auc_score(yte, proba_perm)
            except:
                auc_perm = 0.5
            d_per_rep.append(base_auc - auc_perm)
        deltas[j] = np.mean(d_per_rep)
    return deltas, base_auc

results_clf_imp = {}
configs = {
    'preop_only':            preop_features,
    'preop+contemporaneous': all_features,
}

for tp in ['0-6','6-12','12-24']:
    sub = adults[(adults['tp']==tp) & adults['Mono1_post'].notna()].copy().reset_index(drop=True)
    for c in earlier_post:
        sub = sub[sub[c].notna()]
    sub = sub.reset_index(drop=True)
    
    if len(sub) < 25:
        print(f"\n{tp}: skipped (N={len(sub)} too small)")
        continue
    y = (sub['Mono1_post'] < 40).astype(int).values
    n_poor = int(y.sum())
    if n_poor < 5:
        print(f"\n{tp}: skipped (only {n_poor} poor outcomes)")
        continue
    
    print(f"\n--- {tp} mo (N={len(sub)}, poor={n_poor}) ---")
    
    for cfg_name, feats in configs.items():
        feats_avail = [f for f in feats if f in sub.columns]
        X = sub[feats_avail].values
        t0 = time.time()
        deltas, base_auc = perm_imp_clf(X, y, feats_avail)
        elapsed = time.time() - t0
        results_clf_imp[(tp, cfg_name)] = {
            'features': feats_avail,
            'deltas': deltas,
            'base_auc': base_auc,
            'N': len(sub),
            'n_poor': n_poor,
        }
        print(f"  {cfg_name}: base AUC={base_auc:.3f}, top 5:")
        order = np.argsort(-deltas)[:5]
        for k in order:
            print(f"    {deltas[k]:+.4f}  {feats_avail[k]}")
        print(f"    ({elapsed:.0f}s)")

# Save as DataFrame and pickle
rows = []
for (tp, cfg), v in results_clf_imp.items():
    for f, d in zip(v['features'], v['deltas']):
        rows.append({'timepoint':tp, 'config':cfg, 'N':v['N'], 'n_poor':v['n_poor'],
                     'base_auc':v['base_auc'], 'feature':f, 'delta_auc':d})
R = pd.DataFrame(rows)
R.to_csv(OUT / 'perm_imp_classification.csv', index=False)
with open(OUT / 'perm_imp_classification.pkl', 'wb') as fh:
    pickle.dump(results_clf_imp, fh)
print(f"\nSaved {OUT}/perm_imp_classification.csv and .pkl")

# ============================================================
# ANALYSIS B: Longitudinal sample check
# ============================================================
print("\n" + "="*60)
print("ANALYSIS B: Longitudinal sample check")
print("="*60)

# For each patient, find all their post-op Mono1 measurements
# We need: at least one Mono1 at early post-op (0-6 or 6-12) AND one at late (12-24).
# Question: is Mono1_post a single column or multiple measurements per patient?

# Quick inspection: how many rows per patient?
if 'Pat_ID' in adults.columns:
    id_col = 'Pat_ID'
elif 'Patient_ID' in adults.columns:
    id_col = 'Patient_ID'
else:
    # try to find an ID-like column
    candidates = [c for c in adults.columns if 'id' in c.lower() or 'pat' in c.lower()]
    print(f"Possible ID columns: {candidates}")
    id_col = None

report_lines = []
report_lines.append("Longitudinal sample check\n" + "="*40 + "\n")

if id_col is None:
    msg = "No clear patient ID column found. Each row is treated as one measurement."
    print(msg); report_lines.append(msg + "\n")
    # Just count by timepoint based on existing rows
    for tp in ['0-6','6-12','12-24']:
        n = ((adults['tp']==tp) & adults['Mono1_post'].notna()).sum()
        msg = f"  Mono1 at {tp} mo: N = {n}"
        print(msg); report_lines.append(msg + "\n")
else:
    # Count unique patients with measurements at each timepoint
    print(f"\nUsing patient ID column: {id_col}\n")
    report_lines.append(f"Using patient ID column: {id_col}\n\n")
    
    # Patients with any post-op Mono1
    n_total = adults[adults['Mono1_post'].notna()][id_col].nunique()
    msg = f"Total adults with at least one post-op Mono1: {n_total}"
    print(msg); report_lines.append(msg + "\n")
    
    # Per timepoint
    for tp in ['0-6','6-12','12-24']:
        n = adults[(adults['tp']==tp) & adults['Mono1_post'].notna()][id_col].nunique()
        msg = f"  Patients with Mono1 at {tp} mo: {n}"
        print(msg); report_lines.append(msg + "\n")
    
    # Patients with measurements at MULTIPLE timepoints
    pat_tps = adults[adults['Mono1_post'].notna()].groupby(id_col)['tp'].apply(set)
    
    n_early_and_late = sum(({'0-6','12-24'}.issubset(s) or {'6-12','12-24'}.issubset(s)) for s in pat_tps)
    msg = f"\nPatients with EARLY (0-6 or 6-12) AND LATE (12-24) Mono1: {n_early_and_late}"
    print(msg); report_lines.append(msg + "\n")
    
    n_06_and_late = sum({'0-6','12-24'}.issubset(s) for s in pat_tps)
    msg = f"  Specifically: 0-6 AND 12-24: {n_06_and_late}"
    print(msg); report_lines.append(msg + "\n")
    
    n_612_and_late = sum({'6-12','12-24'}.issubset(s) for s in pat_tps)
    msg = f"  Specifically: 6-12 AND 12-24: {n_612_and_late}"
    print(msg); report_lines.append(msg + "\n")
    
    msg = "\nAssessment:"
    print(msg); report_lines.append(msg + "\n")
    if n_early_and_late >= 50:
        msg = f"  >= 50 patients with paired early+late: longitudinal analysis IS feasible."
    elif n_early_and_late >= 30:
        msg = f"  30-50 patients: longitudinal analysis is borderline; would yield wide CIs."
    else:
        msg = f"  < 30 patients: NOT enough for a clean longitudinal analysis."
    print(msg); report_lines.append(msg + "\n")

with open(OUT / 'longitudinal_sample_check.txt', 'w') as fh:
    fh.writelines(report_lines)
print(f"\nSaved {OUT}/longitudinal_sample_check.txt")

print("\n" + "="*60)
print("DONE. Send back:")
print(f"  - {OUT}/perm_imp_classification.csv")
print(f"  - {OUT}/longitudinal_sample_check.txt")
print("="*60)
