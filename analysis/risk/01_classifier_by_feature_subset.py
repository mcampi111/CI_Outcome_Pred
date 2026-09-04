"""
TabPFN binary classification by Wasserstein category subset.

Mirrors the LR/RF analysis already run, adding fine-tuned TabPFN as the
gold-standard ML benchmark for the AUC-by-category figure (Figure 3 v9).

Computes 10-fold stratified CV AUC for poor outcome (Mono1 < 40%) on three
feature subsets:
  - Decaying (n=6 robust decaying features)
  - Non-decaying (n=68: 38 stable + 30 labile)
  - All combined (n=82: 79 pre-op + 3 intra-op)

Run on Mac (mps) or DGX (cuda):
    python run_tabpfn_classifier_subsets.py CI_UNIFIED_DATASET.xlsx

Output: tabpfn_classifier/auc_by_subset.csv
"""

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# TabPFN
try:
    from tabpfn import TabPFNClassifier
    print("TabPFN imported")
except ImportError:
    print("pip install tabpfn")
    sys.exit(1)

DEVICE = 'mps' if sys.platform == 'darwin' else 'cuda'
print(f"Device: {DEVICE}")

XLSX = sys.argv[1] if len(sys.argv) > 1 else 'CI_UNIFIED_DATASET.xlsx'
OUT = Path('tabpfn_classifier')
OUT.mkdir(exist_ok=True)

# ---------- 1. Load + cohort ----------
df = pd.read_excel(XLSX, sheet_name='Sheet1')
df['Age_at_OP'] = pd.to_numeric(df['Age_at_OP'], errors='coerce')
adults = df[df['Age_at_OP'] >= 18].copy().reset_index(drop=True)
adults['months_since_OP'] = (adults['Mono1_poDat'] - adults['Datum_ErstesCI']).dt.days / 30.44

# Cohort: adults with post-op Mono1 (any timepoint)
cohort = adults[adults['Mono1_post'].notna()].copy().reset_index(drop=True)
y = (cohort['Mono1_post'] < 40).astype(int).values
print(f"Cohort N={len(cohort)}, poor outcomes={y.sum()} ({y.mean()*100:.1f}%)")

# ---------- 2. Build feature lists (same as Wasserstein analysis) ----------
audio = [c for c in df.columns if c.startswith('prPTA_')]
ff    = [c for c in df.columns if c.startswith('prFF_')]
bone  = [c for c in df.columns if c.startswith('prBONE_')]
speech_pre = ['Mono1_pre','Mono2_pre','V08_pre','C12_pre','FM_pre','Num_pre']
demo  = ['Age_at_OP','Geschlecht']
eva   = ['EVA_ETIOLOGY','EVA_DEAF_ONSET','EVA_HL_PROGREDIENT','EVA_SUBJ_HL_DIAG','EVA_SUBJ_CI_DIAG']
com   = ['COM_ARTICULATION','COM_MAIN_LANGUAGE','COM_MULTILANG_HOME','COM_PHONE_USE','COM_LIP_READING']
ses   = ['SES_EDUCATION_LEVEL','SES_MARITIAL_STATE','SES_HEALTH_INSURANCE','SES_WORKING']
sur_pre = ['Sur_duration','Sur_intracochl_access','Sur_elec_insertion','Sur_insertion_techn','Sur_elec_fixation','Sur_Gusher','Sur_num_of_not_ins_elec']
ct_features = ['CT_Cochl_malform','CT_Cochlear_otoscl','CT_Laby_ossi','CT_Mastoid_middle_ear_abnorm','CT_Brain_path','CT_Pathway_VII_abnorm','CT_Trauma','CT_Carotid_a_abnorm','CT_Sigmoid_sin_abnorm']
side = ['Seite']

# 79 features pool: pre-op + CT, NO intra-op
# (Intra-op pass/fail checks: ECAP elicitation, stapedius, impedance — excluded
#  because >95% positive in our cohort, no discriminative variability)
all_feats = [c for c in audio+ff+bone+speech_pre+demo+eva+com+ses+sur_pre+ct_features+side if c in cohort.columns]

# Encode categoricals (StringArray-aware)
for c in all_feats:
    s = cohort[c]
    s_num = pd.to_numeric(s, errors='coerce')
    if s_num.notna().sum() == s.notna().sum():
        cohort[c] = s_num
    else:
        s_obj = s.astype('object')
        mask = s_obj.notna()
        unique_vals = sorted(s_obj[mask].unique().tolist(), key=str)
        mapping = {v: i for i, v in enumerate(unique_vals)}
        cohort[c] = s_obj.map(mapping).astype(float)

# Subsets from Wasserstein classification
robust_decaying = ['Mono1_pre','Mono2_pre','V08_pre','C12_pre','Num_pre','prPTA_6000_Co']
robust_decaying = [f for f in robust_decaying if f in cohort.columns]

# Non-decaying = all features minus decaying
non_decaying = [f for f in all_feats if f not in robust_decaying]

subsets = {
    'Decaying':      robust_decaying,
    'Non-decaying':  non_decaying,
    'All combined':  all_feats,
}

print(f"\nSubsets:")
for name, feats in subsets.items():
    print(f"  {name}: {len(feats)} features")

# ---------- 3. CV AUC across 5 random seeds ----------
def auc_cv_tabpfn(X, y, n_splits=10, n_seeds=5):
    aucs_per_seed = []
    for seed in range(n_seeds):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        fold_aucs = []
        for fold, (tr, te) in enumerate(cv.split(X, y)):
            imp = SimpleImputer(strategy='median')
            X_tr = imp.fit_transform(X[tr])
            X_te = imp.transform(X[te])
            clf = TabPFNClassifier(device=DEVICE, ignore_pretraining_limits=True)
            clf.fit(X_tr, y[tr])
            proba = clf.predict_proba(X_te)[:, 1]
            fold_aucs.append(roc_auc_score(y[te], proba))
        aucs_per_seed.append(np.mean(fold_aucs))
    return np.mean(aucs_per_seed), np.std(aucs_per_seed)

results = []
for name, feats in subsets.items():
    print(f"\n[{name}] running TabPFN CV AUC...")
    t0 = time.time()
    X = cohort[feats].values
    auc_m, auc_s = auc_cv_tabpfn(X, y, n_splits=10, n_seeds=5)
    print(f"  AUC = {auc_m:.3f} +/- {auc_s:.3f}  ({time.time()-t0:.0f}s)")
    results.append({
        'subset': name, 'n_features': len(feats),
        'auc_mean': auc_m, 'auc_std': auc_s,
    })

R = pd.DataFrame(results)
R.to_csv(OUT / 'auc_by_subset.csv', index=False)
print("\n=== TabPFN classifier AUC by Wasserstein subset ===")
print(R.to_string(index=False))
print(f"\nSaved to {OUT}/auc_by_subset.csv")
