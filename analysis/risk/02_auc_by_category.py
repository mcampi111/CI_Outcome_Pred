#!/usr/bin/env python3
"""
Rerun Tab/Fig auc_by_category on the N=378 SSD-excluded cohort.

This replaces the previous N=431 analysis (which incorrectly included 53 SSD patients).
Output: auc_by_category_n378.csv + figure3_n378.pdf

Expected runtime: 5-10 minutes on Mac (TabPFN zero-shot is fast).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
DATA_PATH = "CI_UNIFIED_DATASET.xlsx"  # change to your local path
N_SEEDS = 5
N_FOLDS = 10
THRESHOLD = 40  # poor outcome: Mono1_post < 40

# The 6 robust decaying features (from Wasserstein analysis, Table 3)
DECAYING_FEATURES = [
    'Mono1_pre',
    'Mono2_pre',
    'V08_pre',
    'C12_pre',
    'Num_pre',
    'prPTA_6000_Co',
]

# ---------------------------------------------------------------------------
# 1. LOAD AND FILTER COHORT TO N=378
# ---------------------------------------------------------------------------
print("=" * 60)
print("Loading dataset")
print("=" * 60)
df = pd.read_excel(DATA_PATH)
print(f"Total rows: {len(df)}")

# Adults
adults = df[df['Age_at_OP'] >= 18].copy()
print(f"Adults (age >= 18): {len(adults)}")

# SSD exclusion: contralateral 4-freq PTA <= 30 dB HL
pta_co_cols = ['prPTA_500_Co', 'prPTA_1000_Co', 'prPTA_2000_Co', 'prPTA_4000_Co']
adults['pta_co_avg'] = adults[pta_co_cols].mean(axis=1, skipna=False)
adults['is_SSD'] = (adults['pta_co_avg'] <= 30) & adults['pta_co_avg'].notna()

# Final cohort: adults, SSD-excluded, with Mono1_post
cohort = adults[(~adults['is_SSD']) & adults['Mono1_post'].notna()].copy()
print(f"Cohort N=378 (SSD-excluded, with Mono1_post): {len(cohort)}")

# Outcome
cohort['poor'] = (cohort['Mono1_post'] < THRESHOLD).astype(int)
n_poor = cohort['poor'].sum()
print(f"Poor outcomes (Mono1 < {THRESHOLD}%): {n_poor} ({n_poor/len(cohort)*100:.1f}%)")

# ---------------------------------------------------------------------------
# 2. DEFINE THE 79-FEATURE POOL
# ---------------------------------------------------------------------------

# Audiometric PTA (16): 8 freqs × 2 ears
pta_freqs = [125, 250, 500, 1000, 2000, 4000, 6000, 8000]
pta_features = [f'prPTA_{f}_{side}' for side in ['CI', 'Co'] for f in pta_freqs]

# Free-field aided thresholds (14): 7 freqs × 2 ears
ff_freqs = [250, 500, 1000, 2000, 4000, 6000, 8000]
ff_features = [f'prFF_{f}_{side}' for side in ['CI', 'Co'] for f in ff_freqs]

# Bone conduction (10): 5 freqs × 2 ears
bc_freqs = [250, 500, 1000, 2000, 4000]
bc_features = [f'prBC_{f}_{side}' for side in ['CI', 'Co'] for f in bc_freqs]

# Pre-op speech tests (6)
speech_features = ['Mono1_pre', 'Mono2_pre', 'V08_pre', 'C12_pre', 'FM_pre', 'Num_pre']

# Demographics + clinical (everything else - we'll use what's actually in the df)
candidate_other = [
    'Age_at_OP', 'Geschlecht',
    'EVA_DEAF_ONSET', 'EVA_HL_DURATION', 'EVA_ETIOLOGY', 'EVA_HL_PROGREDIENT', 'EVA_DIAGN',
    'COM_ARTICULATION', 'COM_MULTILANG_HOME', 'COM_LIP_READING', 'COM_PHONE_USE', 'COM_MAIN_LANGUAGE',
    'SES_EDUCATION', 'SES_MARITAL', 'SES_WORK', 'SES_INSURANCE',
    'Sur_surg_access', 'Sur_insertion_techn', 'Sur_electrode_insertion',
    'Sur_electrode_fixation', 'Sur_gusher', 'Sur_duration', 'Sur_num_of_not_ins_elec',
    'CT_Cochlear_malf', 'CT_Ossification', 'CT_Otosclerosis', 'CT_Mastoid_middle_ear_abnorm',
    'CT_Brain_path', 'CT_Pathway_VII_abnorm', 'CT_Trauma', 'CT_Carotid_a_abnorm', 'CT_Sigmoid_sin_abnorm',
    'Seite',
]

# Keep only features that exist in the dataframe
all_features_attempt = pta_features + ff_features + bc_features + speech_features + candidate_other
all_features = [f for f in all_features_attempt if f in cohort.columns]
print(f"\nFeatures available: {len(all_features)}")

# Trim to 79 if we have more (or just use what we have)
# Print a sanity check on the 6 decaying features
missing_decaying = [f for f in DECAYING_FEATURES if f not in cohort.columns]
if missing_decaying:
    print(f"WARNING: missing decaying features: {missing_decaying}")
    # Try alternative names
    if 'prPTA_6000_Co' in missing_decaying:
        # Look for similar
        candidates = [c for c in cohort.columns if '6000' in c and 'Co' in c]
        print(f"  Alternatives for prPTA_6000_Co: {candidates}")

# Encode categorical features as integer codes (median imputation later)
cat_cols = [c for c in all_features if cohort[c].dtype == 'object']
print(f"Categorical columns to encode: {len(cat_cols)}")
for c in cat_cols:
    cohort[c] = pd.Categorical(cohort[c]).codes
    cohort.loc[cohort[c] == -1, c] = np.nan  # -1 means NaN in pd.Categorical.codes

NON_DECAYING_FEATURES = [f for f in all_features if f not in DECAYING_FEATURES]
print(f"\nDecaying features: {len(DECAYING_FEATURES)}")
print(f"Non-decaying features: {len(NON_DECAYING_FEATURES)}")
print(f"All combined: {len(all_features)}")

# ---------------------------------------------------------------------------
# 3. TABPFN SETUP (zero-shot)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Setting up TabPFN")
print("=" * 60)
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
    print("TabPFN imported OK")
except ImportError:
    HAS_TABPFN = False
    print("WARNING: TabPFN not installed. Will skip TabPFN columns.")
    print("Install with: pip install tabpfn")

# ---------------------------------------------------------------------------
# 4. RUN AUC ANALYSIS
# ---------------------------------------------------------------------------

def run_classifier(X, y, clf_factory, n_seeds=N_SEEDS, n_folds=N_FOLDS):
    """Run a classifier with stratified k-fold CV across multiple seeds."""
    aucs = []
    for seed in range(n_seeds):
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Median imputation
            imputer = SimpleImputer(strategy='median')
            X_train_imp = imputer.fit_transform(X_train)
            X_test_imp = imputer.transform(X_test)
            
            # Standardize for LR
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_imp)
            X_test_scaled = scaler.transform(X_test_imp)
            
            try:
                clf = clf_factory()
                if isinstance(clf, LogisticRegression):
                    clf.fit(X_train_scaled, y_train)
                    y_pred = clf.predict_proba(X_test_scaled)[:, 1]
                else:
                    clf.fit(X_train_imp, y_train)
                    y_pred = clf.predict_proba(X_test_imp)[:, 1]
                auc = roc_auc_score(y_test, y_pred)
                fold_aucs.append(auc)
            except Exception as e:
                print(f"    Fold error: {e}")
                continue
        if fold_aucs:
            aucs.append(np.mean(fold_aucs))
    return np.mean(aucs), np.std(aucs)


def run_subset(name, features, y):
    """Run all 3 classifiers on a feature subset."""
    print(f"\n--- {name} ({len(features)} features) ---")
    X = cohort[features].values
    y_arr = y.values
    
    results = {}
    
    # Logistic Regression
    print("  Running LR...", end=" ", flush=True)
    auc_mean, auc_std = run_classifier(
        X, y_arr,
        lambda: LogisticRegression(class_weight='balanced', max_iter=2000, random_state=0)
    )
    results['LR'] = (auc_mean, auc_std)
    print(f"AUC = {auc_mean:.3f} ± {auc_std:.3f}")
    
    # Random Forest
    print("  Running RF...", end=" ", flush=True)
    auc_mean, auc_std = run_classifier(
        X, y_arr,
        lambda: RandomForestClassifier(n_estimators=500, class_weight='balanced',
                                        random_state=0, n_jobs=-1)
    )
    results['RF'] = (auc_mean, auc_std)
    print(f"AUC = {auc_mean:.3f} ± {auc_std:.3f}")
    
    # TabPFN
    if HAS_TABPFN:
        print("  Running TabPFN (zero-shot)...", end=" ", flush=True)
        try:
            auc_mean, auc_std = run_classifier(
                X, y_arr,
                lambda: TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)
            )
            results['TabPFN'] = (auc_mean, auc_std)
            print(f"AUC = {auc_mean:.3f} ± {auc_std:.3f}")
        except Exception as e:
            print(f"FAILED: {e}")
            results['TabPFN'] = (np.nan, np.nan)
    else:
        results['TabPFN'] = (np.nan, np.nan)
    
    return results

print("\n" + "=" * 60)
print("Running AUC analysis on N=378")
print("=" * 60)

y = cohort['poor']
results = {
    'Decaying':     run_subset('Decaying',     DECAYING_FEATURES,     y),
    'Non-decaying': run_subset('Non-decaying', NON_DECAYING_FEATURES, y),
    'All':          run_subset('All',          all_features,          y),
}

# ---------------------------------------------------------------------------
# 5. SAVE RESULTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Saving results")
print("=" * 60)

# CSV output
rows = []
for subset_name, subset_results in results.items():
    n_vars = {'Decaying': 6, 'Non-decaying': len(NON_DECAYING_FEATURES), 'All': len(all_features)}[subset_name]
    for clf_name, (auc_mean, auc_std) in subset_results.items():
        rows.append({
            'Subset': subset_name,
            'N_features': n_vars,
            'Classifier': clf_name,
            'AUC_mean': auc_mean,
            'AUC_std': auc_std,
        })
out_df = pd.DataFrame(rows)
out_df.to_csv('auc_by_category_n378.csv', index=False)
print(f"\nSaved auc_by_category_n378.csv")
print(out_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 6. GENERATE FIG 3 (figure3_n378.pdf)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Generating Fig 3")
print("=" * 60)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Helvetica', 'DejaVu Sans', 'Arial'],
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, ax = plt.subplots(figsize=(10, 6))

subsets = ['Decaying', 'Non-decaying', 'All']
classifiers = ['LR', 'RF', 'TabPFN']
colors = {'LR': '#5b8db8', 'RF': '#d29240', 'TabPFN': '#7a4f7e'}

x = np.arange(len(subsets))
width = 0.27

for i, clf in enumerate(classifiers):
    means = [results[s][clf][0] for s in subsets]
    stds  = [results[s][clf][1] for s in subsets]
    offset = (i - 1) * width
    bars = ax.bar(x + offset, means, width, yerr=stds,
                   color=colors[clf], label=clf, capsize=4,
                   edgecolor='white', linewidth=1)
    # Add AUC values on bars
    for bar, mean in zip(bars, means):
        if not np.isnan(mean):
            ax.text(bar.get_x() + bar.get_width()/2., mean + 0.01,
                    f'{mean:.3f}', ha='center', va='bottom', fontsize=9, color='#333')

# Chance line
ax.axhline(0.5, color='#999', linestyle='--', linewidth=1, alpha=0.7, zorder=0)
ax.text(2.4, 0.51, 'chance', fontsize=9, color='#888', style='italic')

ax.set_xticks(x)
ax.set_xticklabels([f'{s}\n(N features={[6, len(NON_DECAYING_FEATURES), len(all_features)][i]})' for i, s in enumerate(subsets)])
ax.set_ylabel('Cross-validated AUC', fontsize=11)
ax.set_ylim(0.4, 0.85)
ax.legend(loc='upper left', frameon=False, fontsize=10)
ax.set_title(f'Pre-operative classification of poor outcome (Mono1 < 40%)\nN={len(cohort)} adults SSD-excluded, {n_poor} poor outcomes ({n_poor/len(cohort)*100:.1f}%)',
              fontsize=11, pad=15)

plt.tight_layout()
plt.savefig('figure3_n378.pdf', bbox_inches='tight', dpi=200)
plt.savefig('figure3_n378.png', bbox_inches='tight', dpi=200)
print("Saved figure3_n378.pdf and figure3_n378.png")

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)
print(f"\nSend back:")
print(f"  - auc_by_category_n378.csv")
print(f"  - figure3_n378.pdf (or .png)")
