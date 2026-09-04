"""
TabPFN variable importance × timepoint × configuration.

Mirrors the analysis already run with RF/GBM/Lasso/permutation, adding TabPFN
permutation importance as the gold-standard ML benchmark.

Run on Mac (mps) or DGX (cuda):
    python run_tabpfn_varimp_over_time.py CI_UNIFIED_DATASET.xlsx

Output: tabpfn_varimp_over_time/{tp}_{config}_importance.csv  (one per cell)
        + tabpfn_cv_r2_summary.csv
"""

import sys
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

# TabPFN local
try:
    from tabpfn import TabPFNRegressor
    print("✓ TabPFN imported")
except ImportError:
    print("✗ pip install tabpfn")
    sys.exit(1)

DEVICE = 'mps' if sys.platform == 'darwin' else 'cuda'
print(f"  Device: {DEVICE}")

XLSX = sys.argv[1] if len(sys.argv) > 1 else 'CI_UNIFIED_DATASET.xlsx'
OUT = Path('tabpfn_varimp_over_time')
OUT.mkdir(exist_ok=True)

# ---------- 1. Load + cohort ----------
df = pd.read_excel(XLSX, sheet_name='Sheet1')
df['Age_at_OP'] = pd.to_numeric(df['Age_at_OP'], errors='coerce')
adults = df[df['Age_at_OP'] >= 18].copy()
adults['months_since_OP'] = (adults['Mono1_poDat'] - adults['Datum_ErstesCI']).dt.days / 30.44
valid = adults[adults['Mono1_post'].notna() & adults['months_since_OP'].notna() & (adults['months_since_OP'] > 0)].copy()

def bin_tp(m):
    if m <= 6: return '0-6'
    if m <= 12: return '6-12'
    if m <= 24: return '12-24'
    return None
valid['tp'] = valid['months_since_OP'].apply(bin_tp)
valid = valid[valid['tp'].notna()].copy()

# ---------- 2. Build feature set (same as RF/GBM run) ----------
ci_freqs = [c for c in valid.columns if c.startswith('prPTA_') and c.endswith('_CI')]
co_freqs = [c for c in valid.columns if c.startswith('prPTA_') and c.endswith('_Co')]
aided = sorted([c for c in valid.columns if c.startswith('prFF_')])
speech_pre = ['Mono1_pre','Mono2_pre','V08_pre','C12_pre','FM_pre','Num_pre']
demo = ['Age_at_OP']
cat_cols_raw = ['Geschlecht','EVA_DEAF_ONSET','COM_ARTICULATION','EVA_HL_PROGREDIENT']
for c in cat_cols_raw:
    if c in valid.columns:
        valid[c+'_enc'] = LabelEncoder().fit_transform(valid[c].astype(str).fillna('NA'))
cat_cols = [c+'_enc' for c in cat_cols_raw if c+'_enc' in valid.columns]

preop_features = demo + ci_freqs + co_freqs + aided + speech_pre + cat_cols
preop_features = [f for f in preop_features if f in valid.columns]

# Earlier post-op tests for "+earlier" config
earlier_post = ['FM_post','Num_post','V08_post','C12_post']
for c in earlier_post:
    if c in valid.columns:
        valid[c] = pd.to_numeric(valid[c], errors='coerce')

config_features = {
    'preop': preop_features,
    'preop+earlier_post': preop_features + [c for c in earlier_post if c in valid.columns],
}

print(f"\nN preop features: {len(preop_features)}")
print(f"Cohort: {len(valid)} (0-6: {sum(valid.tp=='0-6')}, 6-12: {sum(valid.tp=='6-12')}, 12-24: {sum(valid.tp=='12-24')})")

# ---------- 3. Loop over timepoints × configs ----------
def cv_r2(model_factory, X, y, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in kf.split(X):
        m = model_factory()
        m.fit(X[tr], y[tr])
        scores.append(r2_score(y[te], m.predict(X[te])))
    return np.mean(scores), np.std(scores)

def perm_importance_tabpfn(model, X, y, n_repeats=10, random_state=42):
    """Lightweight permutation importance for TabPFN (slow → fewer repeats)."""
    rng = np.random.default_rng(random_state)
    base = r2_score(y, model.predict(X))
    imp = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            drops.append(base - r2_score(y, model.predict(Xp)))
        imp[j] = np.mean(drops)
    return imp

summary_rows = []

for tp in ['0-6', '6-12', '12-24']:
    sub = valid[valid['tp'] == tp].copy()
    y = sub['Mono1_post'].values

    for cfg_name, feats in config_features.items():
        feats_avail = [f for f in feats if f in sub.columns]
        X_df = sub[feats_avail].apply(pd.to_numeric, errors='coerce')
        X_df = X_df.dropna(axis=1, how='all')
        feats_avail = X_df.columns.tolist()
        X = SimpleImputer(strategy='median').fit_transform(X_df)

        if len(sub) < 30:
            print(f"\n[{tp} | {cfg_name}] SKIP (N<30)")
            continue

        print(f"\n[{tp} | {cfg_name}]  N={len(sub)}  features={len(feats_avail)}")

        t0 = time.time()
        # CV R²
        def make_model():
            return TabPFNRegressor(device=DEVICE, ignore_pretraining_limits=True)
        r2_mean, r2_std = cv_r2(make_model, X, y, n_splits=5)
        print(f"  TabPFN CV R² = {r2_mean:.3f} ± {r2_std:.3f}")

        # Refit on full data → permutation importance
        model = make_model()
        model.fit(X, y)
        imp = perm_importance_tabpfn(model, X, y, n_repeats=10, random_state=42)
        imp_df = pd.DataFrame({'feature': feats_avail, 'importance': imp}).sort_values('importance', ascending=False)
        out_path = OUT / f"tabpfn_imp_{tp}_{cfg_name.replace('+','_')}.csv"
        imp_df.to_csv(out_path, index=False)
        print(f"  Top 5: {imp_df.head(5)['feature'].tolist()}")
        print(f"  Saved: {out_path}  ({time.time()-t0:.0f}s)")

        summary_rows.append({
            'tp': tp, 'config': cfg_name, 'N': len(sub), 'n_feat': len(feats_avail),
            'cv_r2_mean': r2_mean, 'cv_r2_std': r2_std,
        })

pd.DataFrame(summary_rows).to_csv(OUT / 'tabpfn_cv_r2_summary.csv', index=False)
print(f"\n✓ All done. Output in: {OUT}/")
print(pd.DataFrame(summary_rows).to_string(index=False))
