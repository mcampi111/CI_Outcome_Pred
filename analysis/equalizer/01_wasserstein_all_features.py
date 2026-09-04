"""
Optimal-transport analysis for all 49 features listed in Table S9.

For each feature, patients are grouped by its value (continuous variables are
split into quartiles). W1 is the size-weighted mean Wasserstein distance
between the groups' Mono1 distributions, computed separately before and after
implantation. The change is tested by permutation of the group labels.

Method matched against Marta's SI_table_COMPLETE_OT.csv: the size-weighted
mean and a minimum group size of 10 reproduce her values for articulation
(18.3 vs 18.0 -> 19.7 vs 19.4), onset (3.6 vs 4.1 -> 17.1 vs 17.4),
Co PTA4 (26.9 -> 5.9 vs 5.7) and sign language (9.3 vs 9.7 -> 20.2 vs 20.5).
"""
import re, sys
import numpy as np, pandas as pd
from scipy.stats import wasserstein_distance

N_PERM = 5000
MIN_GROUP = 10
RNG = np.random.default_rng(0)

df = pd.read_excel('data/CI_UNIFIED_DATASET.xlsx')
for c in [c for c in df.columns if re.match(r'pr(PTA|FF)_', c)]:
    df[c] = df[c].where(df[c] < 150, 130)
df['PTA_Co'] = df[['prPTA_500_Co','prPTA_1000_Co','prPTA_2000_Co','prPTA_4000_Co']].mean(axis=1)
coh = df[(df.Age_at_OP >= 18) & ~(df['PTA_Co'] <= 30)].copy()
coh = coh.loc[:, ~coh.columns.duplicated()].dropna(subset=['Mono1_pre', 'Mono1_post'])

# the 49 features of Table S9
FEATURES = [
 'COM_ARTICULATION','SES_PROFESSION_LEARNED','prPTA_1000_CI','COM_Gebaerden','prPTA_2000_CI',
 'SES_EDUCATION_LEVEL','COM_MULTILANG_HOME','prPTA_6000_Co','EVA_DEAF_ONSET','prFF_2000_CI',
 'EVA_DEAF_ONSET_L','Mono2_pre','COM_FATHER_HACI_USER','prPTA_500_CI','prPTA_1000_Co','C12_pre',
 'prPTA_4000_CI','prFF_6000_Co','prFF_500_Co','prFF_1000_CI','FM_pre','prPTA_500_Co',
 'EVA_HL_PROGREDIENT','Mono1_pre','prPTA_2000_Co','prFF_4000_CI','prPTA_8000_Co','prFF_6000_CI',
 'prPTA_4000_Co','Num_pre','prFF_250_Co','prFF_500_CI','prPTA_6000_CI','prFF_4000_Co',
 'prFF_250_CI','prPTA_250_CI','V08_pre','prPTA_8000_CI','COM_PHONE_USE','prPTA_125_Co',
 'prFF_1000_Co','prPTA_125_CI','COM_LIP_READING','Geschlecht','prPTA_250_Co','prFF_8000_Co',
 'prFF_8000_CI','prFF_2000_Co','Age_at_OP',
]

def weighted_w1(groups, col_idx):
    """size-weighted mean pairwise W1 across groups"""
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i+1:]]
    if not pairs:
        return np.nan
    w = [len(a[col_idx]) * len(b[col_idx]) for a, b in pairs]
    d = [wasserstein_distance(a[col_idx], b[col_idx]) for a, b in pairs]
    return float(np.average(d, weights=w))

def analyse(var):
    sub = coh[[var, 'Mono1_pre', 'Mono1_post']]
    sub = sub.loc[:, ~sub.columns.duplicated()]
    d = sub.dropna()
    col = d[var]
    if isinstance(col, pd.DataFrame): col = col.iloc[:, 0]
    if col.dtype.kind in 'ifc' and col.nunique() > 6:
        lab = pd.qcut(col, 4, labels=False, duplicates='drop')
    else:
        lab = col.astype(str)
    d = d.assign(g=lab)
    counts = d.g.value_counts()
    keep = counts[counts >= MIN_GROUP].index
    d = d[d.g.isin(keep)]
    if d.g.nunique() < 2:
        return None
    pre = np.asarray(d['Mono1_pre'], dtype=float).ravel()
    post = np.asarray(d['Mono1_post'], dtype=float).ravel()
    g = d['g'].values

    def groups_from(labels):
        return [(pre[labels == k], post[labels == k]) for k in np.unique(labels)]

    obs = groups_from(g)
    w_pre, w_post = weighted_w1(obs, 0), weighted_w1(obs, 1)
    obs_change = w_pre - w_post                       # positive = convergence

    null = np.empty(N_PERM)
    for i in range(N_PERM):
        perm = RNG.permutation(g)
        gp = groups_from(perm)
        null[i] = weighted_w1(gp, 0) - weighted_w1(gp, 1)
    p_conv = (np.sum(null >= obs_change) + 1) / (N_PERM + 1)
    p_div  = (np.sum(null <= obs_change) + 1) / (N_PERM + 1)
    rel = 100 * obs_change / w_pre if w_pre else np.nan
    return dict(Variable=var, n=len(d), n_groups=d.g.nunique(),
                W1_pre=round(w_pre, 1), W1_post=round(w_post, 1),
                Rel_change=round(rel, 1), p_converge=round(p_conv, 4),
                p_diverge=round(p_div, 4))

rows = []
for i, v in enumerate(FEATURES, 1):
    if v not in coh.columns:
        print(f'  [{i:2d}/49] {v:26s} NOT IN DATASET'); continue
    r = analyse(v)
    if r is None:
        print(f'  [{i:2d}/49] {v:26s} too few groups'); continue
    rows.append(r)
    print(f'  [{i:2d}/49] {v:26s} {r["W1_pre"]:5.1f} -> {r["W1_post"]:5.1f}  '
          f'p_conv={r["p_converge"]:.4f}  p_div={r["p_diverge"]:.4f}', flush=True)

out = pd.DataFrame(rows)
out.to_csv('results/W1_all49.csv', index=False)
print(f'\nwritten: {len(out)} of 49 features')
