"""
Figure 1: score distributions by group, before and after implantation, for two
pre-operative features that behave in opposite ways.

W1 is computed exactly as in Table S9: size-weighted mean pairwise Wasserstein
distance, paired sample (N = 264), quartiles for continuous features, groups
smaller than 10 patients dropped.
"""
import re
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, gaussian_kde

df = pd.read_excel('data/CI_UNIFIED_DATASET.xlsx')
for c in [c for c in df.columns if re.match(r'pr(PTA|FF)_', c)]:
    df[c] = df[c].where(df[c] < 150, 130)
df['PTA_Co'] = df[['prPTA_500_Co','prPTA_1000_Co','prPTA_2000_Co','prPTA_4000_Co']].mean(axis=1)
df['PTA_CI'] = df[['prPTA_500_CI','prPTA_1000_CI','prPTA_2000_CI','prPTA_4000_CI']].mean(axis=1)
coh = df[(df.Age_at_OP >= 18) & ~(df['PTA_Co'] <= 30)].copy()
coh = coh.loc[:, ~coh.columns.duplicated()]
paired = coh.dropna(subset=['Mono1_pre', 'Mono1_post'])

MIN_GROUP = 10
COL = ['#1b3a6b', '#3e8fa8', '#c9821b', '#a33b1e']
ONSET = {'post-ling': 'Post-lingual onset', 'pre-ling': 'Pre-lingual onset'}

def grouped(var, quartiles=False):
    """returns [(label, pre_values, post_values), ...] using the Table S9 rules"""
    d = paired[[var, 'Mono1_pre', 'Mono1_post']]
    d = d.loc[:, ~d.columns.duplicated()].dropna()
    if quartiles:
        d = d.assign(g=pd.qcut(d[var], 4, labels=False, duplicates='drop'))
        edges = d.groupby('g')[var].agg(['min', 'max'])
        names = {k: f"Q{k+1}  {edges.loc[k,'min']:.0f}-{edges.loc[k,'max']:.0f} dB" for k in edges.index}
    else:
        d = d.assign(g=d[var].astype(str))
        names = {k: ONSET.get(k, k) for k in d.g.unique()}
    cnt = d.g.value_counts()
    keep = cnt[cnt >= MIN_GROUP].index
    d = d[d.g.isin(keep)]
    return [(names[k], v['Mono1_pre'].values, v['Mono1_post'].values)
            for k, v in d.groupby('g')]

def weighted_w1(groups, idx):
    pairs = [(a, b) for i, a in enumerate(groups) for b in groups[i+1:]]
    w = [len(a[idx]) * len(b[idx]) for a, b in pairs]
    dist = [wasserstein_distance(a[idx], b[idx]) for a, b in pairs]
    return float(np.average(dist, weights=w))

xs = np.linspace(0, 100, 400)
fig, ax = plt.subplots(2, 2, figsize=(11.0, 7.6), sharex=True)
rows = [(0, 'EVA_DEAF_ONSET', False, 'Onset of deafness'),
        (1, 'PTA_CI',         True,  'Pre-operative PTA,\nimplanted ear')]

for r, var, q, ylab in rows:
    gs = grouped(var, q)
    for k, (col, when) in enumerate([(1, 'Before implantation'), (2, 'After implantation')]):
        a = ax[r, k]
        peak = 0
        for j, g in enumerate(gs):
            v = g[col]
            kde = gaussian_kde(v, bw_method=.4)
            y = kde(xs); peak = max(peak, y.max())
            a.fill_between(xs, y, alpha=.24, color=COL[j])
            a.plot(xs, y, color=COL[j], lw=2.1, label=g[0])
            a.axvline(np.mean(v), color=COL[j], ls='--', lw=1.2)
        a.set_ylim(0, peak * 1.50)
        if r == 0:
            a.set_title(when, fontsize=14)
        a.text(.03, .93, f'$W_1$ = {weighted_w1(gs, col):.1f}', transform=a.transAxes,
               fontsize=14, va='top', fontweight='bold')
        a.tick_params(axis='x', labelsize=11)
        a.set_yticks([]); a.spines[['top','right','left']].set_visible(False)
        if k == 0:
            a.set_ylabel(ylab, fontsize=13)
            a.legend(fontsize=11, frameon=False, loc='upper right')
        a.set_xlabel('Monosyllable word recognition (%)', fontsize=13) if r == 1 else None

fig.tight_layout(rect=[0.02, 0, 1, 1])
fig.savefig('figures/fig_distributions_v2.pdf')
fig.savefig('figures/fig_distributions_v2.eps')
fig.savefig('/tmp/F1.png', dpi=74)
for r, var, q, lab in rows:
    gs = grouped(var, q)
    print(f'  {lab.splitlines()[0]:24s} W1 {weighted_w1(gs,1):5.1f} -> {weighted_w1(gs,2):5.1f}  '
          f'({len(gs)} groups, n = {sum(len(g[1]) for g in gs)})')
