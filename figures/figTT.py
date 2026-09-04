"""
Trajectory types: four-panel summary, in the style of the earlier
patient-profiles figure.

  A  mean trajectory of each type across the follow-up intervals
  B  how many patients fall into each type
  C  pre-operative score by type
  D  final score (12-24 months, or beyond) by type

Types are assigned by the change in Mono1 from before to after implantation
(threshold 10 percentage points), as in 06_phenotyping.py.

Cohort here is N = 160 (adults, SSD excluded, with a pre-operative, an early
and a late score). The manuscript reports N = 136; the proportions match
(50/19/21/10) but the counts do not.
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

df = pd.read_excel('data/CI_UNIFIED_DATASET.xlsx')
CO = ['prPTA_500_Co','prPTA_1000_Co','prPTA_2000_Co','prPTA_4000_Co']
for c in CO: df[c] = df[c].where(df[c] < 150, 130)
df['PTA_Co'] = df[CO].mean(axis=1)
coh = df[(df.Age_at_OP >= 18) & ~(df['PTA_Co'] <= 30)].copy()

PRE, E, M, L, F = ('SQL_Freiburger_pre','SQL_Freiburger_0_6mo','SQL_Freiburger_6_12mo',
                   'SQL_Freiburger_12_24mo','SQL_Freiburger_24plus')
T   = [PRE, E, M, L, F]
LAB = ['Pre-op', '0-6 mo', '6-12 mo', '12-24 mo', '>24 mo']
G   = 10

d = coh.copy()
d['Final'] = d[L].fillna(d[F])
d = d[d[PRE].notna() & d[E].notna() & d['Final'].notna()]

def kind(r):
    eg, tg = r[E] - r[PRE], r['Final'] - r[PRE]
    if tg > G and eg > G: return 'Early Responder'
    if tg > G:            return 'Late Responder'
    if abs(tg) <= G:      return 'Non-Responder'
    return 'Decliner'
d['type'] = d.apply(kind, axis=1)

ORDER = ['Early Responder', 'Late Responder', 'Non-Responder', 'Decliner']
SHORT = ['Early\nResponder', 'Late\nResponder', 'Non-\nResponder', 'Decliner']
COL   = ['#1b3a6b', '#2a9d8f', '#c9821b', '#a33b1e']
N     = len(d)


fig, a = plt.subplots(figsize=(8.8, 5.2))
x = np.arange(len(T))
for c, name in zip(COL, ORDER):
    g = d[d.type == name][T].astype(float)
    lo, hi = g.quantile(.25), g.quantile(.75)
    a.fill_between(x, lo.values, hi.values, color=c, alpha=.13, lw=0)
    a.plot(x, g.mean().values, color=c, lw=2.9, marker='o', ms=7.5,
           label=f'{name}  (n = {len(g)}, {len(g)/N*100:.0f}%)')
a.axvline(1, color='0.65', ls=':', lw=1.3)
a.set_xticks(x); a.set_xticklabels(LAB, fontsize=12)
a.set_ylabel('Monosyllable word recognition (%)', fontsize=13)
a.set_xlabel('Time relative to implantation', fontsize=13)
a.set_ylim(0, 100)
a.tick_params(axis='y', labelsize=11.5)
a.legend(fontsize=11.5, frameon=False, loc='lower right')
a.spines[['top','right']].set_visible(False)
fig.tight_layout()
fig.savefig('figures/fig_trajectory_types.pdf')
fig.savefig('figures/fig_trajectory_types.eps')
fig.savefig('/tmp/TT.png', dpi=72)
print('N =', N)
for t in ORDER:
    g = d[d.type == t]
    print(f'  {t:17s} n={len(g):3d} ({len(g)/N*100:.0f}%)  pre {g[PRE].mean():.0f}%  final {g["Final"].mean():.0f}%')
