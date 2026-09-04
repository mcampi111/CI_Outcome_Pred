"""
Figure 2: cross-validated AUC for poor outcome classification, by feature
subset and algorithm.

Values and fold-level SDs taken from Supplementary Table S10. Counts corrected
to 35 / 6 / 5 / 46 to match the main text.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

SUBSETS = [
    ('Audiometric, aided-field\nand pre-operative speech', 35),
    ('Developmental\nand communicative',                    6),
    ('Demographic\nand clinical',                           5),
    ('All combined',                                      46),
]
ALGOS = ['Logistic regression', 'Random Forest', 'TabPFN']
AUC = np.array([[0.529, 0.587, 0.585],
                [0.641, 0.577, 0.654],
                [0.650, 0.594, 0.666],
                [0.667, 0.613, 0.657]])
SD  = np.array([[0.08, 0.06, 0.09],
                [0.13, 0.09, 0.13],
                [0.15, 0.12, 0.16],
                [0.13, 0.08, 0.14]])
COL = ['#1b3a6b', '#3d8b84', '#c9992b']

x = np.arange(len(SUBSETS))
w = 0.26
fig, ax = plt.subplots(figsize=(9.4, 5.4))
ax.axhline(0.5, ls='--', color='0.55', lw=1.2, zorder=1)
ax.text(len(SUBSETS) - 0.42, 0.505, 'chance', fontsize=10.5, color='0.45', va='bottom')

for j, algo in enumerate(ALGOS):
    ax.bar(x + (j - 1) * w, AUC[:, j], w, yerr=SD[:, j], capsize=4,
           color=COL[j], label=algo, zorder=3,
           error_kw=dict(lw=1.2, ecolor='0.35'))
    for i in range(len(SUBSETS)):
        ax.text(x[i] + (j - 1) * w, AUC[i, j] + SD[i, j] + .012, f'{AUC[i,j]:.2f}',
                ha='center', fontsize=9.2, color='0.25')

ax.set_xticks(x)
ax.set_xticklabels([f'{n}\n({k} features)' for n, k in SUBSETS], fontsize=11)
ax.set_ylabel('AUC (10-fold cross-validated)', fontsize=12.5)
ax.set_ylim(0.40, 0.86)
ax.tick_params(axis='y', labelsize=11)
ax.set_axisbelow(True)
ax.legend(fontsize=10.5, frameon=False, loc='upper left', ncol=3,
          bbox_to_anchor=(0, 1.02))
ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
fig.savefig('figures/fig_auc_by_category_v22.pdf')
fig.savefig('figures/fig_auc_by_category_v22.eps')
fig.savefig('/tmp/F2.png', dpi=76)
print('figure 2 regenerated with counts 35 / 6 / 5 / 46')
