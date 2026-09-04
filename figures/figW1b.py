"""
All 46 pre-operative features in the W1_pre vs W1_post plane.

Below the diagonal: the feature's groups converge after implantation.
Above: they move apart. Colour = permutation test (5,000 label permutations).
Filled = significant at p < 0.05, open = not significant.

Source: W1_all49.csv (computed by w1_all.py on the paired sample, N = 264).
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from adjustText import adjust_text

d = pd.read_csv('results/W1_all49.csv')

PRETTY = {
 'COM_ARTICULATION':'Articulation','SES_PROFESSION_LEARNED':'Profession','COM_Gebaerden':'Sign language',
 'SES_EDUCATION_LEVEL':'Education','COM_MULTILANG_HOME':'Multilingual home','EVA_DEAF_ONSET':'Onset of deafness',
 'EVA_DEAF_ONSET_L':'Onset, left ear','COM_FATHER_HACI_USER':'Father HA/CI','EVA_HL_PROGREDIENT':'Progressive HL',
 'COM_PHONE_USE':'Phone use','COM_LIP_READING':'Lip reading','Geschlecht':'Sex','Age_at_OP':'Age at implantation',
 'Mono1_pre':'WRS pre','Mono2_pre':'WRS list 2 pre','C12_pre':'Consonants pre','V08_pre':'Vowels pre',
 'FM_pre':'Voice discrim. pre','Num_pre':'Numbers pre',
}
def pretty(v):
    if v in PRETTY: return PRETTY[v]
    m = pd.Series(v).str.extract(r'pr(PTA|FF)_(\d+)_(CI|Co)').iloc[0]
    if m.notna().all():
        kind = 'PTA' if m[0] == 'PTA' else 'FF'
        return f"{kind} {int(m[1])/1000:g}k {m[2]}" if int(m[1]) >= 1000 else f"{kind} {m[1]} {m[2]}"
    return v
d['label'] = d.Variable.map(pretty)
d['kind'] = ['converge' if r.p_converge < .05 else 'diverge' if r.p_diverge < .05 else 'ns'
             for r in d.itertuples()]

COL = {'converge':'#1b3a6b', 'diverge':'#c9821b', 'ns':'#9aa0a6'}
LAB = {'converge':'Groups converge  ($p < .05$)',
       'diverge':'Groups move apart  ($p < .05$)',
       'ns':'No significant change'}

fig, ax = plt.subplots(figsize=(9.2, 8.4))
hi = max(d.W1_pre.max(), d.W1_post.max()) * 1.18
ax.plot([0, hi], [0, hi], ls='--', color='0.6', lw=1.2, zorder=1)
ax.text(hi*.80, hi*.835, 'no change', color='0.5', fontsize=10.5, rotation=45)
ax.fill_between([0, hi], [0, hi], 0, color='#1b3a6b', alpha=.035, zorder=0)
ax.fill_between([0, hi], [0, hi], hi, color='#c9821b', alpha=.045, zorder=0)

for k in ['ns', 'converge', 'diverge']:
    g = d[d.kind == k]
    ax.scatter(g.W1_pre, g.W1_post, s=78, color=COL[k],
               alpha=.9 if k != 'ns' else .65, edgecolor='white', lw=1.0,
               label=f'{LAB[k]}   (n = {len(g)})', zorder=3)

# label only what carries information: everything significant, plus any
# feature that starts or ends far from the origin. The dense low-low cloud
# is annotated as a group instead.
show = d[(d.kind != 'ns') | (d.W1_pre > 13) | (d.W1_post > 13)]
hide = d.drop(show.index)
texts = [ax.text(r.W1_pre, r.W1_post, r.label, fontsize=9.2,
                 color=COL[r.kind] if r.kind != 'ns' else '0.4')
         for r in show.itertuples()]
adjust_text(texts, ax=ax, expand=(1.22, 1.45), force_text=(.5, .7),
            arrowprops=dict(arrowstyle='-', color='0.55', lw=.9, ls=(0,(2,2)), alpha=.6))
if len(hide):
    ax.annotate(f'{len(hide)} further audiometric and\naided-field thresholds',
                xy=(hide.W1_pre.mean(), hide.W1_post.mean()),
                xytext=(hi*.55, hi*.30), fontsize=9.5, color='0.45',
                ha='left', va='center',
                arrowprops=dict(arrowstyle='-', color='0.55', lw=.9, ls=(0,(2,2)), alpha=.6))

ax.set_xlabel('Distance between groups before implantation ($W_1$, percentage points)', fontsize=12.5)
ax.set_ylabel('Distance between groups after implantation ($W_1$, percentage points)', fontsize=12.5)
ax.set_xlim(-1.5, hi); ax.set_ylim(-1.5, hi)
ax.grid(True, which='major', ls=':', lw=.7, color='0.85', zorder=0)
ax.set_axisbelow(True)
ax.tick_params(labelsize=11)
ax.legend(fontsize=11, frameon=False, loc='upper left')
ax.spines[['top','right']].set_visible(False)
ax.set_aspect('equal')
fig.tight_layout()
fig.savefig('figures/fig_w1_scatter.pdf')
fig.savefig('figures/fig_w1_scatter.eps')
fig.savefig('/tmp/W1b.png', dpi=76)
print(d.kind.value_counts().to_string())
