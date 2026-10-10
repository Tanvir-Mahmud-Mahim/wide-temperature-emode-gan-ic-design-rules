"""Normalized varactor C-V-T figure for the article (Fig. 1 of the main text).
Capacitance is shown relative to its maximum for each device, so no absolute device quantity of
the PDK (covered by a non-disclosure agreement) is shown.  Data: digitized Spectre curves
(ganpll.data.load_varactor_cv)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from ganpll import data
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
                     'font.size': 8.5, 'axes.labelsize': 9, 'legend.fontsize': 7.5, 'axes.linewidth': 0.7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': 0.3})
COL = {218: '#440154', 273: '#3b528b', 300: '#21918c', 398: '#5ec962', 423: '#c8a11b'}
fig, axs = plt.subplots(2, 1, figsize=(3.3, 3.9), sharex=True)
for ax, dev, lab in ((axs[0], 'small', '(a) 10-$\\mu$m fingers, 50 fingers'), (axs[1], 'large', '(b) 300-$\\mu$m fingers, 350 fingers')):
    cv = data.load_varactor_cv(dev)
    cmax = max(np.max(c['C_pF']) for c in cv)
    for c in sorted(cv, key=lambda c: c['T']):
        ax.plot(c['V'], c['C_pF'] / cmax, color=COL.get(int(c['T']), '0.5'), lw=1.3, label='%d K' % c['T'])
    ax.set_ylabel('$C/C_{\\max}$'); ax.set_ylim(0, 1.05)
    ax.text(0.02, 0.95, lab, transform=ax.transAxes, va='top', fontsize=8)
axs[0].legend(loc='lower right', ncol=2, fontsize=7)
axs[1].set_xlabel('Gate voltage $V_{AB}$ (V)')
fig.tight_layout(pad=0.3)
fig.savefig('figures/fig_cv_norm.pdf'); fig.savefig('figures/fig_cv_norm.svg')
print('fig_cv_norm done')
