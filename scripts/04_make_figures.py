"""Publication figures (IEEE single/double column, serif, PDF+SVG).

FN1 fig_pinn.svg      : PINN C-V-T fit + held-out test + ablation bars
FN2 fig_valid.svg     : tuning-surface fit, 14-corner blind validation, 300K transient
FN3 fig_opt.svg       : convergence vs random search + gradient cost + design shifts
FN4 fig_corners.svg   : f_end vs T before/after, lock transients, tuning windows
(iteration-1 study; model level)
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import json as _json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import jax.numpy as jnp
from ganpll import data, pinn, pll, adjoint

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9,
    'legend.fontsize': 7.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
    'grid.linewidth': 0.4, 'grid.alpha': 0.3, 'savefig.dpi': 300,
    'axes.grid': True, 'legend.framealpha': 1.0, 'legend.edgecolor': '0.85',
    'legend.borderpad': 0.35, 'legend.handletextpad': 0.5,
    'legend.labelspacing': 0.3, 'legend.borderaxespad': 0.4,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.titlepad': 6,
})
W1 = 3.5   # single column (in)
W2 = 7.16  # double column

TCOLORS = {218: '#440154', 273: '#3b528b', 300: '#21918c', 398: '#5ec962', 423: '#c8a11b'}
tc = lambda T: TCOLORS.get(int(T), '#888888')

pst = pinn.load_state('results/pinn_state.json')
st_nn = pinn.load_state('results/plainnn_state.json')
st_an = pinn.load_state('results/analytic_state.json')
vp = pll.load_vcro('results/vcro_params.json')
circ = pll.default_circuit(); circ['icp'] = json.load(open('results/circuit_cal.json'))['icp']
pres = json.load(open('results/pinn_results.json'))
cres = json.load(open('results/calibration_results.json'))
ares = json.load(open('results/adjoint_results.json'))
raw_opt = jnp.asarray(np.load('results/raw_opt_final.npy'))
design_opt = adjoint.raw_to_design(raw_opt)

# =====================================================================
# FN1: PINN varactor surrogate
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.35))
cv_l = data.load_varactor_cv('large')
ax = axs[0]
for c in cv_l:
    T = c['T']; col = tc(T)
    held = abs(T - 300) < 1
    ax.plot(c['V'], c['C_pF'], 'o', ms=2.6, mfc='none', mew=0.8, color=col)
    Vg = jnp.linspace(float(c['V'].min()), float(c['V'].max()), 120)
    Cm = np.asarray(pinn.cap_fF(pst, Vg, jnp.full_like(Vg, T), jnp.ones_like(Vg))) / 1e3
    ax.plot(np.asarray(Vg), Cm, '-', color=col, lw=1.3,
            label=f'{T:.0f} K' + (' (held out)' if held else ''))
ax.set_xlabel('Control voltage $V_{AB}$ (V)')
ax.set_ylabel('Capacitance (pF)')
ax.legend(ncol=1, handlelength=1.4, loc='lower right', fontsize=7)
ax.set_title('(a) PINN surrogate vs. Spectre C–V–T', fontsize=8.5)

ax = axs[1]
labels = ['PINN\n(this work)', 'Plain NN', 'Analytic\nprior only']
rmse = [pres['pinn']['rmse_pct_300K'], pres['plain_nn']['rmse_pct_300K'], pres['analytic']['rmse_pct_300K']]
cols = ['#0f62fe', '#a2a9b0', '#a2a9b0']
b = ax.bar(labels, rmse, width=0.55, color=cols, edgecolor='none')
for r, v in zip(b, rmse):
    ax.text(r.get_x() + r.get_width()/2, v + 0.12, f'{v:.2f}%', ha='center', fontsize=7.5)
ax.set_ylabel('Held-out 300 K RMSE (%)')
ax.set_ylim(0, max(rmse)*1.25)
ax.set_title('(b) Interpolation accuracy', fontsize=8.5)

ax = axs[2]
mono = [pres['pinn']['mono_viol_pct_inrange'], pres['plain_nn']['mono_viol_pct_inrange'], pres['analytic']['mono_viol_pct_inrange']]
b = ax.bar(labels, mono, width=0.55, color=cols, edgecolor='none')
for r, v in zip(b, mono):
    ax.text(r.get_x() + r.get_width()/2, v + 0.4, f'{v:.2f}%', ha='center', fontsize=7.5)
ax.set_ylabel('Monotonicity violation (%)')
ax.set_ylim(0, max(mono)*1.3 + 1)
ax.set_title('(c) Physics consistency (218–423 K)', fontsize=8.5)
fig.tight_layout(pad=0.4)
fig.savefig('figures/FN1.svg'); fig.savefig('figures/FN1.pdf'); plt.close(fig)
print('FN1 done')

# =====================================================================
# FN2: model calibration + blind validation
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.35))
ax = axs[0]
for t in data.load_vcro_tuning():
    T = t['T']; col = tc(T)
    ax.plot(t['V'], t['f_MHz'], 'o', ms=2.2, mfc='none', mew=0.7, color=col)
    Vg = jnp.linspace(float(t['V'].min()), float(t['V'].max()), 150)
    fm = np.asarray(pll.f_vcro(vp, pst, Vg, T)) / 1e6
    ax.plot(np.asarray(Vg), fm, '-', color=col, lw=1.2)
    ax.annotate(f'{T:.0f} K', xy=(float(Vg[-1]), float(fm[-1])),
                xytext=(3, 0), textcoords='offset points', fontsize=7,
                color=col, va='center')
ax.set_xlim(None, 7.1)
ax.set_xlabel('Control voltage $V_c$ (V)')
ax.set_ylabel('VCRO frequency (MHz)')
ax.set_title('(a) Tuning surface: model vs. Spectre', fontsize=8.5)

ax = axs[1]
rows = cres['corner_validation']['rows']
seen = [r for r in rows if r['in_fit_set']]
uns = [r for r in rows if not r['in_fit_set']]
ax.plot([0, 110], [0, 110], '-', color='0.75', lw=0.8, zorder=1)
ax.plot([r['f_meas'] for r in seen], [r['f_pred'] for r in seen], 'o', ms=4,
        color='#0f62fe', mew=0, label='calibration corners')
ax.plot([r['f_meas'] for r in uns], [r['f_pred'] for r in uns], 's', ms=4.5,
        color='#d02670', mew=0, label='blind corners')
ax.set_xlabel('Spectre plateau frequency (MHz)')
ax.set_ylabel('Model prediction (MHz)')
ax.legend(loc='upper left', fontsize=7.2)
ax.set_title('(b) Corner validation (218–423 K)', fontsize=8.5)

ax = axs[2]
z = np.load('results/transient_300K.npz')
ax.plot(z['meas_t']*1e6, z['meas_f']/1e6, 'o', ms=2.0, mfc='none', mew=0.7,
        color='0.45', label='Spectre (transistor level)')
ax.plot(z['t']*1e6, z['f']/1e6, '-', color='#0f62fe', lw=1.3, label='differentiable model')
ax.axhline(50, color='0.75', lw=0.8, ls='--')
ax.set_xlabel('Time (µs)')
ax.set_ylabel('Output frequency (MHz)')
ax.set_xlim(0, 60)
ax.legend(loc='lower right', fontsize=7.2)
ax.set_title('(c) 300 K lock transient', fontsize=8.5)
fig.tight_layout(pad=0.4)
fig.savefig('figures/FN2.svg'); fig.savefig('figures/FN2.pdf'); plt.close(fig)
print('FN2 done')

# =====================================================================
# FN3: adjoint optimization
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.85), gridspec_kw=dict(width_ratios=[1.0, 0.85, 1.25]))
ax = axs[0]
h2 = json.load(open('results/opt_hist2.json'))   # fixed-bias stage
htt = json.load(open('results/opt_hist_tt.json'))  # thermally tracked stage
it1 = [h[0] for h in h2]; j1 = [h[1] for h in h2]
off = it1[-1] + 1
it2 = [h[0] + off for h in htt]; j2 = [h[1] for h in htt]
rs = ares['random_search']['hist']
ax.semilogy(it1, j1, '-', color='#0f62fe', lw=1.4, label='adjoint, fixed bias')
ax.semilogy(it2, j2, '-', color='#d02670', lw=1.4, label='adjoint, tracked bias')
ax.axvline(off, color='0.6', lw=0.7, ls=':')
ax.annotate('thermally tracked\nbias enabled', xy=(off, j2[0]), fontsize=6.8,
            color='0.35', ha='left', va='bottom', xytext=(off + 25, j1[0]*2.6),
            arrowprops=dict(arrowstyle='-', color='0.5', lw=0.6))
ax.semilogy([r[0] for r in rs], [r[1] for r in rs], '-', color='#a2a9b0', lw=1.2,
            label='random search')
ax.set_xlabel('Iteration / evaluation')
ax.set_ylabel('Objective $J$')
ax.legend(fontsize=7.2)
ax.set_title('(a) Convergence', fontsize=8.5)

ax = axs[1]
gc = ares['gradient_cost']
names = ['forward\ntransients', 'adjoint\ngradient', 'finite-diff.\ngradient']
vals = [gc['t_forward_s'], gc['t_adjoint_s'], gc['t_fd_s']]
cols = ['#a2a9b0', '#0f62fe', '#a2a9b0']
b = ax.bar(names, vals, width=0.55, color=cols)
for r, v in zip(b, vals):
    ax.text(r.get_x()+r.get_width()/2, v*1.03, f'{v:.2f} s', ha='center', fontsize=7.5)
ax.set_ylabel('Wall-clock time (s)')
ax.set_ylim(0, max(vals)*1.2)
ax.set_title(f"(b) Gradient cost, {gc['n_params']} parameters", fontsize=8.5)
ax.text(0.04, 0.96, f"gradient agreement\n(cosine) {gc['cosine_agreement']:.3f}", transform=ax.transAxes, ha='left', va='top', fontsize=6.8, color='0.3')

ax = axs[2]
dnames = [n for n, _, _ in adjoint.DESIGN_SPACE if n in ares['optimization']['design']]
_dsrc = ares['optimization']['design']          # iteration-1 co-designed parameters
dvals = [_dsrc.get(n, 1.0) for n in dnames]
pretty = {'s_icp': 'Pump current', 's_rs1': '$R_{s1}$', 's_cs1': '$C_{s1}$',
          's_rs23': '$R_{s23}$', 's_c2': '$C_2$ width', 's_c3': '$C_3$ width',
          'drive': 'Drive', 's_cfix': 'Delay cap.', 's_var': 'VCO varactor',
          'dvb': '$\\Delta V_b$ (V)', 'dvb_t': '$|s_T|$ (mV/K)'}
y = np.arange(len(dnames))
mult = [abs(v)*1e3 if n == 'dvb_t' else max(v, 1e-2) for n, v in zip(dnames, dvals)]
cols = ['#0f62fe' if n not in ('dvb', 'dvb_t') else '#d02670' for n in dnames]
ax.barh(y, mult, height=0.6, color=cols)
ax.axvline(1.0, color='0.4', lw=0.8, ls='--')
ax.set_yticks(y, [pretty[n] for n in dnames])
ax.set_xscale('log')
ax.set_xlabel('Optimized value (scale factor)', fontsize=8)
ax.set_xlim(min(mult)/2.5, max(mult)*5.0)
for yi, v in zip(y, mult):
    ax.annotate(f'{v:.2f}', xy=(v, yi), xytext=(3, 0), textcoords='offset points', va='center', ha='left', fontsize=6.8)
ax.set_title('(c) Optimized design', fontsize=8.5)
ax.invert_yaxis()
fig.tight_layout(pad=0.5, w_pad=1.2)
fig.savefig('figures/FN3.svg', bbox_inches='tight', pad_inches=0.04); fig.savefig('figures/FN3.pdf', bbox_inches='tight', pad_inches=0.04); plt.close(fig)
print('FN3 done')

# =====================================================================
# FN4: before/after across temperature
# =====================================================================
fig, axs = plt.subplots(1, 3, figsize=(W2, 2.4))
ax = axs[0]
vb = ares['verify_baseline']; vo = ares['verify_optimized']
Tb = [r['T_K'] for r in vb]
ax.axhspan(49.5, 50.5, color='#defbe6', zorder=0)
ax.plot(Tb, [r['f_end_MHz'] for r in vb], 'o-', color='#a2a9b0', lw=1.2, ms=3.5,
        label='baseline design')
ax.plot(Tb, [r['f_end_MHz'] for r in vo], 's-', color='#0f62fe', lw=1.3, ms=3.5,
        label='adjoint-optimized')
ax.axhline(50, color='#198038', lw=0.7, ls='--')
ax.text(224, 47.6, '±1% lock band', fontsize=7, color='#198038')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('Settled output frequency (MHz)')
ax.legend(loc='lower left', fontsize=7.2)
ax.set_title('(a) Frequency lock across temperature', fontsize=8.5)

ax = axs[1]
z = np.load('results/opt_traces.npz')
t_us = z['t']*1e6
for T, ls in ((398, '-'), (423, '--')):
    ax.plot(t_us, z[f'base_{T}_f']/1e6, ls, color='#a2a9b0', lw=1.1)
    ax.plot(t_us, z[f'opt_{T}_f']/1e6, ls, color='#0f62fe', lw=1.2)
ax.axhline(50, color='#198038', lw=0.7, ls='--')
ax.set_xlim(0, 40)
ax.set_xlabel('Time (µs)')
ax.set_ylabel('Output frequency (MHz)')
leg = [Line2D([], [], color='#a2a9b0', lw=1.2, label='baseline'),
       Line2D([], [], color='#0f62fe', lw=1.2, label='optimized'),
       Line2D([], [], color='0.3', lw=1.0, ls='-', label='398 K'),
       Line2D([], [], color='0.3', lw=1.0, ls='--', label='423 K')]
ax.legend(handles=leg, fontsize=7, ncol=2, loc='upper right')
ax.set_title('(b) Hot-corner lock transients', fontsize=8.5)

ax = axs[2]
To = [r['T_K'] for r in vo]
ax.fill_between(Tb, [r['f_min_MHz'] for r in vb], [r['f_max_MHz'] for r in vb],
                color='#a2a9b0', alpha=0.30, label='baseline tuning range')
ax.fill_between(To, [r['f_min_MHz'] for r in vo], [r['f_max_MHz'] for r in vo],
                color='#0f62fe', alpha=0.22, label='optimized tuning range')
ax.axhline(50, color='#198038', lw=1.0, ls='--', label='50 MHz target')
ax.set_xlabel('Temperature (K)')
ax.set_ylabel('VCRO frequency (MHz)')
ax.set_yscale('log')
ax.legend(fontsize=7, loc='upper right')
ax.set_title('(c) Tuning-range coverage', fontsize=8.5)
fig.tight_layout(pad=0.4)
fig.savefig('figures/FN4.svg'); fig.savefig('figures/FN4.pdf'); plt.close(fig)
print('FN4 done')
print('all figures saved to figures/')

# Figures of the iteration-1 study; the figures of the revised article are made by
# scripts/30_paper_figures.py.
