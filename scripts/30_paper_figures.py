#!/usr/bin/env python3
"""30_paper_figures.py -- figures of the revised manuscript (Elsevier single column).

fig_pinn     : PINN surrogate, ablations, leave-one-temperature-out, ensemble band
fig_tlrecon  : open-flow oscillator vs Spectre; iteration-1 sizing with VDD vs 7 V chargers
fig_bands    : surrogate lockable bands vs open-flow tuning curves of the co-designed VCRO
fig_closed   : closed-loop transistor-level transients and lock summary at ten temperatures
fig_robust   : closed-loop robustness corners
Reads results/*.json and results/tl/*.json; writes figures/*.pdf.
Colours: reference categorical order (blue, orange, aqua, ...) and an ordinal blue ramp
for temperature; every series also differs by marker or line style.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['STIXGeneral', 'DejaVu Serif'], 'mathtext.fontset': 'stix',
    'font.size': 8.5, 'axes.labelsize': 9, 'axes.titlesize': 9, 'legend.fontsize': 7.5,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'axes.linewidth': 0.7, 'lines.linewidth': 1.4,
    'grid.linewidth': 0.4, 'grid.alpha': 0.3, 'savefig.dpi': 300, 'axes.grid': True,
    'legend.framealpha': 1.0, 'legend.edgecolor': '0.85', 'axes.spines.top': False, 'axes.spines.right': False,
})
W = 6.5                                   # text width (in)
BLUE, ORANGE, AQUA, YELLOW, GRAY = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8d8b85'
RAMP = ['#86b6ef', '#6da7ec', '#5598e7', '#3987e5', '#2a78d6', '#256abf', '#1c5cab', '#184f95', '#104281', '#0d366b']
TV = [218, 233, 253, 273, 300, 323, 348, 373, 398, 423]
def tcol(T):
    return RAMP[int(np.argmin([abs(T - t) for t in TV]))]
MK = ['o', 's', '^', 'D', 'v', 'P', 'X', '<', '>', 'h']
def tmk(T):
    return MK[int(np.argmin([abs(T - t) for t in TV]))]
os.makedirs('figures', exist_ok=True)
which = sys.argv[1:] or ['pinn', 'tlrecon', 'bands', 'closed', 'robust']

if 'pinn' in which:
    import jax.numpy as jnp
    from ganpll import data, pinn
    pst = pinn.load_state('results/pinn_state.json')
    pres = json.load(open('results/pinn_results.json')); uq = json.load(open('results/pinn_uq.json'))
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.6))
    ax = axs[0, 0]
    # capacitance normalized to the device maximum: no absolute device quantity of the PDK is shown
    cvl = data.load_varactor_cv('large'); CN = max(float(np.max(c['C_pF'])) for c in cvl)
    for c in cvl:
        T = c['T']; held = abs(T - 300) < 1
        ax.plot(c['V'], c['C_pF'] / CN, tmk(T), ms=3, mfc='none', mew=0.8, color=tcol(T))
        Vg = jnp.linspace(float(c['V'].min()), float(c['V'].max()), 120)
        Cm = np.asarray(pinn.cap_fF(pst, Vg, jnp.full_like(Vg, T), jnp.ones_like(Vg))) / 1e3
        ax.plot(np.asarray(Vg), Cm / CN, '-', color=tcol(T), lw=1.2, label='%d K%s' % (T, ' (held out)' if held else ''))
    ax.set_xlabel('$V_{AB}$ (V)'); ax.set_ylabel('$C/C_{\\max}$'); ax.legend(loc='lower right', fontsize=7)
    ax.set_title('(a) Surrogate (lines) vs Spectre (markers)', fontsize=8.5)
    ax = axs[0, 1]
    lab = ['PINN', 'Unconstrained\nnetwork', 'Analytic\nprior only']
    v = [pres['pinn']['rmse_pct_300K'], pres['plain_nn']['rmse_pct_300K'], pres['analytic']['rmse_pct_300K']]
    b = ax.bar(lab, v, width=0.5, color=[BLUE, GRAY, GRAY])
    for r_, x in zip(b, v):
        ax.text(r_.get_x() + r_.get_width() / 2, x + 0.1, '%.2f%%' % x, ha='center', fontsize=7.5)
    ax.set_ylabel('Held-out 300 K RMSE (%)'); ax.set_ylim(0, max(v) * 1.25); ax.grid(axis='x', visible=False)
    ax.set_title('(b) Held-out error and ablations', fontsize=8.5)
    ax = axs[1, 0]
    Ts = ['218', '273', '300', '398', '423']
    v = [uq['loto'][t]['rmse_pct'] for t in Ts]
    cols = [ORANGE if uq['loto'][t]['extrapolation'] else BLUE for t in Ts]
    b = ax.bar([t + ' K' for t in Ts], v, width=0.5, color=cols)
    for r_, x in zip(b, v):
        ax.text(r_.get_x() + r_.get_width() / 2, x * 1.15, '%.1f%%' % x, ha='center', fontsize=7.5)
    ax.set_yscale('log'); ax.set_ylim(0.5, 60); ax.set_ylabel('RMSE of held-out temperature (%)')
    ax.bar([0], [0], color=BLUE, label='interpolation'); ax.bar([0], [0], color=ORANGE, label='extrapolation')
    ax.legend(loc='upper center', fontsize=7); ax.grid(axis='x', visible=False)
    ax.set_title('(c) Leave-one-temperature-out', fontsize=8.5)
    ax = axs[1, 1]
    z = np.load('results/pinn_uq_pred.npz')
    m = z['g'] > 0.5
    o = np.argsort(z['V'][m])
    V, C, mu, sd = z['V'][m][o], z['C'][m][o] / 1e3 / CN, z['mu'][m][o] / 1e3 / CN, z['sd'][m][o] / 1e3 / CN
    ax.fill_between(V, mu - 2 * sd, mu + 2 * sd, color=BLUE, alpha=0.25, lw=0, label='ensemble $\\pm2\\sigma$')
    ax.plot(V, mu, '-', color=BLUE, lw=1.2, label='ensemble mean')
    ax.plot(V, C, 'o', ms=3, mfc='none', mew=0.8, color='#0b0b0b', label='Spectre, 300 K (held out)')
    ax.set_xlabel('$V_{AB}$ (V)'); ax.set_ylabel('$C/C_{\\max}$'); ax.legend(loc='lower right', fontsize=7)
    ax.set_title('(d) Deep-ensemble uncertainty', fontsize=8.5)
    fig.tight_layout(pad=0.5); fig.savefig('figures/fig_pinn.pdf'); plt.close(fig); print('fig_pinn')

if 'tlrecon' in which:
    from ganpll import data
    base = json.load(open('results/tl/vcro_baseline_sweep.json')); it1 = json.load(open('results/tl/iter1_tl_check.json'))
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.6))
    ax = axs[0]
    for t in data.load_vcro_tuning():
        T = int(round(t['T']))
        if T not in (218, 273, 300):
            continue
        ax.plot(t['V'], t['f_MHz'], tmk(T), ms=2.6, mfc='none', mew=0.7, color=tcol(T))
        r = sorted([x for x in base if x['T'] == T], key=lambda x: x['vc'])
        ax.plot([x['vc'] for x in r], [x['f'] / 1e6 for x in r], '-', color=tcol(T), label='%d K' % T)
    ax.set_xlim(3.0, 5.6); ax.set_xlabel('$V_c$ (V)'); ax.set_ylabel('Frequency (MHz)')
    ax.legend(loc='lower right', fontsize=7, title='open flow (lines)\nSpectre (markers)', title_fontsize=6.5)
    ax.set_title('(a) Manual-design oscillator', fontsize=8.5)
    ax = axs[1]
    for tag, mf, ls in (('diode', 'none', '--'), ('rail', None, '-')):
        for T in (218, 300, 398, 423):
            r = sorted([x for x in it1 if x['tag'] == tag and x['T'] == T], key=lambda x: x['vc'])
            ax.plot([x['vc'] for x in r], [x['f'] / 1e6 for x in r], ls, marker=tmk(T), ms=4,
                    mfc=mf if mf else tcol(T), color=tcol(T), label=('%d K' % T) if tag == 'rail' else None)
    ax.set_ylim(-5, 230)
    ax.set_xlabel('$V_c$ (V)'); ax.set_ylabel('Frequency (MHz)'); ax.legend(loc='upper left', fontsize=7, ncol=4,
              title='filled, solid: 7 V rail;  open, dashed: $V_{DD}$ charger', title_fontsize=6.5, columnspacing=0.8)
    ax.set_title('(b) Iteration-1 sizing: charger supply', fontsize=8.5)
    fig.tight_layout(pad=0.5); fig.savefig('figures/fig_tlrecon.pdf'); plt.close(fig); print('fig_tlrecon')

if 'cert' in which:
    r = json.load(open('results/rl_results.json'))
    fig, axs = plt.subplots(1, 2, figsize=(W, 2.5))
    ax = axs[0]
    fs = r['feasibility_screen']; temps = r['temps_train']
    v = [100 * x for x in fs['per_corner_pass_rate']] + [100 * fs['n_all6'] / fs['n_screened']]
    lab = ['%d' % t for t in temps] + ['all six']
    b = ax.bar(lab, v, width=0.6, color=[BLUE] * len(temps) + [ORANGE])
    for r_, x in zip(b, v):
        ax.text(r_.get_x() + r_.get_width() / 2, x + 1, '%.0f%%' % x if x > 0 else '0', ha='center', fontsize=7.5)
    ax.set_ylabel('Designs that can lock (%)'); ax.set_xlabel('Temperature (K)'); ax.grid(axis='x', visible=False)
    ax.set_title('(a) Monte Carlo screen, 100,000 designs', fontsize=8.5)
    ax = axs[1]
    for key, col, ls, lab in (('gcn_sac_per', BLUE, '-', 'graph encoder + prioritized replay'),
                              ('mlp_sac_uniform', GRAY, '--', 'MLP encoder + uniform replay')):
        er = np.asarray(r[key]['episode_returns'], dtype=float)      # (env step, return, length)
        k = 10; ys = np.convolve(er[:, 1], np.ones(k) / k, mode='valid')
        ax.plot(er[k - 1:, 0], ys, ls, color=col, label=lab)
    ax.set_xlabel('Environment step'); ax.set_ylabel('Episode return (10-episode mean)'); ax.legend(fontsize=7)
    ax.set_title('(b) RL exploration of the fixed-bias space', fontsize=8.5)
    fig.tight_layout(pad=0.5); fig.savefig('figures/fig_cert.pdf'); plt.close(fig); print('fig_cert')

def load_tl(tag, mode='int'):
    out = {}
    for f in glob.glob('results/tl/%s_*_%s*.json' % (tag, mode)):
        r = json.load(open(f)); out[int(round(r['T_K']))] = r
    return out

def settle_vc(r, tol=0.02):
    t = np.array(r['wave']['t']); vc = np.array(r['wave']['vc']); T = 1e-7; nb = int(t[-1] / T)
    tb, vb = [], []
    for k in range(nb):
        m = (t >= k * T) & (t < (k + 1) * T)
        if m.any(): tb.append((k + 0.5) * T); vb.append(vc[m].mean())
    tb, vb = np.array(tb), np.array(vb); vf = vb[tb > 0.8 * t[-1]].mean()
    bad = np.where(np.abs(vb - vf) > tol)[0]
    return tb[bad[-1]] * 1e6 if len(bad) else 0.0

if 'closed' in which:
    I = load_tl('f2', 'int'); F = load_tl('f2', 'frac')
    sys.path.insert(0, 'tl'); from frac_residual import phase_metrics
    for r in F.values():                     # fractional: subtract the known quantization phase
        pm = phase_metrics(r); r['phase_mean_deg'] = pm['res_mean_deg']; r['phase_pp_deg'] = pm['res_pp_deg']
        r['lock'] = pm['res_pp_deg'] < 90.0
    for r in I.values():
        if 'phase_series' in r:
            r['lock'] = phase_metrics(r)['res_pp_deg'] < 30.0
    fig, axs = plt.subplots(2, 2, figsize=(W, 4.6))
    ax = axs[0, 0]
    C3 = {218: '#104281', 300: AQUA, 423: ORANGE}
    for T in (218, 300, 423):
        if T in I:
            w = I[T]['wave']; ax.plot(np.array(w['t']) * 1e6, w['vc'], color=C3[T], lw=1.0, label='%d K' % T)
    ax.set_xlabel('Time ($\\mu$s)'); ax.set_ylabel('$V_c$ (V)'); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('(a) Acquisition from power-up, $N=5$', fontsize=8.5)
    ax = axs[0, 1]
    for T in (218, 300, 423):
        if T in I and 'phase_series' in I[T]:
            ps = I[T]['phase_series']; ph = (np.array(ps['deg']) + 180.0) % 360.0 - 180.0
            ax.plot(np.array(ps['t']) * 1e6, ph, '.', ms=2.5, color=C3[T], label='%d K' % T)
    ax.set_ylim(-185, 185); ax.set_yticks([-180, -90, 0, 90, 180]); ax.set_xlabel('Time ($\\mu$s)'); ax.set_ylabel('Phase error (deg)'); ax.legend(fontsize=7, loc='lower right', markerscale=4)
    ax.set_title('(b) Phase error at each reference edge', fontsize=8.5)
    ax = axs[1, 0]
    for D, mk, lab, col in ((I, 'o', 'integer, 50 MHz', BLUE), (F, 's', 'fractional, 52 MHz', ORANGE)):
        Ts = sorted(D)
        if not Ts: continue
        y = [D[T]['phase_mean_deg'] for T in Ts]; e = [D[T]['phase_pp_deg'] / 2 for T in Ts]
        ax.errorbar(Ts, y, yerr=e, fmt=mk, ms=4, color=col, capsize=2, lw=1, label=lab,
                    mfc=col if mk == 'o' else 'white')
        for T in Ts:
            if not D[T]['lock']:
                ax.plot(T, D[T]['phase_mean_deg'], 'x', color='#e34948', ms=8)
    ax.axhline(0, color='0.6', lw=0.6); ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Static phase error (deg)')
    ax.legend(fontsize=7); ax.set_title('(c) Static phase error and residual wander', fontsize=8.5)
    ax = axs[1, 1]
    Ts = sorted(I)
    ax.plot(Ts, [I[T]['p_vdd_mW'] for T in Ts], 'o-', color=BLUE, ms=4, label='power, integer mode')
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Power from 6 V (mW)')
    ax2 = ax.twinx() if False else None
    for T in Ts:
        nb = sum(1 for k in range(4) if I[T]['bnd%d' % k] > 3)
        ax.annotate('%d' % nb, (T, I[T]['p_vdd_mW']), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=7)
    ax.legend(fontsize=7, loc='upper right'); ax.set_title('(d) Power; numbers: band segments switched in', fontsize=8.5)
    fig.tight_layout(pad=0.5); fig.savefig('figures/fig_closed.pdf'); plt.close(fig); print('fig_closed')

if 'bands' in which:
    from ganpll import pinn, tl_anchor as ta, pll2
    import jax.numpy as jnp
    pst = pinn.load_state('results/pinn_state.json'); tp, ex = ta.load('results/tl_vcro.json')
    cd = json.load(open('results/tl_codesign.json')); d = cd['design']
    bc = [json.loads(l) for l in open('results/tl/band_char.jsonl')]
    fig, ax = plt.subplots(1, 1, figsize=(W, 2.8))
    ax.axhspan(50, 52.5, color=YELLOW, alpha=0.25, lw=0, label='target 50--52.5 MHz')
    for r in cd['integer']:
        T = r['T_K']; b = r['band']
        ax.plot([T, T], [r['win_lo_MHz'], r['win_hi_MHz']], '-', color=BLUE, lw=6, alpha=0.35, solid_capstyle='butt')
        pts = sorted([x for x in bc if x['T'] == T and x['band'] == b and x['f'] > 1e6], key=lambda x: x['vc'])
        if pts:
            ax.plot([T] * len(pts), [x['f'] / 1e6 for x in pts], '_', color='#0b0b0b', ms=8, mew=1.2)
    ax.plot([], [], '-', color=BLUE, lw=6, alpha=0.35, label='surrogate band (onset + 0.15 V to 5.4 V)')
    ax.plot([], [], '_', color='#0b0b0b', ms=8, mew=1.2, label='open flow, $V_c$ = 3.2--5.4 V in 0.2 V steps')
    ax.set_xlabel('Temperature (K)'); ax.set_ylabel('Oscillator frequency (MHz)'); ax.legend(fontsize=7, loc='upper right')
    fig.tight_layout(pad=0.5); fig.savefig('figures/fig_bands.pdf'); plt.close(fig); print('fig_bands')

if 'precharge' in which:
    pr = [json.loads(l) for l in open('results/tl/precharge_sweep.jsonl')]
    fig, ax = plt.subplots(1, 1, figsize=(W * 0.62, 2.6))
    for T in sorted({r['T'] for r in pr}):
        rr = sorted([r for r in pr if r['T'] == T], key=lambda r: r['vpre'])
        v = [r['vpre'] for r in rr]; f = [r['f'] / 1e6 for r in rr]
        ax.plot(v, f, '-', marker=tmk(T), ms=4, color=tcol(T), label='%d K' % T)
        for r in rr:
            if r['f'] <= 1e6:
                ax.plot(r['vpre'], 0, 'x', color='#e34948', ms=7, mew=1.5)
    ax.plot([], [], 'x', color='#e34948', ms=7, mew=1.5, label='no oscillation')
    ax.axvline(6.0, color='0.6', lw=0.7, ls='--'); ax.text(6.02, 3, '$V_{pre}=V_{DD}$', fontsize=7, color='0.35')
    ax.set_xlabel('Bootstrap precharge voltage $V_{pre}$ (V)'); ax.set_ylabel('Ring frequency (MHz)')
    ax.legend(fontsize=6.5, ncol=2, loc='lower right'); ax.set_ylim(-4, 90)
    fig.tight_layout(pad=0.4); fig.savefig('figures/fig_precharge.pdf'); plt.close(fig); print('fig_precharge')

if 'cvcheck' in which:
    # normalized shapes only: the absolute capacitance and the threshold shift of the PDK device
    # are not shown (non-disclosure agreement)
    e = json.load(open('results/eda_verification.json'))
    V = np.asarray(e['vbias']); Ts = sorted(int(float(t)) for t in e['curves'])
    fig, ax = plt.subplots(1, 1, figsize=(W * 0.5, 2.5))
    nz = lambda c: (np.asarray(c) - np.min(c)) / (np.max(c) - np.min(c))
    for T in Ts:
        c = e['curves'][str(T)] if str(T) in e['curves'] else e['curves']['%d' % T]
        ax.plot(V, nz(c['c_ngspice_fF']), tmk(T), ms=3, mfc='none', mew=0.8, color=tcol(T))
        ax.plot(V, nz(c['c_pinn_fF']), '-', color=tcol(T), lw=1.1, label='%d K' % T)
    ax.set_xlabel('$V_{AB}$ (V)'); ax.set_ylabel('Normalized capacitance'); ax.legend(fontsize=7, loc='lower right')
    ax.set_title('Open flow (markers) vs PINN (lines)', fontsize=8.5)
    fig.tight_layout(pad=0.5); fig.savefig('figures/fig_cvcheck.pdf'); plt.close(fig); print('fig_cvcheck')
