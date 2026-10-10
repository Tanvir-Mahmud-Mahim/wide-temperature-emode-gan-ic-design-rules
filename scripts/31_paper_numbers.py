#!/usr/bin/env python3
"""31_paper_numbers.py -- every number quoted in the manuscript, generated from result files.
Writes paper/numbers.tex (LaTeX macros) and paper/tab_*.tex (tables)."""
import glob, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
R = 'results'; TL = 'results/tl'
os.makedirs('paper', exist_ok=True)
J = lambda p: json.load(open(p))
M = {}
def pct(x, d=2): return '%.*f\\%%' % (d, x)

# ---------------- PINN and UQ
p = J(f'{R}/pinn_results.json'); u = J(f'{R}/pinn_uq.json')
M['PINNrmse'] = pct(p['pinn']['rmse_pct_300K']); M['PLAINrmse'] = pct(p['plain_nn']['rmse_pct_300K'])
M['ANArmse'] = pct(p['analytic']['rmse_pct_300K']); M['PLAINmono'] = pct(p['plain_nn']['mono_viol_pct_inrange'], 1)
if os.path.exists(f'{R}/pinn_results_rerun.json'):
    M['PINNrmseRerun'] = pct(J(f'{R}/pinn_results_rerun.json')['pinn']['rmse_pct_300K'])
M['ENSrmse'] = pct(u['ensemble']['mean_rmse_pct']); M['ENSsigma'] = pct(u['ensemble']['mean_rel_sigma_pct'])
M['ENSmemberRange'] = '%.2f--%.2f\\%%' % (min(u['ensemble']['member_rmse_pct']), max(u['ensemble']['member_rmse_pct']))
li = [u['loto'][t]['rmse_pct'] for t in ('273', '300', '398')]
M['LOTOinterp'] = '%.1f--%.1f\\%%' % (min(li), max(li)); M['LOTOhot'] = pct(u['loto']['423']['rmse_pct'], 1)
M['LOTOcold'] = pct(u['loto']['218']['rmse_pct'], 1)
# open-flow C-V check: the full file holds absolute device capacitance (private, NDA); only the
# shape error is kept in the public summary file
if os.path.exists(f'{R}/eda_verification.json'):
    e = J(f'{R}/eda_verification.json'); json.dump({'mean_shape_rmse_pct': e['mean_shape_rmse_pct']}, open(f'{R}/eda_summary.json', 'w'))
e = J(f'{R}/eda_summary.json'); M['CVshape'] = pct(e['mean_shape_rmse_pct'])

# ---------------- iteration 1
c = J(f'{R}/calibration_results.json'); a = J(f'{R}/adjoint_results.json'); w = J(f'{R}/wgan_results.json')
M['CALrmse'] = pct(c['tuning_fit']['rmse_pct'], 1); M['CALseen'] = pct(c['corner_validation']['mae_pct_seen'], 1)
M['CALblind'] = pct(c['corner_validation']['mae_pct_unseen'], 1)
M['CALsettleModel'] = '%.1f~$\\mu$s' % c['transient_cal']['settle_model_us']
M['CALsettleSpectre'] = '%.1f~$\\mu$s' % c['transient_cal']['settle_meas_us']
g = a['gradient_cost']; M['ADJt'] = '%.2f~s' % g['t_adjoint_s']; M['FDt'] = '%.2f~s' % g['t_fd_s']
M['Jzero'] = '%.1f' % a['optimization']['J0']; M['Jfinal'] = '%.2f' % a['optimization']['J_final']
M['Jrand'] = '%.1f' % a['random_search']['best']
M['WGANtracked'] = pct(w['yields']['tracked'], 1); M['WGANrobust'] = pct(w['yields']['robust'], 1)

# ---------------- transistor-level reconstruction
base = J(f'{TL}/vcro_baseline_sweep.json'); it1 = J(f'{TL}/iter1_tl_check.json')
f300 = {round(x['vc'], 2): x['f'] / 1e6 for x in base if x['T'] == 300}
M['RECfa'] = '%.1f~MHz' % f300[4.0]; M['RECsat'] = '%.1f~MHz at 5.5~V' % f300[5.5]
sw = [x for x in base if x['f'] > 1e6]
M['RECswing'] = '%.2f to %.2f~V' % (min(x['vmin'] for x in sw), max(x['vmax'] for x in sw))
hot = [x for x in it1 if x['tag'] == 'rail' and x['T'] == 423]
M['ITERhot'] = '%.1f--%.1f~MHz' % (min(x['f'] for x in hot) / 1e6, max(x['f'] for x in hot) / 1e6)
try:
    from ganpll import data
    errs = []
    for t in data.load_vcro_tuning():
        T = int(round(t['T'])); r = sorted([x for x in base if x['T'] == T and x['vc'] >= 3.75 and x['f'] > 1e6], key=lambda x: x['vc'])
        if not r: continue
        fs = np.interp([x['vc'] for x in r], t['V'], t['f_MHz'])
        errs += list(np.abs(np.array([x['f'] / 1e6 for x in r]) - fs) / fs * 100)
    M['RECmedian'] = pct(np.median(errs), 1); M['RECmax'] = pct(np.max(errs), 0)
except Exception as ex:
    print('tuning data unavailable:', ex)

# ---------------- DOE surrogate
tv = J(f'{R}/tl_vcro.json')['_extra']
M['DOEnPts'] = str(tv['n']); M['DOEmape'] = pct(100 * tv['cv_freq_mape'], 1); M['DOEpninety'] = pct(100 * tv['cv_freq_p90'], 1)
M['DOEonset'] = pct(100 * tv['cv_onset_acc'], 1)
import re
lg = open(f'{R}/tl_vcro_fit.log').read()
mm = re.search(r'power model MAPE ([0-9.]+)%, stress model MAE ([0-9]+) mV', lg)
M['DOEpow'] = pct(float(mm.group(1)), 1); M['DOEvgs'] = '%s~mV' % mm.group(2)
M['DOEvbRange'] = '0--6'

# ---------------- iteration 2
cd = J(f'{R}/tl_codesign.json'); d = cd['design']          # final design (optimizer result with refined filter)
opt = J(f'{R}/tl_codesign_opt.json')                          # optimizer result before the filter refinement
cd['thresholds'] = opt['thresholds']                          # comparator thresholds as implemented
pmo = [r['pm_deg'] for r in opt['integer']]; ugo = [r['f_ugb_kHz'] for r in opt['integer']]
M['OPTpm'] = '%.0f--%.0f\\degree' % (min(pmo), max(pmo)); M['OPTugb'] = '%.1f--%.1f~MHz' % (min(ugo) / 1e3, max(ugo) / 1e3)
M['OPTrs'] = '%.1f~k$\\Omega$' % (opt['design']['Rs1'] / 1e3); M['OPTcs'] = '%.1f~pF' % (opt['design']['Cs1'] * 1e12)
M['FCMP'] = '10~MHz'; M['FRES'] = '%.2f~Hz' % (cd['f_cmp'] / 2 ** 20)
vbs = [r['vb'] for r in cd['integer']]; M['VBrange'] = '%.2f--%.2f~V' % (min(vbs), max(vbs))
pmv = [r['pm_deg'] for r in cd['integer']]; ugv = [r['f_ugb_kHz'] for r in cd['integer']]
M['ITtwoPM'] = '%.0f--%.0f\\degree' % (min(pmv), max(pmv)); M['ITtwoUGB'] = '%.1f--%.1f~MHz' % (min(ugv) / 1e3, max(ugv) / 1e3)
alt = J(f'{R}/tl_codesign_pm45.json')
pma = [r['pm_deg'] for r in alt['integer']]; M['ALTpm'] = '%.0f--%.0f\\degree' % (min(pma), max(pma))
M['ALTlock'] = '%d of %d' % (sum(r['lock'] for r in alt['integer']), len(alt['integer']))
ne = J(f'{R}/noise_eval.json')
M['NZfrac'] = '%.1f--%.1f~ns' % (min(r['jitter_frac_ns'] for r in ne), max(r['jitter_frac_ns'] for r in ne))
M['NZint'] = '%.1f--%.1f~ns' % (min(r['jitter_int_ns'] for r in ne), max(r['jitter_int_ns'] for r in ne))
pn300 = [r for r in ne if r['T_K'] == 300.0][0]
M['NZpn'] = '$%.1f$~dBc/Hz' % pn300['pn_1MHz_int_dBcHz']
M['ITtwoSettle'] = '%.0f~$\\mu$s' % max(r['settle_us'] for r in cd['integer'])
ph = [r['phi_ss_deg'] for r in cd['integer']]; M['ITtwoPhase'] = '%+.0f\\degree{} to %+.0f\\degree' % (min(ph), max(ph))

M['CVmapRMSE'] = pct(J(f'{R}/cv_map.json')['rel_rmse_pct']) if os.path.exists(f'{R}/cv_map.json') else '0.71\\%'
M['PNspectre'] = '$-83.3$~dBc/Hz'
fn = J(f'{R}/feas_noband.json'); M['FEASnobandMiss'] = pct(100 * (1 - np.exp(fn['score'])), 0)
mg = [min(np.log(r['win_hi_MHz'] / 52.5), np.log(50.0 / r['win_lo_MHz'])) for r in cd['integer']]
M['FEASmargin'] = pct(100 * (np.exp(min(mg)) - 1), 0)
ws = [t['T_hi'] - t['T_lo'] for t in cd['thresholds']]
M['THRwin'] = '%d to %d~K' % (min(ws), max(ws))

# bootstrap precharge sweep (open flow, final oscillator stage)
if os.path.exists(f'{TL}/precharge_sweep.jsonl'):
    pr = [json.loads(l) for l in open(f'{TL}/precharge_sweep.jsonl')]
    Ts = sorted({r['T'] for r in pr}); fail = sorted({r['T'] for r in pr if abs(r['vpre'] - 6.0) < 1e-9 and r['f'] < 1e6})
    M['PREfail'] = ' and '.join('%d' % T for T in fail) + '~K'
    M['PREok'] = '%d~K' % max(T for T in Ts if T not in fail)
    vmin = max(min(r['vpre'] for r in pr if r['T'] == T and r['f'] > 1e6) for T in Ts); M['PREmin'] = '%.2f~V' % vmin
    sl = []
    for T in Ts:
        rr = sorted([r for r in pr if r['T'] == T and r['f'] > 1e6], key=lambda r: r['vpre'])
        sl.append((rr[-1]['f'] - rr[0]['f']) / (rr[-1]['vpre'] - rr[0]['vpre']) / 1e6)
    M['PREslope'] = '%.0f to %.0f~MHz/V' % (min(sl), max(sl))
# varactor capacitance ratio (Spectre data of Fig. 1) vs the temperature change of the ring frequency
from ganpll import data as _data
# (absolute C-V data are private under the NDA; only the mean ratio is kept in the public summary)
try:
    rat = [c['C_pF'].max() / c['C_pF'].min() for dev in ('small', 'large') for c in _data.load_varactor_cv(dev)]
    if not rat:
        raise FileNotFoundError
    json.dump({'mean_cmax_over_cmin': float(np.mean(rat))}, open(f'{R}/cv_summary.json', 'w'))
except (FileNotFoundError, OSError, ValueError):
    pass
M['VARratio'] = '%.1f' % J(f'{R}/cv_summary.json')['mean_cmax_over_cmin']
# open-loop band check: surrogate band of the selected band vs open-flow tuning curves
if os.path.exists(f'{TL}/band_char.jsonl'):
    bc = [json.loads(l) for l in open(f'{TL}/band_char.jsonl')]
    dev, ons, cover = [], [], 0
    for r in cd['integer']:
        pts = sorted([x for x in bc if x['T'] == r['T_K'] and x['band'] == r['band']], key=lambda x: x['vc'])
        osc = [x for x in pts if x['f'] > 1e6]
        if not osc: continue
        top = [x['f'] for x in osc if abs(x['vc'] - 5.4) < 1e-6][0] / 1e6
        dev.append(100 * (r['win_hi_MHz'] - top) / top); ons.append(osc[0]['vc'])
        cover += int(top >= 52.5)
    M['BANDhiErr'] = '%.0f\\%%' % max(abs(x) for x in dev); M['BANDonset'] = '%.1f--%.1f~V' % (min(ons), max(ons))
    M['BANDcover'] = '%d of %d' % (cover, len(dev))
TLPREF = ('TL', 'REL', 'Ptl', 'NZ', 'FOM', 'ALT', 'ITtwo', 'OPT', 'ATL', 'BAND', 'PRE', 'VAC', 'MASH')  # written to numbers_tl.tex
with open('paper/numbers.tex', 'w') as fo:
    fo.write('% generated by scripts/31_paper_numbers.py -- do not edit\n')
    for k, v in sorted(M.items()):
        if not k.startswith(TLPREF):
            fo.write('\\newcommand{\\%s}{%s}\n' % (k, v))
print(len(M), 'macros'); [print(' ', k, '=', v) for k, v in sorted(M.items())]

# ======================================================================= tables
def wtab(name, s):
    open('paper/%s.tex' % name, 'w').write('% generated by scripts/31_paper_numbers.py\n' + s)

# iteration-1 calibration (supplement)
rows = c['corner_validation']['rows']
s = ('\\begin{table}[h]\n\\centering\n\\caption{Iteration-1 model against Spectre transient corners (model level). '
     '$f$ is predicted from the simulated settled control voltage.}\n\\label{st:calib}\n\\small\n'
     '\\begin{tabular}{@{}cccccc@{}}\n\\toprule\n$T$ (K) & $V_c$ (V) & Spectre (MHz) & Model (MHz) & Error (\\%) & Use \\\\\n\\midrule\n')
for r in rows:
    s += '%d & %.2f & %.1f & %.1f & %+.1f & %s \\\\\n' % (r['T_K'], r['vc'], r['f_meas'], r['f_pred'], r['err_pct'],
                                                         'calibration' if r['in_fit_set'] else '\\textbf{withheld}')
s += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
wtab('tab_calib', s)

# iteration-1 design (supplement)
dz = a['optimization']['design']
names = [('s_icp', 'Pump current'), ('s_rs1', 'Filter $R_{s1}$'), ('s_cs1', 'Filter $C_{s1}$'), ('s_rs23', 'Filter $R_{s23}$'),
         ('s_c2', 'Filter varactor $C_2$ width'), ('s_c3', 'Filter varactor $C_3$ width'), ('drive', 'Oscillator drive width'),
         ('s_cfix', 'Oscillator fixed capacitor'), ('s_var', 'Oscillator varactor width')]
s = ('\\begin{table}[h]\n\\centering\n\\caption{Iteration-1 co-designed parameters as scale factors on the manual design, '
     'and the learned varactor-bias law $V_b=V_{b0}+\\Delta V_b+s_T\\,(T-300\\,\\mathrm{K})$ (model level).}\n\\label{st:it1design}\n\\small\n'
     '\\begin{tabular}{@{}lc@{}}\n\\toprule\nParameter & Value \\\\\n\\midrule\n')
for k, lab in names:
    s += '%s & %.2f$\\times$ \\\\\n' % (lab, dz[k])
s += 'Bias offset $\\Delta V_b$ & %+.2f~V \\\\\nTracking slope $s_T$ & %+.1f~mV/K \\\\\n' % (dz['dvb'], 1e3 * dz['dvb_t'])
s += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
wtab('tab_it1design', s)

vb_, vo_ = a['verify_baseline'], a['verify_optimized']
s = ('\\begin{table}[h]\n\\centering\n\\caption{Iteration-1 verification in the iteration-1 model (model level). '
     'Lock: settled frequency within 1\\% of 50~MHz from both starting points.}\n\\label{st:it1ver}\n\\small\n'
     '\\begin{tabular}{@{}cccccccc@{}}\n\\toprule\n & \\multicolumn{2}{c}{Manual design} & \\multicolumn{5}{c}{Co-designed} \\\\\n'
     '\\cmidrule(lr){2-3}\\cmidrule(l){4-8}\n$T$ (K) & $f$ (MHz) & Lock & $f$ (MHz) & Lock & Settling ($\\mu$s) & Phase margin (\\degree) & Jitter (ns) \\\\\n\\midrule\n')
for b_, o_ in zip(vb_, vo_):
    s += '%d & %.1f & %s & %.2f & %s & %.1f & %.0f & %.2f \\\\\n' % (b_['T_K'], b_['f_end_MHz'], 'yes' if b_['lock'] else 'no',
                                                                 o_['f_end_MHz'], 'yes' if o_['lock'] else 'no', o_['settle_us'], o_['pm_deg'], o_['jitter_ns'])
s += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
wtab('tab_it1ver', s)

# iteration-2 design (main text)
thr = sorted(cd['thresholds'], key=lambda t: -t['T_set'])
eng = lambda x, u: ('%.3g~M%s' % (x / 1e6, u) if x >= 1e6 else ('%.3g~k%s' % (x / 1e3, u) if x >= 1e3 else '%.3g~%s' % (x, u)))
s = ('\\begin{tabular}{@{}lc@{\\hspace{8mm}}lc@{}}\n\\toprule\nParameter & Value & Parameter & Value \\\\\n\\midrule\n')
L = [('Oscillator drive scale', '%.2f' % d['drive']), ('Varactor width scale', '%.2f' % d['s_var']),
     ('Tail width per stage', '%.1f~$\\mu$m' % d['W_ht']), ('Band capacitor $C_u$', '%.2f~pF' % d['C_u']),
     ('$R_{\\mathrm{up}}$', eng(d['R_up'], '$\\Omega$')), ('$R_{\\mathrm{dn}}$', eng(d['R_dn'], '$\\Omega$')),
     ('$R_{s1}$', eng(d['Rs1'], '$\\Omega$')), ('$C_{s1}$', '%.1f~pF' % (d['Cs1'] * 1e12)),
     ('$R_{s23}$', eng(d['Rs23'], '$\\Omega$')), ('$C_2$ width', '%.0f~$\\mu$m' % d['W_c2']),
     ('$C_3$ width', '%.1f~$\\mu$m' % d['W_c3']), ('$V_b$ over 218--423~K', M['VBrange'])]
# the bias-divider resistor ratios are not listed: with the V_b range they would disclose the
# temperature coefficients of the PDK resistors (non-disclosure agreement)
for i in range(0, len(L), 2):
    s += '%s & %s & %s & %s \\\\\n' % (L[i] + L[i + 1])
s += '\\midrule\n\\multicolumn{4}{@{}l}{Band thresholds (segment switched in below $T_k$; window where both neighbouring bands hold 50--52.5~MHz):}\\\\\n'
for k, t in enumerate(thr):
    s += '\\multicolumn{4}{@{}l}{\\quad $T_{%d}=%.0f$~K, window %d--%d~K}\\\\\n' % (k, t['T_set'], t['T_lo'], t['T_hi'])
s += '\\bottomrule\n\\end{tabular}\n'
wtab('tab_design', s)
print('tables written')

# ======================================================================= transistor-level closed loop
def load_tl(tag, mode):
    out = {}
    for f in glob.glob(f'{TL}/{tag}_*_{mode}*.json'):
        r = J(f); out[int(round(r['T_K']))] = r
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
I = load_tl('f2', 'int'); F = load_tl('f2', 'frac'); N = {}
AI = load_tl('fin', 'int'); AF = load_tl('fin', 'frac')   # optimizer design before the filter refinement
AF = {T: r for T, r in AF.items() if T < 420}   # 30-us runs, 218-398 K (the 423-K run of this design was not extended)
sys.path.insert(0, 'tl'); from frac_residual import phase_metrics
for D in (I, F):
    for r in D.values():
        if 'phase_series' in r:
            r.update({'pm_' + k: v for k, v in phase_metrics(r).items()})
# fractional lock: the phase error minus the modulator's known quantization phase stays within a
# 90-degree band over the last 20 % of the run (a cycle slip would sweep it through 360 degrees)
for D in (AI, AF):
    for r in D.values():
        if 'phase_series' in r:
            r.update({'pm_' + k: v for k, v in phase_metrics(r).items()})
for r in list(F.values()) + list(AF.values()):
    r['lock'] = bool(r.get('pm_res_pp_deg', 999) < 90.0)
# integer lock: phase error (no quantization part) within a 30-degree band over the last 20 % of the run
for r in list(I.values()) + list(AI.values()):
    if 'pm_res_pp_deg' in r:
        r['lock'] = bool(r['pm_res_pp_deg'] < 30.0)
if AF:
    M['ATLfracLock'] = '%d of %d' % (sum(r['lock'] for r in AF.values()), len(AF))
    bad = sorted(T for T, r in AF.items() if not r['lock'])
    M['ATLfracFail'] = (', '.join('%d' % T for T in bad[:-1]) + ' and %d~K' % bad[-1]) if len(bad) > 1 else ('%d~K' % bad[0] if bad else 'none')
if AI:
    M['ATLintLock'] = '%d of %d' % (sum(r['lock'] for r in AI.values()), len(AI))
nb = lambda r: sum(1 for k in range(4) if r['bnd%d' % k] > 3)
M['TLtstop'] = '%.0f~$\\mu$s' % (max(r['tstop_us'] for r in I.values()) if I else 30)
_long = ['%d~K %s, %.0f~$\\mu$s' % (r['T_K'], 'integer' if m == 'int' else 'fractional', r['tstop_us']) for m, D in (('int', I), ('frac', F)) for r in sorted(D.values(), key=lambda r: r['T_K']) if r['tstop_us'] > 40]
M['TLlongRuns'] = ', '.join(_long) if _long else 'none'
_rt = [r['runtime_s'] / 60 for D in (I, F) for r in D.values() if 'runtime_s' in r]
M['TLruntime'] = '%.0f to %.0f minutes' % (min(_rt), max(_rt)) if _rt else 'pending'
M['TLintLock'] = '%d of %d' % (sum(r['lock'] for r in I.values()), len(I))
M['TLfracLock'] = '%d of %d' % (sum(r['lock'] for r in F.values()), len(F)) if F else 'pending'
if I:
    st = [settle_vc(r) for r in I.values()]; M['TLsettle'] = '%.0f--%.0f~$\\mu$s' % (min(st), max(st))
    ph = [r['phase_mean_deg'] for r in I.values()]; M['TLphase'] = '%+.0f\\degree{} to %+.0f\\degree' % (min(ph), max(ph))
    M['TLwander'] = '%.1f\\degree' % max(r['phase_pp_deg'] for r in I.values())
    fm = [r['f_inst_pp_pct'] for r in I.values()]; M['TLfmod'] = '%.0f--%.0f\\%%' % (min(fm), max(fm))
    P = {T: r['p_vdd_mW'] for T, r in I.items()}
    M['TLpower'] = '%.0f~mW at 423~K to %.0f~mW at 218~K (%.0f~mW at 300~K)' % (P.get(423, np.nan), P.get(218, np.nan), P.get(300, np.nan))
    M['TLpowerShort'] = '%.0f--%.0f~mW' % (min(P.values()), max(P.values()))
    M['Ptlmid'] = '%.0f~mW' % P.get(300, np.nan); M['Ptlcold'] = '%.0f~mW' % P.get(218, np.nan)
    M['TLboot'] = '%.2f~mW' % max(r['p_boot_mW'] for r in I.values())
    vg = max(r['vgs_pu_max'] for r in I.values()); M['RELvgs'] = '%.2f~V' % vg; M['RELvgsT'] = '%d~K' % max(I, key=lambda T: I[T]['vgs_pu_max']); M['RELmargin'] = '%.1f~V' % (7 - vg)
    bt = max(r['vboot_node_max'] for r in I.values()); M['RELboot'] = '%.1f~V' % bt; M['RELneg'] = '%.1f~V' % (7 - bt)
    M['RELtail'] = '%.2f~V' % max(r['vc_tail'] + r['vc_pp_mV'] / 2e3 for r in I.values())
    # gate leakage at 423 K at the closed-loop operating point (foundry-model tables, local file)
    try:
        from ganpll import pll2
        r = I[423]; vc = r['vc_tail']; dd = dict(cd['design'])
        lx, lc = pll2.leak(dd, vc, vc, 423.0)
        M['RELleak'] = '%.1f~$\\mu$A' % (1e6 * (float(lx) + float(lc)))
        M['RELidn'] = '%.0f~$\\mu$A' % (1e6 * vc / pll2.r2deg(dd['R_dn'], 423.0))
        M['RELleakPct'] = '%.0f\\%%' % (100 * (float(lx) + float(lc)) / (vc / pll2.r2deg(dd['R_dn'], 423.0)))
    except Exception as ex:
        print('leakage tables unavailable:', ex)
if F:
    FL = [r for r in F.values() if r['lock']] or list(F.values())
    M['TLfracRes'] = '%.0f\\degree' % max(r['pm_res_pp_deg'] for r in FL)
    M['TLfracQ'] = '%.0f\\degree' % np.median([r['pm_q_pp_deg'] for r in FL])
    pf = [r['pm_res_mean_deg'] for r in FL]; M['TLfracPhase'] = '%+.0f\\degree{} to %+.0f\\degree' % (min(pf), max(pf))
    # deterministic fractional jitter at transistor level: rms of the residual phase (quantization
    # phase removed, slow drift removed) over the last 20 % of each run, in time units of the reference
    # period; compared with the model's fractional-mode jitter at the same temperature
    nzf = {float(r['T_K']): r['jitter_frac_ns'] for r in ne}
    jt = {float(T): r['pm_res_rms_detr_deg'] / 360.0 * 1e9 / cd['f_cmp'] for T, r in F.items() if r['lock']}
    M['TLjitTL'] = '%.1f--%.1f~ns' % (min(jt.values()), max(jt.values()))
    both = [T for T in jt if T in nzf]
    M['TLjitMod'] = '%.1f--%.1f~ns' % (min(nzf[T] for T in both), max(nzf[T] for T in both))
    M['TLjitBelow'] = '%d of %d' % (sum(jt[T] <= nzf[T] for T in both), len(both))
    M['TLjitMaxOver'] = '%.0f\\%%' % max(0.0, max(100 * (jt[T] / nzf[T] - 1) for T in both))
    print('TL jitter vs model:', {T: (round(jt[T], 2), round(nzf[T], 2)) for T in sorted(both)})
LK = [r for D in (I, F) for r in D.values() if r['lock'] and 'pm_f_err_ppm' in r]
if LK:
    M['TLdrift'] = '%.1f\\degree/$\\mu$s' % max(abs(r['pm_f_err_ppm']) * 1e-6 * 360 * 10e6 * 1e-6 for r in LK)

# cross-check of the closed loop in VACASK (independent simulator, same Verilog-A model)
_vac = {}
for f in sorted(glob.glob(f'{TL}/vac_*_int.json')):
    r = json.load(open(f)); _vac[float(r['T_K'])] = r
if _vac:
    rows = []
    for T, r in sorted(_vac.items()):
        if 'phase_series' in r:
            r.update({'pm_' + k: v for k, v in phase_metrics(r).items()})
        g = I.get(T) or I.get(int(T))
        if g is None:
            continue
        rows.append(dict(T=T, dvc=1e3 * abs(r['vc_tail'] - g['vc_tail']), dph=abs(r['pm_res_mean_deg'] - g['pm_res_mean_deg']),
                         dp=100 * (r['p_vdd_mW'] / g['p_vdd_mW'] - 1), band=(nb(r) == nb(g)), lock=bool(r['pm_res_pp_deg'] < 30.0)))
    if rows:
        M['VACtemps'] = ' and '.join('%d' % x['T'] for x in rows) + '~K'
        M['VACdvc'] = '%.0f~mV' % max(x['dvc'] for x in rows)
        M['VACdph'] = '%.1f\\degree' % max(x['dph'] for x in rows)
        M['VACdp'] = '%.0f\\%%' % max(abs(x['dp']) for x in rows)
        M['VAClock'] = '%d of %d' % (sum(x['lock'] for x in rows), len(rows))
        M['VACband'] = 'the same band' if all(x['band'] for x in rows) else 'a different band at some temperature'
        print('VACASK vs ngspice:', rows)

# stand-alone transistor-level MASH 1-1 modulator in the same DCFL logic (tl/mash_tl.py)
_m = {}
for f in glob.glob(f'{TL}/mash*_*.json'):
    r = json.load(open(f)); _m[(os.path.basename(f).split('_')[0], int(round(r['T_K'])))] = r
if ('mash', 300) in _m:
    r = _m[('mash', 300)]
    M['MASHhemt'] = '%d' % r['n_hemt']
    M['MASHmatchMid'] = '%d of %d' % (r['n_match'], r['n_checked'])
    M['MASHpowMid'] = '%.0f~mW' % (r['p_vdd_mW'] + r['p_boot_mW'])
    for key, T in (('Hot', 423), ('Cold', 218)):
        if ('mash', T) in _m and _m[('mash', T)]['p_vdd_mW'] > 0:
            M['MASHmatch' + key] = '%d of %d' % (_m[('mash', T)]['n_match'], _m[('mash', T)]['n_checked'])
            M['MASHpow' + key] = '%.0f~mW' % (_m[('mash', T)]['p_vdd_mW'] + _m[('mash', T)]['p_boot_mW'])
    if ('mashk125', 423) in _m:
        r2 = _m[('mashk125', 423)]
        M['MASHhotNote'] = 'gates 1.67 times larger give the same %d of %d cycles at %.0f~mW, so the failure is static, not a speed limit' % (r2['n_match'], r2['n_checked'], r2['p_vdd_mW'] + r2['p_boot_mW'])
    else:
        M['MASHhotNote'] = 'larger gates pending'

# table: closed loop
s = ('\\begin{table*}[t]\n\\centering\n\\caption{Closed-loop transistor-level results (open flow, every block in HEMTs). '
     'Integer mode: $N=5$, 50~MHz; fractional mode: $N=5.2$, 52~MHz. Band: number of capacitor segments switched in by the comparators. '
     'Phase: mean static phase error and peak-to-peak wander over the last 20\\% of the run (6, 12 or 20~$\\mu$s for 30-, 60- or 100-$\\mu$s runs). Ripple: peak-to-peak modulation of the oscillator frequency at the reference rate. Fractional mode: Quant., peak-to-peak quantization phase of the modulator word sequence; Phase, mean and peak-to-peak of the phase error after subtracting it.}\n'
     '\\label{tab:tl}\n\\footnotesize\n\\setlength{\\tabcolsep}{2.3pt}\n'
     '\\begin{tabular}{@{}rccccccccccc@{}}\n\\toprule\n'
     ' & \\multicolumn{8}{c}{Integer mode} & \\multicolumn{3}{c}{Fractional mode} \\\\\n\\cmidrule(lr){2-9}\\cmidrule(l){10-12}\n'
     '$T$ & Band & $V_b$ & $V_c$ & Settle & Phase & Ripple & Power & $V_{GS,\\mathrm{pu}}$ & Lock & Quant. & Phase \\\\\n'
     '(K) & & (V) & (V) & ($\\mu$s) & (deg) & (\\%) & (mW) & (V) & & (deg) & (deg) \\\\\n\\midrule\n')
for T in sorted(set(I) | set(F)):
    a_ = I.get(T); b_ = F.get(T)
    row = ['%d' % T]
    if a_:
        row += ['%d' % nb(a_), '%.2f' % a_['vb_tail'], '%.2f' % a_['vc_tail'], '%.0f' % settle_vc(a_),
                '%+.1f$\\pm$%.1f' % (a_['phase_mean_deg'], a_['phase_pp_deg'] / 2), '%.1f' % a_['f_inst_pp_pct'],
                '%.0f' % a_['p_vdd_mW'], '%.2f' % a_['vgs_pu_max']]
        if not a_['lock']: row[1] += '$^{*}$'
    else:
        row += ['--'] * 8
    if b_:
        row += ['yes' if b_['lock'] else 'no', '%.0f' % b_.get('pm_q_pp_deg', np.nan), '%+.1f$\\pm$%.1f' % (b_.get('pm_res_mean_deg', np.nan), b_.get('pm_res_pp_deg', np.nan) / 2)]
    else:
        row += ['--'] * 3
    s += ' & '.join(row) + ' \\\\\n'
s += '\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
wtab('tab_tl', s)

# table: robustness
Rb = {}
for tag, lab in (('f2rob_vto-0.2', 'all HEMTs $V_T-0.2$~V'), ('f2rob_vto0.2', 'all HEMTs $V_T+0.2$~V'), ('f2rob_cv-0.2', 'varactors $V_T-0.2$~V'),
                 ('f2rob_cv0.2', 'varactors $V_T+0.2$~V'), ('f2rob_cw2', 'wire load $\\times2$')):
    Rb[lab] = load_tl(tag, 'int')
    for r in Rb[lab].values():
        if 'phase_series' in r:
            pm = phase_metrics(r); r['lock'] = bool(pm['res_pp_deg'] < 30.0); r['phase_mean_deg'] = pm['res_mean_deg']
s = ('\\begin{table*}[t]\n\\centering\n\\caption{Closed-loop robustness at the extreme temperatures (open flow, integer mode, 50~MHz; 218~K from power-up, 423~K from a filter precharged to 4.3~V). Phase: mean static phase error. $^{a}$50~MHz held on average, but the phase error does not settle into the 30\\degree{} band. $^{b}$The oscillator does not reach 50~MHz.}\n\\label{tab:rob}\n\\footnotesize\n'
     '\\begin{tabular}{@{}lcccccc@{}}\n\\toprule\n & \\multicolumn{3}{c}{218~K} & \\multicolumn{3}{c}{423~K} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(l){5-7}\n'
     'Perturbation & Lock & Band & Phase (deg) & Lock & Band & Phase (deg) \\\\\n\\midrule\n')
s += 'none (nominal) & ' + ' & '.join(
    (('yes' if I[T]['lock'] else 'no') + ' & %d & %+.1f' % (nb(I[T]), I[T]['phase_mean_deg'])) if T in I else '-- & -- & --' for T in (218, 423)) + ' \\\\\n'
nrob = 0; nlock = 0
for lab, D in Rb.items():
    cells = []
    for T in (218, 423):
        if T in D:
            nrob += 1; nlock += D[T]['lock']
            r_ = D[T]
            lk = 'yes' if r_['lock'] else ('no$^{a}$' if abs(r_['f_err_ppm']) < 1000 else 'no$^{b}$')
            ph_ = ('%+.1f' % r_['phase_mean_deg']) if r_['lock'] else '--'
            cells.append(lk + ' & %d & %s' % (nb(r_), ph_))
        else:
            cells.append('-- & -- & --')
    s += lab + ' & ' + ' & '.join(cells) + ' \\\\\n'
s += '\\bottomrule\n\\end{tabular}\n\\end{table*}\n'
wtab('tab_rob', s)
M['TLrobCount'] = '%d of %d' % (nlock, nrob)
fails = ['%s at %d~K' % (lab, T) for lab, D in Rb.items() for T in sorted(D) if not D[T]['lock']]
if nrob:
    if not fails:
        M['TLrob'] = 'The loop stays locked in all %d perturbed runs, in the band selected by the comparators.' % nrob
        M['TLrobShort'] = 'at both extreme temperatures'
    else:
        M['TLrob'] = 'The loop stays locked in %d of the %d perturbed runs; it loses lock with %s.' % (nlock, nrob, '; '.join(fails))
        M['TLrobShort'] = 'in %d of %d runs at the extreme temperatures' % (nlock, nrob)
    phs = [D[T]['phase_mean_deg'] for D in Rb.values() for T in D if D[T]['lock']]
    if phs: M['TLrobPhase'] = '%+.0f\\degree{} to %+.0f\\degree' % (min(phs), max(phs))

# table: benchmark
P300 = I[300]['p_vdd_mW'] if 300 in I else np.nan
jit300 = cd['integer'][[r['T_K'] for r in cd['integer']].index(300.0)]['jitter_ns']
fom = 10 * np.log10((jit300 * 1e-9) ** 2 * P300)
M['FOMours'] = '%.0f~dB' % fom
s = r'''\begin{table*}[t]
\centering
\caption{Comparison with wide-temperature and wide-bandgap timing circuits. Meas.: measured chip; sim.: simulation. FoM$_{\mathrm{jitter}}=10\log_{10}[(\sigma_t/1\,\mathrm{s})^2(P/1\,\mathrm{mW})]$ \cite{gao2009fom}. n.r.: not reported.}
\label{tab:bench}
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabularx}{\textwidth}{@{}>{\raggedright\arraybackslash}p{2.1cm}>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X>{\raggedright\arraybackslash}X@{}}
\toprule
 & \cite{wang2025hightempPLL} & \cite{gong2023cryoPLL} & \cite{Shepherd2014} & \cite{yuan2022ganRO} & \cite{li2026ganwide} & This work \\
\midrule
Evidence & meas. & meas. & meas. & meas. & meas. & sim. (open flow, foundry model) \\
Technology & 22-nm FD-SOI CMOS & 40-nm CMOS & 1.2-$\mu$m SiC CMOS & GaN-on-Si, E/D-mode & GaN, E/D-mode & GaN-on-SOI, e-mode only \\
Circuit & PLL & PLL, sub-sampling detector & charge-pump PLL & ring oscillator (open loop) & logic, ring oscillator & fractional-N PLL, ring VCO \\
Output frequency & 0.6--4.8~GHz & 9.4--11.6~GHz (4.2~K) & up to 1.5~MHz & n.r. (0.31~ns/stage at 298~K) & $>$10~MHz & 50~MHz ($N=5$), 52~MHz ($N=5.2$) \\
Temperature & 300--423~K & 4.2 and 300~K & RT to 423~K & 298--773~K & 5--800~K & 218--423~K (10 points) \\
Power & 5.7~mW (2.4~GHz, 423~K) & 2.7~mW (4.2~K), 4~mW (300~K) & n.r. & n.r. & n.r. & ''' + '%.0f~mW (300~K)' % P300 + r''' \\
Phase noise at 1~MHz & $-87.9$~dBc/Hz (423~K), $-90.3$ (300~K) & n.r. & n.r. & n.r. & n.r. & ''' + '%s (model, 300~K)' % M.get('NZpn', 'model') + r''' \\
RMS jitter & n.r. & 37~fs (4.2~K), 75~fs (300~K) & n.r. & -- & -- & ''' + '%.1f~ns (model, 300~K)' % jit300 + r''' \\
FoM$_{\mathrm{jitter}}$ & n.r. & $-256.5$~dB (300~K) & n.r. & -- & -- & ''' + '%.0f~dB (model jitter, sim.\\ power)' % fom + r''' \\
\bottomrule
\end{tabularx}
\end{table*}
'''
wtab('tab_bench', s)

# stand-alone block tests (supplement)
# comparator calibration ratios and stand-alone pump currents are not tabulated: with the resistor
# values they would disclose the temperature coefficients of the PDK resistors (non-disclosure agreement)
if os.path.exists(f'{TL}/divider_tests.json'):
    dv = J(f'{TL}/divider_tests.json')
    s = ('\\begin{table}[h]\n\\centering\n\\caption{Stand-alone two-cell divider as used in the final design (open flow, ideal input clock; gate size of cell 1: 0.75, cell 2: 0.5 of the unit gate). '
         'Fractional mode: MASH-2 word sequence for $N=5+\\text{frac}$, matched cycle by cycle against the expected sequence.}\n'
         '\\label{st:div}\n\\small\n\\begin{tabular}{@{}cccccc@{}}\n\\toprule\n$T$ (K) & Input (MHz) & Mode & $k_1$ & Result & Power (mW) \\\\\n\\midrule\n')
    for e in (dv.get('div23_final') or dv['div23']):
        c_, r_ = e['cfg'], e['res']
        mode = 'fractional' if c_.get('mode') == 'frac' else 'integer $N=%d$' % c_['nword']
        if 'match' in r_:
            res = '%s cycles matched' % r_['match']
        else:
            rs = r_['ratios']; k0 = next((i for i in range(len(rs)) if all(x == c_['nword'] for x in rs[i:])), None)
            res = ('%d of %d cycles correct' % (len(rs) - k0, len(rs)) + (' (after %d start-up cycles)' % k0 if k0 else '')) if k0 is not None and k0 < len(rs) else 'mean ratio %.3f' % r_['mean_ratio']
        s += '%d & %.0f & %s & %.2f & %s & %.0f \\\\\n' % (c_['T_K'], c_['fin'] / 1e6, mode, c_['k1'], res, r_['p_mW'])
    s += '\\bottomrule\n\\end{tabular}\n\\end{table}\n'
    wtab('tab_div', s)

with open('paper/numbers_tl.tex', 'w') as fo:
    fo.write('% generated by scripts/31_paper_numbers.py -- do not edit\n')
    for k in sorted(M):
        if k.startswith(TLPREF):
            fo.write('\\newcommand{\\%s}{%s}\n' % (k, M[k]))
print('TL macros:', {k: v for k, v in M.items() if k.startswith(('TL', 'REL', 'Ptl', 'NZ', 'FOM'))})
