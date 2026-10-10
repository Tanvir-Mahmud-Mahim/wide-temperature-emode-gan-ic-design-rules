"""Transient-noise estimate of the random jitter of the final five-stage ring oscillator (open loop),
run in VACASK with the noise sources of the MVSG compact model switched on (noisemod=1: gate shot
noise, channel thermal noise, thermal noise of the access and field-plate resistances; the flicker
coefficient keeps its default kf=0) and thermal noise of all resistors.

The foundry model card does not enable these sources, so this is an uncalibrated, white-noise-only
estimate from the compact model's own physical noise expressions, not a foundry-calibrated phase
noise. A run without noise (same time step) gives the numerical jitter floor.

Period jitter sigma_T of the free-running ring gives, for white noise, the phase noise
L(df) = 10 log10( f0^3 sigma_T^2 / df^2 )  (accumulated-jitter relation for white noise).

usage: python3 ring_noise.py designs/final_f2.json T_K [--tstop 3e-6] [--fmax 5e9] [--seed 1] [--off]
"""
import sys, os, json, time, argparse, re
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlib, vcro, doe, ng2vacask

ap = argparse.ArgumentParser()
ap.add_argument('design'); ap.add_argument('T', type=float)
ap.add_argument('--tstop', type=float, default=3e-6); ap.add_argument('--fmax', type=float, default=1e9)
ap.add_argument('--seed', type=int, default=1); ap.add_argument('--off', action='store_true')
ap.add_argument('--tmax', type=float, default=10e-12); ap.add_argument('--scale', type=float, default=1.0); ap.add_argument('--tag', default='rn')
a = ap.parse_args()

D = json.load(open(a.design)); d = D['design']
cl = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'tl', 'f2_%d_int.json' % round(a.T))))
vc, vb = cl['vc_tail'], cl['vb_tail']                         # closed-loop operating point
nb = sum(1 for s in D['band']['segments'] if a.T < s['T_set'])
p = doe.design_params(d['drive'], d['s_var'], vb, nb * d['C_u'] * 1e-12)
p['vboot'] = 7.0

net = netlib.Net('ring noise T=%g' % a.T, a.T - 273.15)
net.add('vdd vdd 0 dc 6'); net.add('vvc vc 0 dc %g' % vc); net.add('vvb vbn 0 dc %g' % vb)
net.add('vvboot vboot 0 dc 7'); net.add('vband band 0 dc 6')
vcro.add_vcro(net, p)
txt = net.text()
if not a.off:
    txt = txt.replace('ngf=1)', 'ngf=1 noisemod=1)')
sim = ng2vacask.translate(txt, ic={'o0': 6.0, 'o1': 0.0}, tstop=a.tstop, tmax=a.tmax, save=['o0'])
if not a.off:
    sim = re.sub(r'(analysis tran1 tran [^\n]*)', r'\1 noisefmax=%g noisefmin=%g noiseseed=%d noisescale=%g' % (a.fmax, 1e6, a.seed, a.scale), sim)

import tempfile, subprocess
dd = tempfile.mkdtemp(prefix='rn_', dir=ng2vacask.SCRATCH)
open(os.path.join(dd, 'pll.sim'), 'w').write(sim)
t0 = time.time()
r = subprocess.run([ng2vacask.VACASK, 'pll.sim'], cwd=dd, capture_output=True, text=True, timeout=30000)
raw = os.path.join(dd, 'tran1.raw')
if not os.path.exists(raw):
    print((r.stdout + r.stderr)[-3000:]); sys.exit(1)
sys.path.insert(0, os.environ.get('VACASK_PY', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(ng2vacask.VACASK))), 'lib', 'vacask', 'python')))
from rawfile import rawread
w = rawread(raw).get(); t = np.asarray(w['time']); v = np.asarray(w['o0'])
m = t > 0.3 * a.tstop                                          # skip start-up
tt, vv = t[m], v[m]; mid = 0.5 * (vv.min() + vv.max()); s = vv > mid
i = np.where(s[1:] & ~s[:-1])[0]
tc = tt[i] + (mid - vv[i]) / (vv[i + 1] - vv[i]) * (tt[i + 1] - tt[i])   # rising crossings
P = np.diff(tc); f0 = 1.0 / P.mean()
sig_raw = float(np.std(P, ddof=1))
# the self-heating thermal nodes of the compact model drift slowly, so the period drifts over the
# run; remove it with a cubic fit over the period index and use the residual as period jitter
ix = np.arange(len(P)); Pd = P - np.polyval(np.polyfit(ix, P, 3), ix)
sig = float(np.std(Pd, ddof=4))
sig_cc = float(np.std(np.diff(P), ddof=1))           # cycle-to-cycle jitter (insensitive to drift)
res = dict(scale=a.scale, T_K=a.T, vc=vc, vb=vb, band=nb, noise=not a.off, fmax=a.fmax, seed=a.seed, tstop_us=a.tstop * 1e6,
           tmax_ps=a.tmax * 1e12, n_periods=int(len(P)), f0_MHz=float(f0 / 1e6), sigma_T_ps=sig * 1e12, sigma_T_raw_ps=sig_raw * 1e12, sigma_cc_ps=sig_cc * 1e12, periods_ps=(P * 1e12).tolist(),
           sigma_T_rel=float(sig * f0), runtime_s=time.time() - t0)
if not a.off:
    res['L1MHz_dBcHz'] = float(10 * np.log10(f0 ** 3 * sig ** 2 / 1e6 ** 2))
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'tl'); os.makedirs(RES, exist_ok=True)
json.dump(res, open(RES + '/%s_%d_%s.json' % (a.tag, round(a.T), 'off' if a.off else ('on%d' % a.seed + ('' if a.scale == 1 else '_x%g' % a.scale))), 'w'))
print({k: v for k, v in res.items() if k != 'periods_ps'})
