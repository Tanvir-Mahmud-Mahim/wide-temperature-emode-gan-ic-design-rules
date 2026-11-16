#!/usr/bin/env python3
"""23_noise_eval.py -- model-level noise of the iteration-2 design: integer and fractional
RMS jitter (1 kHz to f_cmp/2) and output phase noise at 1 MHz offset, at ten temperatures.
Uses the settled control voltage of the iteration-2 verification.  Writes results/noise_eval.json."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax.numpy as jnp
from ganpll import pinn, tl_anchor as ta, pll2
pst = pinn.load_state('results/pinn_state.json'); tp, ex = ta.load('results/tl_vcro.json')
cd = json.load(open('results/tl_codesign.json'))
d = {k: jnp.asarray(v) for k, v in cd['design'].items()}
rows = []
for r in cd['integer']:
    T = r['T_K']; dd = dict(d, cband=r['band'] * float(d['C_u'])); vc = r['vc_end']; pw = r['p_vco_mW'] * 1e-3
    out = dict(T_K=T)
    for tag, n, fo in (('int', 5.0, 0.0), ('frac', 5.2, 1.0)):
        sig, sp = pll2.jitter(tp, pst, dd, T, vc, n, fo, pw)
        f = np.asarray(sp['f']); s = np.asarray(sp['s_out'])
        out['jitter_%s_ns' % tag] = float(sig) * 1e9
        out['pn_1MHz_%s_dBcHz' % tag] = float(10 * np.log10(np.interp(1e6, f, s) / 2))
        if tag == 'int':
            L = np.asarray(sp['L']); i = int(np.argmin(np.abs(np.abs(L) - 1.0)))
            out['f_ugb_kHz'] = float(f[i] / 1e3); out['pm_deg'] = float(180 + np.degrees(np.angle(L[i])))
    rows.append(out); print(out)
json.dump(rows, open('results/noise_eval.json', 'w'), indent=1)
