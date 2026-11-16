#!/usr/bin/env python3
"""20_fit_tl_vcro.py -- fit the transistor-anchored oscillator surrogate (tl_anchor) to the open-flow DOE
with five-fold cross-validation.  usage: python3 scripts/20_fit_tl_vcro.py results/tl/doe_all.jsonl 5000 results/tl_vcro.json"""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ganpll import pinn, tl_anchor as ta
pst = pinn.load_state(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'pinn_state.json'))
X, f, osc, idd, vgs = ta.load_doe(sys.argv[1])
print('points', len(f), 'oscillating', int(osc.sum()))
rng = np.random.default_rng(0); idx = rng.permutation(len(f)); K = 5
errs, accs = [], []
for k in range(K):
    te = idx[k::K]; tr = np.setdiff1d(idx, te)
    p, L = ta.fit(X[tr], f[tr], osc[tr], pst, steps=int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
    fm = np.asarray(ta.f_model(p, pst, *X[te].T)); po = np.asarray(ta.p_osc(p, *X[te].T))
    m = osc[te] > 0
    e = np.abs(fm[m] - f[te][m]) / f[te][m]
    errs += e.tolist(); accs += ((po > 0.5) == (osc[te] > 0)).tolist()
    print(f'fold {k}: blind freq MAPE {100*e.mean():.2f}% (median {100*np.median(e):.2f}%), onset accuracy {100*np.mean((po>0.5)==(osc[te]>0)):.1f}%', flush=True)
print(f'5-fold blind: freq MAPE {100*np.mean(errs):.2f}%, p90 {100*np.percentile(errs,90):.2f}%, onset accuracy {100*np.mean(accs):.1f}%')
p, L = ta.fit(X, f, osc, pst, steps=int(sys.argv[2]) if len(sys.argv) > 2 else 4000)
mo = osc > 0
pw = ta.ridge_fit(X[mo], np.log(idd[mo] * 6.0))
vg = ta.ridge_fit(X[mo], vgs[mo])
res_pw = np.exp(ta.ridge_eval(pw, X[mo])) / (idd[mo] * 6.0) - 1
res_vg = ta.ridge_eval(vg, X[mo]) - vgs[mo]
print(f'power model MAPE {100*np.mean(np.abs(res_pw)):.2f}%, stress model MAE {np.mean(np.abs(res_vg))*1e3:.0f} mV')
ta.save(p, sys.argv[3], extra=dict(power=pw, vgs=vg, cv_freq_mape=float(np.mean(errs)),
        cv_freq_p90=float(np.percentile(errs, 90)), cv_onset_acc=float(np.mean(accs)), n=int(len(f))))
print('saved', sys.argv[3])
