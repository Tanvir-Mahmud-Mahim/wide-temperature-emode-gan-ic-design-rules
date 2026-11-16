"""Closed-loop run of the final design in VACASK (independent open-source simulator, same MVSG
model compiled by OpenVAF-Reloaded), analysed exactly like tlclosed.py.
usage: python3 vacask_closed.py design.json T_K --tstop 30e-6 --tag vac"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pllnet, ng2vacask
from tlclosed import analyse

ap = argparse.ArgumentParser()
ap.add_argument('design'); ap.add_argument('T', type=float)
ap.add_argument('--tstop', type=float, default=30e-6); ap.add_argument('--tag', default='vac')
ap.add_argument('--tmax', type=float, default=50e-12); ap.add_argument('--icvc', type=float, default=None)
a = ap.parse_args()
D = json.load(open(a.design)); d, bias, band = D['design'], D['bias'], D.get('band')
net, ic = pllnet.build(a.T, d, mode='int', n_int=5, f_cmp=10e6, bias=bias, band=band, ic_vc=a.icvc)
nseg = len(band['segments'])
wv = ['ref', 'fb', 'ck', 'vc', 'o0', 'b0', 'vbn'] + ['bnd%d' % k for k in range(nseg)]
t0 = time.time()
w = ng2vacask.run(net.text(), wv, a.tstop, tmax=a.tmax, ic=ic, save_extra=['i(vdd)', 'i(vbt)'])
res = analyse(w, 10e6, 5.0, a.tstop)
m = w['t'] > 0.8 * a.tstop
for key, name in (('i(vdd)', 'p_vdd_mW'), ('i(vbt)', 'p_boot_mW')):
    if key in w:
        V = 6.0 if key == 'i(vdd)' else 7.0
        res[name] = float(abs(np.mean(w[key][m])) * V * 1e3)
dec = max(1, len(w['t']) // 20000)
res['wave'] = dict(t=w['t'][::dec].tolist(), vc=w['vc'][::dec].tolist())
res.update(icvc=a.icvc, T_K=a.T, mode='int', n_div=5.0, simulator='VACASK 0.3.4', runtime_s=time.time() - t0, tstop_us=a.tstop * 1e6)
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'tl'); os.makedirs(RES, exist_ok=True)
json.dump(res, open(RES + '/%s_%d_int%s.json' % (a.tag, round(a.T), '' if a.icvc is None else '_pre'), 'w'))
print({k: v for k, v in res.items() if k not in ('wave', 'phase_series')})
