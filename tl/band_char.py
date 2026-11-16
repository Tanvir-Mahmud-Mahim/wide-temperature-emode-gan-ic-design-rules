"""Open-loop transistor-level tuning curves of the co-designed VCRO for each temperature
and band (selected band and its neighbours): checks the surrogate's lockable windows.
usage: python3 band_char.py designs/X.json out.jsonl"""
import sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
import vcro, doe

from netlib import pdk_local
def g2(T): return pdk_local.r2deg_factor(T)
def hom(T): return pdk_local.rom_factor(T)

def job(a):
    T, b, vc, d, cu, vb = a
    p = doe.design_params(d['drive'], d['s_var'], vb, b * cu * 1e-12 if b > 0 else 0.0)
    r = vcro.tb(p, T - 273.15, vc, tstop=2e-6)
    return dict(T=T, band=b, vc=vc, vb=vb, f=r['f'], swing=r['swing'], vgs_max=r['vgs_max'], idd=r['idd'])

if __name__ == '__main__':
    D = json.load(open(sys.argv[1])); d = D['design']; cu = d['C_u']
    src = json.load(open(D['source']))
    bands = {float(k): v for k, v in src['bands'].items()}
    jobs = []
    for T, bsel in sorted(bands.items()):
        vb = 6.0 * (d['a2'] * g2(T) + d['ao'] * hom(T)) / (d['a2'] * g2(T) + d['ao'] * hom(T) + hom(T))
        for b in sorted({max(bsel - 1, 0), bsel, min(bsel + 1, 4)}):
            for vc in np.arange(3.2, 5.41, 0.2):
                jobs.append((T, b, round(float(vc), 2), d, cu, vb))
    with ProcessPoolExecutor(2) as ex, open(sys.argv[2], 'a') as fo:
        for r in ex.map(job, jobs):
            fo.write(json.dumps(r) + '\n'); fo.flush()
