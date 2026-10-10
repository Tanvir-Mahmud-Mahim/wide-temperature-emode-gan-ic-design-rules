"""Minimum bootstrap-precharge voltage versus temperature for the co-designed oscillator stage.
Open-loop five-stage ring (final sizing, band capacitance of the band selected at each
temperature, varactor bias from the on-chip divider) with the chargers fed from a rail V_pre; V_b is taken from the closed-loop run at the same temperature.
usage: python3 precharge_sweep.py designs/final_f2.json out.jsonl"""
import sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
import os
import vcro, doe

def job(a):
    T, vpre, vc, d, cb, vb = a
    p = doe.design_params(d['drive'], d['s_var'], vb, cb)
    p['vboot'] = vpre
    r = vcro.tb(p, T - 273.15, vc, tstop=2e-6)
    return dict(T=T, vpre=vpre, vc=vc, vb=vb, cband_pF=cb * 1e12, f=r['f'], swing=r['swing'],
                vmax=r['vmax'], bmax=r['bmax'], vgs_max=r['vgs_max'])

if __name__ == '__main__':
    D = json.load(open(sys.argv[1])); d = D['design']
    jobs = []
    for T in (300.0, 348.0, 373.0, 398.0, 423.0):
        nb = sum(1 for s in D['band']['segments'] if T < s['T_set'])
        vb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), 'f2_%d_int.json' % T)))['vb_tail']  # closed-loop V_b
        for vpre in (6.0, 6.25, 6.5, 6.75, 7.0):
            jobs.append((T, vpre, 4.6, d, nb * d['C_u'] * 1e-12, vb))
    with ProcessPoolExecutor(1) as ex, open(sys.argv[2], 'w') as fo:
        for r in ex.map(job, jobs):
            fo.write(json.dumps(r) + '\n'); fo.flush(); print(r, flush=True)
