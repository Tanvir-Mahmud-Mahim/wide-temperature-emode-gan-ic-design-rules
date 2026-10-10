"""Transistor-level VCRO design-of-experiments (bootstrapped stage, 7 V precharge rail).
Usage: python3 doe.py points.json out.jsonl [shard k of n]"""
import json, sys, os
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import vcro

CW, RW = 150e-15, 2.9   # mean extracted per-stage wire load (layout PEX)

def design_params(drive, s_var, vb, cband=0.0):
    return dict(vcro.BASE, Cband=cband, W_hu=6.5e-6 * drive, W_hd=32e-6 * drive, W_hs=26e-6 * drive,
                W_ht=26e-6 * drive, Cb=3.5e-12 * drive, W_var=150e-6 * s_var, Cfix=1.2e-15,
                var_gate='node', vb=vb, charger='rail', vboot=7.0, Cw=CW, Rw=RW)

def job(pt):
    p = design_params(pt['drive'], pt['s_var'], pt['vb'], pt.get('cband', 0.0))
    try:
        r = vcro.tb(p, pt['T'] - 273.15, pt['vc'], tstop=2e-6)
    except Exception as e:
        r = dict(f=float('nan'), err=str(e)[-200:])
    return dict(pt, **r)

if __name__ == '__main__':
    pts = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    k, n = (int(sys.argv[3]), int(sys.argv[4])) if len(sys.argv) > 4 else (0, 1)
    pts = pts[k::n]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)['id'] for l in open(out)}
    todo = [p for p in pts if p['id'] not in done]
    with ProcessPoolExecutor(2) as ex, open(out, 'a') as fo:
        for r in ex.map(job, todo):
            fo.write(json.dumps(r) + '\n'); fo.flush()
