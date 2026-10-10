"""Assemble a transistor-level design file from the co-design result:
bias divider resistors, comparator divider ratios (from the calibrated trip voltage)."""
import sys, json, numpy as np
src, dst = sys.argv[1], sys.argv[2]
D = json.load(open(src)); d = dict(D['design'])
cal = json.load(open(RES + '/comp_cal2.json'))['ratio_at_T']      # static TL calibration
Tt = np.array(sorted(float(k) for k in cal)); Rt_ = np.array([cal[k] for k in sorted(cal, key=float)])
from netlib import pdk_local
import os as _os
RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results', 'tl')
def g2(T): return pdk_local.r2deg_factor(T)
def hom(T): return pdk_local.rom_factor(T)
R = 2e3; RT = 100e3
segs = []
for th in sorted(D['thresholds'], key=lambda t: -t['T_set']):     # segment k switches in below T_set
    T = th['T_set']; a = float(np.exp(np.interp(T, Tt, np.log(Rt_))))
    segs.append(dict(T_set=T, T_window=[th['T_lo'], th['T_hi']], Rt_om=RT, Rb_2deg=a * RT))
out = dict(design=d, bias=dict(Rtop_om=R, Rbot_2deg=d['a2'] * R, Rbot_om=d['ao'] * R),
           band=dict(segments=segs), f_cmp=D['f_cmp'], n_int=D['n_int'], source=src)
json.dump(out, open(dst, 'w'), indent=1)
print(json.dumps(segs, indent=0))
