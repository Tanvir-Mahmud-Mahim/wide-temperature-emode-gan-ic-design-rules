"""Static calibration of the band comparators with the real divider (ohmic top 100 kOhm,
2DEG bottom): bisection on the resistor ratio so that the band bit flips at T_set.
Also maps the trip temperature vs ratio.  usage: python3 comp_cal2.py T1 T2 ..."""
import sys, json, numpy as np
from concurrent.futures import ProcessPoolExecutor
import comp_probe
import os as _os
RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results', 'tl')

def bit(T, r):
    return comp_probe.run((T, r, 100e3))[3] > 3.0

def solve(T):
    lo, hi = 0.2, 2.0           # bit high (cold side) for small ratio, low for large ratio
    for _ in range(14):
        mid = 0.5 * (lo + hi)
        if bit(T, mid): lo = mid
        else: hi = mid
    return T, 0.5 * (lo + hi)

if __name__ == '__main__':
    Ts = [float(x) for x in sys.argv[1:]]
    with ProcessPoolExecutor(2) as ex:
        res = dict(ex.map(solve, Ts))
    print(json.dumps(res))
    json.dump(dict(ratio_at_T=res, method='static, divider 100 kOhm ohmic top / 2DEG bottom, three RTL inverters (200 kOhm 2DEG load, 6.5 um)'),
              open(RES + '/comp_cal2.json', 'w'), indent=1)
