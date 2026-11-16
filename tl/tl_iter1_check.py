"""(1) Manual-design oscillator stage in the open flow vs temperature (diode charger);
(2) iteration-1 co-designed sizing with the VDD-fed (diode) charger and with the 7 V rail.
Writes results/vcro_baseline_sweep.json and results/iter1_tl_check.json."""
import json
from concurrent.futures import ProcessPoolExecutor
import vcro
import os as _os
RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results', 'tl')

BASEP = dict(vcro.BASE, W_hu=6.5e-6, W_hd=32e-6, var_gate='node')
def it1(charger):
    D, S = 3.75, 5.91
    return dict(vcro.BASE, W_hu=6.5e-6 * D, W_hd=32e-6 * D, W_hs=26e-6 * D, W_ht=26e-6 * D, Cb=3.5e-12 * D,
                W_var=150e-6 * S, Cfix=1.2e-15, var_gate='node', vb=1.5, charger=charger,
                vboot=7.0 if charger == 'rail' else None, Cw=150e-15, Rw=2.9)

def job(a):
    tag, T, vc = a
    p = BASEP if tag == 'manual' else it1(tag)
    r = vcro.tb(p, T - 273.15, vc, tstop=3e-6)
    return dict(tag=tag, T=T, vc=vc, **r)

if __name__ == '__main__':
    jobs = [('manual', T, v) for T in (218, 273, 300, 398, 423) for v in (3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 5.0, 5.5)]
    jobs += [(c, T, v) for c in ('diode', 'rail') for T in (218, 300, 398, 423) for v in (4.0, 5.0, 5.8)]
    out = []
    with ProcessPoolExecutor(2) as ex:
        for r in ex.map(job, jobs):
            out.append(r); print(r['tag'], r['T'], r['vc'], round(r['f'] / 1e6, 2), round(r['swing'], 2), round(r['vgs_max'], 2), flush=True)
    json.dump([r for r in out if r['tag'] == 'manual'], open(RES + '/vcro_baseline_sweep.json', 'w'), indent=1)
    json.dump([r for r in out if r['tag'] != 'manual'], open(RES + '/iter1_tl_check.json', 'w'), indent=1)
