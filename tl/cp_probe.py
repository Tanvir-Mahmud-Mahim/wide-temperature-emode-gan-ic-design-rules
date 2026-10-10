"""Charge-pump probe: UP current into a forced control voltage, UP inactive (leakage) and
active (driver pulsed at 10 MHz, 50 % duty, as in the reference-gated start-up case)."""
import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
import netlib, pllnet
import os as _os
RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results', 'tl')

def run(args):
    T, vc, active, R_up = args
    tc = T - 273.15; k = 1.0      # same sizes as the closed-loop netlist
    net = netlib.Net('cp probe', tc)
    net.add('.options method=gear reltol=2e-3 abstol=1e-10 vntol=1e-5 itl4=100 gmin=1e-12 rshunt=1e9')
    net.add('vdd vdds 0 pwl(0 0 100n 6)'); net.add('rdds vdds vdd 0.5'); net.add('cdd vdd 0 100p')
    if active:
        net.add('vin upbs 0 pulse(6 0 300n 0.5n 0.5n 49.5n 100n)')
    else:
        net.add('vin upbs 0 dc 6')
    net.add('rin upbs upb 100')
    net.hemt('vdd', 'vdd', 'bup', 13e-6 * k); net.hemt('vdd', 'bup', 'updrv', 6.5e-6 * k)
    net.c('bup', 'updrv', 3.5e-12 * k); net.r2deg('vdd', 'updrv', 200e3)
    net.hemt('updrv', 'upb', '0', 26e-6 * k)
    net.hemt('vdd', 'bup', 'u1', 26e-6 * k)
    net.r2deg('u1', 'cp', R_up)
    net.add('vcp cp 0 dc %g' % vc)
    w = netlib.run(net.text(), ['bup', 'u1'], 2e-6, 0.2e-9, tmax=0.5e-9, save_extra=['i(vcp)'],
                   extra_ctl=['meas tran icp avg i(vcp) from=1.2e-6 to=2e-6'])
    return T, vc, active, -w['_meas']['icp'], float(w['bup'][w['t'] > 1.2e-6].min()), float(w['bup'][w['t'] > 1.2e-6].max())

if __name__ == '__main__':
    R_up = float(sys.argv[1]) if len(sys.argv) > 1 else 30e3
    jobs = [(T, vc, a, R_up) for T in (218, 300, 423) for a in (0, 1) for vc in (1.0, 2.5, 3.5, 4.5, 5.6)]
    out = []
    with ProcessPoolExecutor(2) as ex:
        for T, vc, a, i, bmin, bmax in ex.map(run, jobs):
            print('T %d Vc %.1f %s  I_up %.3f uA   boot [%.2f, %.2f]' % (T, vc, 'active  ' if a else 'inactive', i * 1e6, bmin, bmax), flush=True)
            out.append(dict(T=T, vc=vc, active=a, i_up_uA=-i * 1e6, boot_min=bmin, boot_max=bmax))
    import json
    json.dump(dict(R_up=R_up, rows=out, note='i_up_uA > 0: current into the filter node; active = driver pulsed at 10 MHz, 50 % duty'), open(RES + '/cp_probe.json', 'w'), indent=1)
