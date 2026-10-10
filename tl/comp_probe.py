"""Transistor-level probe of the band comparator: ohmic-metal top / 2DEG bottom divider
into three resistor-load (RTL) inverters.  Prints the static output (V) vs temperature."""
import sys, numpy as np
from concurrent.futures import ProcessPoolExecutor
import netlib, pllnet

def run(args):
    T, ratio, rt = args[:3]; state = args[3] if len(args) > 3 else None; rh = args[4] if len(args) > 4 else 2e6
    tc = T - 273.15
    net = netlib.Net('comparator', tc)
    net.add('.options method=gear reltol=2e-3 abstol=1e-9 vntol=1e-5 itl4=100 gmin=1e-12 rshunt=1e9')
    net.add('vdd vdds 0 pwl(0 0 100n 6)'); net.add('rdds vdds vdd 0.5'); net.add('cdd vdd 0 100p')
    net.add('vbt vboots 0 pwl(0 0 100n 7)'); net.add('rbts vboots vboot 0.5'); net.add('cbt vboot 0 100p')
    net.r('vdd', 'tq', pllnet.rom(rt, tc)); net.r2deg('tq', '0', ratio * rt)
    for a_, b_ in (('tq', 'tqa'), ('tqa', 'tqb'), ('tqb', 'bnd')):
        net.r2deg('vdd', b_, 200e3); net.hemt(b_, a_, '0', 6.5e-6)
    net.c('tq', '0', 2e-12)
    if state == 'hot':      # start in the hot state: a HEMT pulls the tap high for 1 us, then opens
        net.add('vkick kg 0 pwl(0 0 150n 0 160n 7 1u 7 1.01u 0)')
        net.hemt('vdd', 'kg', 'tq', 10e-6)
    w = netlib.run(net.text(), ['tq', 'bnd'], 3e-6, 1e-9, tmax=2e-9)
    return T, ratio, float(w['tq'][-1]), float(w['bnd'][-1])

if __name__ == '__main__':
    rt = 100e3
    ratios = [float(x) for x in sys.argv[1:]] or [0.5, 0.8, 1.2]
    jobs = [(T, r, rt) for r in ratios for T in (218, 253, 300, 348, 398, 423)]
    with ProcessPoolExecutor(2) as ex:
        for T, r, vq, vb in ex.map(run, jobs):
            print('ratio %.2f T %d  v_div %.3f  band_bit %.2f' % (r, T, vq, vb), flush=True)
