"""Stand-alone transistor-level MASH 1-1 (second-order) delta-sigma modulator built from the same
bootstrapped DCFL HEMT logic as the PLL (ngspice + MVSG), checked cycle by cycle against the
cycle-exact model (tl/frac_residual.mash2) and measured for supply power.

Architecture (B-bit accumulators, constant input word K):
  stage 1: S1 = A1 + K, carry c1;     stage 2: S2 = A2 + S1, carry c2;
  registers (six-NAND edge-triggered DFFs): A1 <- S1, A2 <- S2, C2P <- c2 on each clock edge;
  output word v = N - 4 = 1 + c1 + c2 - c2p  ->  p0 = XNOR(c1 xor c2, c2p),
  p1 = c1 c2 + (c1 xor c2) not(c2p);  N = 4 + 2 p1 + p0 (the divider's control word).
Full adders: 9 NAND gates each (two 4-NAND XORs, carry = NAND(NAND(a,b), NAND(a xor b, cin))).

usage: python3 mash_tl.py T_K [--bits 4] [--k 3] [--ncyc 48] [--size 0.75] [--fclk 10e6]
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlib, pllnet
from frac_residual import mash2

ap = argparse.ArgumentParser()
ap.add_argument('T', type=float); ap.add_argument('--bits', type=int, default=4)
ap.add_argument('--k', type=int, default=3); ap.add_argument('--ncyc', type=int, default=48)
ap.add_argument('--size', type=float, default=0.75); ap.add_argument('--fclk', type=float, default=10e6)
ap.add_argument('--tag', default='mash'); ap.add_argument('--tmax', type=float, default=2e-10); ap.add_argument('--debug', action='store_true')
a = ap.parse_args()
B, K = a.bits, a.k
VDD = 6.0

net = netlib.Net('MASH 1-1 %d-bit, K=%d, T=%g K' % (B, K, a.T), a.T - 273.15)
net.add('.options method=gear reltol=2e-3 abstol=1e-9 vntol=1e-5 itl4=100 gmin=1e-12 rshunt=1e9')
net.add('vdd vdd 0 dc 6'); net.add('vvboot vboot 0 dc 7'); net.add('vzero gnd0 0 dc 0')
Tc = 1.0 / a.fclk
net.add('vck ck 0 pulse(0 6 %g 1n 1n %g %g)' % (Tc / 2, Tc / 2 - 1e-9, Tc))
for i in range(B):
    net.add('vk%d k%d 0 dc %g' % (i, i, VDD if (K >> i) & 1 else 0.0))
lg = pllnet.Logic(net, k=a.size)

def xor(out, x, y, tag):
    n1 = tag + 'n1'
    lg.nand(n1, [x, y]); lg.nand(tag + 'n2', [x, n1]); lg.nand(tag + 'n3', [y, n1])
    lg.nand(out, [tag + 'n2', tag + 'n3'])
    return n1

def fa(s, co, x, y, ci, tag):
    n1 = xor(tag + 'x1', x, y, tag + 'a')
    m1 = xor(s, tag + 'x1', ci, tag + 'b')
    lg.nand(co, [n1, m1])

# stage 1: S1 = A1 + K
c = 'gnd0'
for i in range(B):
    fa('s1_%d' % i, 'c1_%d' % i, 'a1_%d' % i, 'k%d' % i, c, 'f1%d' % i); c = 'c1_%d' % i
c1 = c
# stage 2: S2 = A2 + S1
c = 'gnd0'
for i in range(B):
    fa('s2_%d' % i, 'c2_%d' % i, 'a2_%d' % i, 's1_%d' % i, c, 'f2%d' % i); c = 'c2_%d' % i
c2 = c
# registers
for i in range(B):
    lg.dff_d('ck', 's1_%d' % i, 'a1_%d' % i, 'a1b_%d' % i, 'r1%d' % i)
    lg.dff_d('ck', 's2_%d' % i, 'a2_%d' % i, 'a2b_%d' % i, 'r2%d' % i)
lg.dff_d('ck', c2, 'c2p', 'c2pb', 'r3')
# output word
n12 = xor('x12', c1, c2, 'w1')                    # n12 = NAND(c1, c2)
xor('px', 'x12', 'c2p', 'w2'); lg.inv('p0', 'px')  # p0 = XNOR(c1 xor c2, c2p)
lg.nand('p1n', ['x12', 'c2pb']); lg.nand('p1', [n12, 'p1n'])

n_hemt = sum(1 for l in net.lines if l.startswith('n'))
tstop = (a.ncyc + 0.5) * Tc
ic = {}
for i in range(B):                                 # registers start at zero
    ic.update({'a1_%d' % i: 0.0, 'a2_%d' % i: 0.0, 'a1b_%d' % i: VDD, 'a2b_%d' % i: VDD})
ic.update({'c2p': 0.0, 'c2pb': VDD})
t0 = time.time()
DBG = (['c1_3', 'c2_3', 'c2pb'] + ['a1_%d' % i for i in range(B)] + ['a2_%d' % i for i in range(B)] + ['s1_%d' % i for i in range(B)] + ['s2_%d' % i for i in range(B)]) if a.debug else []
w = netlib.run(net.text(), ['ck', 'p0', 'p1', 'c2p'] + DBG, tstop, 0.1e-9, tmax=a.tmax, ic=ic,
               save_extra=['i(vdd)', 'i(vvboot)'],
               extra_ctl=['meas tran iddavg avg i(vdd) from=%g to=%g' % (4 * Tc, tstop),
                          'meas tran ibtavg avg i(vvboot) from=%g to=%g' % (4 * Tc, tstop)])
t = w['t']
# word during cycle m (after m rising clock edges at Tc/2 + m*Tc), sampled just before the next edge
samp = [Tc / 2 + m * Tc - 3e-9 for m in range(1, a.ncyc)]
bit = lambda v, ts: int(np.interp(ts, t, v) > VDD / 2)
got = [4 + 2 * bit(w['p1'], ts) + bit(w['p0'], ts) for ts in samp]
exp, _ = mash2(K / 2 ** B, 5, a.ncyc, bits=B)
exp = exp[:len(got)]
# the modulator's first output (before any clock edge) is exp[0]; after m edges it is exp[m]
expm = [exp[m] if m < len(exp) else None for m in range(1, a.ncyc)]
ok = [g == e for g, e in zip(got, expm) if e is not None]
p_vdd = abs(w['_meas'].get('iddavg', np.nan)) * 6e3; p_bt = abs(w['_meas'].get('ibtavg', np.nan)) * 7e3
res = dict(tmax_ps=a.tmax * 1e12, T_K=a.T, bits=B, K=K, frac=K / 2 ** B, size=a.size, fclk_MHz=a.fclk / 1e6, n_hemt=n_hemt,
           ncyc=a.ncyc, n_checked=len(ok), n_match=int(sum(ok)), got=got, expected=expm,
           p_vdd_mW=p_vdd, p_boot_mW=p_bt, runtime_s=time.time() - t0)
RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'tl'); os.makedirs(RES, exist_ok=True)
json.dump(res, open(RES + '/%s_%d.json' % (a.tag, round(a.T)), 'w'))
print({k: v for k, v in res.items() if k not in ('got', 'expected')})
print('got     ', got); print('expected', expm)
if a.debug:
    for m, ts in enumerate(samp, 1):
        A1 = sum(bit(w['a1_%d' % i], ts) << i for i in range(B)); A2 = sum(bit(w['a2_%d' % i], ts) << i for i in range(B))
        S1 = sum(bit(w['s1_%d' % i], ts) << i for i in range(B)); S2 = sum(bit(w['s2_%d' % i], ts) << i for i in range(B))
        print(m, 'A1', A1, 'A2', A2, 'S1', S1, 'S2', S2, 'c1', bit(w['c1_3'], ts), 'c2', bit(w['c2_3'], ts), 'c2p', bit(w['c2p'], ts), 'c2pb', bit(w['c2pb'], ts))
