"""Transistor-level closed-loop netlist of the co-designed GaN PLL (ngspice-46, OSDI MVSG).

Every analog and logic block is built from PDK LV e-mode HEMTs (MVSG) and PDK
resistor models (2DEG with its temperature coefficients; ohmic metal):
  * bootstrapped DCFL logic (charger from the 7 V precharge rail, bootstrapped
    pull-up, bootstrap capacitor, series pull-down network);
  * PFD: two 6-NAND edge-triggered flip-flops + reset NAND (13 gates);
  * charge pump: bootstrapped high-side HEMT switch + 2DEG resistor (UP),
    2DEG resistor + HEMT switch (DOWN);
  * loop filter: damping resistor, HEMT varactor C2, Rs1-Cs1, Rs23, HEMT varactor C3;
  * VCRO: five bootstrapped current-starved stages with HEMT-varactor delay
    branches on the tracked bias rail;
  * tracked bias generator: resistive divider, ohmic-metal top leg and 2DEG (+ ohmic)
    bottom leg (different PDK temperature coefficients) with a decoupling capacitor;
  * coarse band select: thermometer-coded MIM capacitor segments switched to ground
    by HEMTs, each enabled below its threshold temperature by a comparator made of an
    ohmic/2DEG divider and three DCFL inverters;
  * feedback: 3-bit synchronous loadable down-counter (moduli 2..7) with a
    registered terminal-count output; in fractional mode its load word is
    driven by a cycle-exact MASH-2 sequence (XSPICE d_state, clocked by the
    divider output).
"""
import numpy as np
import netlib, vcro

VDD, VBOOT = 6.0, 7.0

def rom(r25, tc):
    return netlib.rom(r25, tc)

class Logic:
    def __init__(self, net, k=1.0, cb=3.5e-12, r_keep=200e3, kpu=1.0, chg='vboot'):
        # kpu scales the bootstrapped pull-up and its capacitor (static current when the
        # output is low flows through the pull-up, so kpu sets the static power)
        self.net, self.k, self.cb, self.g, self.r_keep, self.kpu, self.chg = net, k, cb, 0, r_keep, kpu, chg
    def _gid(self):
        g = getattr(self.net, '_gid', 0); self.net._gid = g + 1
        return g
    def nand(self, out, ins, k=None):
        k = k or self.k; n = self.net; g = self._gid()
        b = 'lb%d' % g
        n.hemt(self.chg, self.chg, b, 13e-6 * k * self.kpu)
        n.hemt('vdd', b, out, 6.5e-6 * k * self.kpu)
        n.c(b, out, self.cb * k * self.kpu)
        if self.r_keep:                       # weak static keeper (2DEG resistor): holds a full static
            n.r2deg('vdd', out, self.r_keep)  # high level between transitions at hot corners
        wpd = 26e-6 * k * (1.0 if len(ins) <= 2 else 1.5)
        top = out
        for i, a in enumerate(ins):
            bot = '0' if i == len(ins) - 1 else 'ls%d_%d' % (g, i)
            n.hemt(top, a, bot, wpd)
            top = bot
        return b
    def inv(self, out, a, k=None):
        return self.nand(out, [a], k)
    def nor(self, out, ins, k=None):
        """Bootstrapped DCFL NOR: parallel pull-downs (no series stack)."""
        k = k or self.k; n = self.net; g = self._gid()
        b = 'lb%d' % g
        n.hemt(self.chg, self.chg, b, 13e-6 * k * self.kpu)
        n.hemt('vdd', b, out, 6.5e-6 * k * self.kpu)
        n.c(b, out, self.cb * k * self.kpu)
        if self.r_keep:
            n.r2deg('vdd', out, self.r_keep)
        for a in ins:
            n.hemt(out, a, '0', 26e-6 * k)
        return b
    def clk2(self, ck, phi1, phi2, tag='cg'):
        """Two-phase non-overlapping clocks: phi1 high while ck low (master),
        phi2 high while ck high (slave); each rises only after the other has fallen."""
        self.inv(tag + 'b', ck)
        self.nor(tag + '1', [ck, tag + '2d'], k=1.5 * self.k)
        self.nor(tag + '2', [tag + 'b', tag + '1d'], k=1.5 * self.k)
        self.inv(tag + '1x', tag + '1'); self.inv(tag + '1d', tag + '1x')
        self.inv(tag + '2x', tag + '2'); self.inv(tag + '2d', tag + '2x')
        self.inv(phi1 + 'n', tag + '1'); self.inv(phi1, phi1 + 'n', k=2 * self.k)
        self.inv(phi2 + 'n', tag + '2'); self.inv(phi2, phi2 + 'n', k=2 * self.k)
    def dff_edge(self, clk, q, rb, tag):            # 6-NAND positive-edge DFF, D = 1, async reset
        n1, n2, n3, n4, qb = [tag + s for s in ('n1', 'n2', 'n3', 'n4', 'qb')]
        self.nand(n1, [n4, n2]); self.nand(n2, [n1, clk, rb]); self.nand(n3, [n2, clk, n4])
        self.nand(n4, [n3, rb]); self.nand(q, [n2, qb]); self.nand(qb, [q, n3, rb])
    def pfd(self, ref, fb, qr, qf):
        self.dff_edge(ref, qr, 'rstb', 'pr'); self.dff_edge(fb, qf, 'rstb', 'pf')
        self.nand('rstb', [qr, qf])
    def dlatch(self, q, qb, d, en, tag):
        a, b = tag + 'a', tag + 'b'
        # enable transistor on top of each stack (next to the output): a changing data
        # input cannot charge-share with the output while the latch is closed
        self.nand(a, [en, d]); self.nand(b, [en, a]); self.nand(q, [a, qb]); self.nand(qb, [q, b])
    def msdff(self, d, ck, ckb, q, qb, tag):        # master on ckb, slave on ck: positive edge
        # NOTE: ngspice node names are case-insensitive; keep internal names distinct
        self.dlatch(tag + '_mq', tag + '_mqb', d, ckb, tag + '_li')
        self.dlatch(q, qb, tag + '_mq', ck, tag + '_ls')
    def xnor(self, out, a, b, tag):
        self.nand(tag + '1', [a, b]); self.nand(tag + '2', [a, tag + '1']); self.nand(tag + '3', [b, tag + '1'])
        self.nand(tag + 'x', [tag + '2', tag + '3']); self.inv(out, tag + 'x')
    def counter(self, ck, L, fb):
        """3-bit loadable down-counter: states L..0, reload at terminal count; fb = registered TC."""
        self.clk2(ck, 'dck_b', 'dck')          # dck_b = master enable, dck = slave enable (non-overlapping)
        Q = ['q0', 'q1', 'q2']; QB = ['q0b', 'q1b', 'q2b']
        self.nand('tcn', QB, k=1.5 * self.k); self.inv('tc', 'tcn')               # TC = all zero
        self.xnor('b1', 'q1', 'q0', 'xa')                                       # bit1 of S-1
        self.nand('nb2a', ['q1b', 'q0b']); self.inv('bor', 'nb2a')              # borrow into bit2
        self.xnor('b2n', 'q2', 'bor', 'xb'); self.inv('b2', 'b2n')              # bit2 of S-1 = q2 xor borrow
        dec = ['q0b', 'b1', 'b2']
        for i in range(3):
            self.nand('ma%d' % i, ['tc', L[i]]); self.nand('mb%d' % i, ['tcn', dec[i]])
            self.nand('dn%d' % i, ['ma%d' % i, 'mb%d' % i])
            self.msdff('dn%d' % i, 'dck', 'dck_b', Q[i], QB[i], 'f%d' % i)
        self.msdff('tc', 'dck', 'dck_b', fb, fb + 'b', 'fo')
    def dff_d(self, clk, d, q, qb, tag, k=None):
        """Six-NAND positive-edge-triggered D flip-flop (single clock, no non-overlap needed)."""
        n1, n2, n3, n4 = [tag + s for s in ('n1', 'n2', 'n3', 'n4')]
        self.nand(n1, [n4, n2], k); self.nand(n2, [n1, clk], k); self.nand(n3, [n2, clk, n4], k)
        self.nand(n4, [n3, d], k); self.nand(q, [n2, qb], k); self.nand(qb, [q, n3], k)
    def cell23e(self, ck, M, A, tag):
        """Divide-by-2/3 cell with six-NAND edge-triggered flip-flops:
        A+ = NOR(A, B), B+ = A AND M...  (/3 when all M high, else /2)."""
        B = tag + 'bq'
        self.nor(tag + 'dA', [A, B])
        self.nand(tag + 'nB', [A] + list(M)); self.inv(tag + 'dB', tag + 'nB')
        self.dff_d(ck, tag + 'dA', A, A + 'x', tag + 'fa')
        self.dff_d(ck, tag + 'dB', B, B + 'x', tag + 'fb')
    def cell23(self, ck, M, A, tag):
        """Divide-by-2/3 cell (two positive-edge flip-flops A, B clocked by ck):
        A+ = NOR(A, B),  B+ = A AND M1 AND M2 ...  ->  /3 when all M high, else /2.
        Output A is high for one ck period per output cycle."""
        self.clk2(ck, tag + 'm', tag + 's', tag=tag + 'c')     # m: master enable (ck low), s: slave enable
        B = tag + 'bq'
        self.nor(tag + 'dA', [A, B])
        self.nand(tag + 'nB', [A] + list(M)); self.inv(tag + 'dB', tag + 'nB')
        self.msdff(tag + 'dA', tag + 's', tag + 'm', A, A + 'b', tag + 'fa')
        self.msdff(tag + 'dB', tag + 's', tag + 'm', B, B + 'b', tag + 'fb')

def divider23(net, k1, k2, mode='int', nword=5, seq=None, chg='vboot'):
    """Two cascaded divide-by-2/3 cells (six-NAND edge-triggered flip-flops):
    N = 4 + 2 p1 + p0 in 4..7.  Cell 1 (input 'ck') swallows one input cycle per output
    cycle when p0 and cell 2's state bit a2 are high; cell 2 divides by 2 + p1.
    In fractional mode the word (p0, p1) is a cycle-exact MASH-2 sequence advanced on
    each divider output edge (XSPICE d_state + bridges; the modulator is not built from
    HEMTs).  Output 'fb'."""
    lg1 = Logic(net, k=k1, chg=chg); lg2 = Logic(net, k=k2, chg=chg)
    if mode == 'int':
        w = nword - 4
        net.add('vp0 p0 0 dc %g' % (VDD if w & 1 else 0.0)); net.add('vp1 p1 0 dc %g' % (VDD if w & 2 else 0.0))
    else:
        sfile = '%s/mash23_%d.state' % (netlib.SCRATCH, abs(hash(tuple(seq[:64]))) % 10 ** 8)
        with open(sfile, 'w') as fh:
            for st, m in enumerate(seq):
                w = m - 4; bits = ' '.join('1s' if (w >> i) & 1 else '0s' for i in range(2))
                for clkv in ('0', '1'):
                    fh.write('%d %s %s -> %d\n' % (st, bits, clkv, (st + 1) % len(seq)))
        net.add('aadc [fb] [fbd] adcb'); net.add('.model adcb adc_bridge(in_low=2.0 in_high=4.0)')
        net.add('vzero zz 0 dc 0'); net.add('azz [zz] [zzd] adcb')
        net.add('ast [zzd] fbd zzd [ld0 ld1] dst')
        net.add('.model dst d_state(clk_delay=1e-10 reset_delay=1e-10 state_file="%s" reset_state=0)' % sfile)
        net.add('adac [ld0 ld1] [p0r p1r] dacb')
        for i in range(2):
            net.add('rp%d p%dr p%d 1k' % (i, i, i)); net.add('cp%d p%d 0 50f' % (i, i))
        net.add('.model dacb dac_bridge(out_low=0 out_high=6 t_rise=0.5e-9 t_fall=0.5e-9)')
    lg1.cell23e('ck', ['p0', 'a2'], 'a1', 'xa')
    lg2.cell23e('a1', ['p1'], 'a2', 'xb')
    lg2.inv('fbn', 'a2'); lg2.inv('fb', 'fbn')

def build(T_K, d, mode='int', n_int=5, mash_seq=None, f_cmp=10e6, tstop=60e-6,
          bias=None, vb_ideal=None, logic_k=0.5, ic_vc=0.0, vto_shift=0.0, tmax=100e-12,
          band=None, cw_scale=1.0, logic_kpu=1.0, k_div1=0.75, k_div2=0.5, k_cmp=0.25, vto_var=0.0,
          por=None):
    tc = T_K - 273.15
    net = netlib.Net('closed-loop GaN PLL T=%.1fK mode=%s' % (T_K, mode), tc)
    over = {} if vto_shift == 0 else {'vto': float(netlib.BASE['vto']) + vto_shift}
    if over:   # apply a threshold shift to every HEMT (process-spread corner)
        hemt0, var0 = net.hemt, net.varactor
        net.hemt = lambda dd, g, s, w, **o: hemt0(dd, g, s, w, **dict(over, **o))
        net.varactor = lambda a, b, w, **o: var0(a, b, w, **dict(over, **o))
    if vto_var:  # shift of the varactor C-V curve only (trapping / bias-temperature drift)
        var1 = net.varactor
        net.varactor = lambda a, b, w, **o: var1(a, b, w, **dict({'vto': float(netlib.BASE['vto']) + vto_shift + vto_var}, **o))
    net.add('.options method=gear reltol=2e-3 abstol=1e-9 vntol=1e-5 itl4=100 gmin=1e-12 rshunt=1e9')
    net.add('vdd vdds 0 pwl(0 0 100n %g)' % VDD); net.add('rdds vdds vdd 0.5'); net.add('cdd vdd 0 100p'); net.add('vbt vboots 0 pwl(0 0 100n %g)' % VBOOT); net.add('rbts vboots vboot 0.5'); net.add('cbt vboot 0 100p')   # supply ramp (power-on)
    p = 1.0 / f_cmp
    net.add('vref refs 0 pulse(0 6 200n 0.5n 0.5n %gn %gn)' % (p / 2 * 1e9, p * 1e9)); net.add('rrefs refs ref 100')
    lg = Logic(net, k=logic_k, kpu=logic_kpu)
    # ---- PFD + charge pump
    lg.pfd('ref', 'fb', 'qr', 'qf')
    # bootstrapped UP driver, gated by the reference: a held-high UP (oscillator not yet
    # running) still pulses at the reference rate, so the driver's bootstrap is refreshed
    # every cycle and the high-side switch keeps its gate overdrive (no start-up deadlock)
    lg.nand('upb', ['qr', 'ref'])
    # UP driver: bootstrapped inverter whose charger is fed from VDD (not the 7 V rail), so
    # the boot node idles near VDD - Vth: the high-side switch is off for Vc above ~3 V while
    # UP is inactive, and is driven ~5 V above its source while UP is active
    kk = 1.0
    net.hemt('vdd', 'vdd', 'bup', 13e-6 * kk); net.hemt('vdd', 'bup', 'updrv', 6.5e-6 * kk)
    net.c('bup', 'updrv', 3.5e-12 * kk); net.r2deg('vdd', 'updrv', 200e3)
    net.hemt('updrv', 'upb', '0', 26e-6 * kk)
    net.hemt('vdd', 'bup', 'u1', 26e-6 * kk)                     # high-side switch, gate on boot node
    net.r2deg('u1', 'cp', d['R_up'])
    net.r2deg('cp', 'd1', d['R_dn'])
    net.hemt('d1', 'qf', '0', 26e-6)
    # ---- third-order loop filter (drawn topology)
    # (filter node names are distinct from the VCRO's internal x<i>/n<i> nodes; ngspice
    #  node names are global and case-insensitive)
    net.r2deg('cp', 'lfx', 1e3)
    net.varactor('lfx', '0', d['W_c2'] * 1e-6)
    net.r2deg('lfx', 'lfs1', d['Rs1']); net.c('lfs1', '0', d['Cs1'])
    net.r2deg('lfx', 'vc', d['Rs23'])
    net.varactor('vc', '0', d['W_c3'] * 1e-6)
    # ---- optional power-on acquisition aid: a source follower from VDD into the filter whose
    # gate is a capacitively coupled copy of the 7-V precharge rail that decays through a 2DEG
    # resistor. During the first microseconds after power-up it lifts the filter to about
    # (V_por - V_th); it then turns off for good and draws no static current.
    if por:
        net.c('vboot', 'por', por.get('C', 10e-12))
        net.r2deg('por', '0', por.get('R', 1e6))
        net.hemt('vdd', 'por', 'porx', por.get('W', 6.5e-6))
        net.r2deg('porx', 'lfx', por.get('Rs', 5e3))
    # ---- tracked varactor bias rail: ohmic-metal top leg, 2DEG + ohmic-metal bottom leg
    if vb_ideal is not None:
        net.add('vvb vbn 0 dc %g' % vb_ideal)
    else:
        net.r('vdd', 'vbn', rom(bias['Rtop_om'], tc))
        net.r2deg('vbn', 'vbm', bias['Rbot_2deg'])
        if bias.get('Rbot_om', 0) > 0:
            net.r('vbm', '0', rom(bias['Rbot_om'], tc))
        else:
            net.add('rzb vbm 0 1e-3')
        net.c('vbn', '0', bias.get('Cdec', 50e-12))
    # ---- coarse band select: thermometer-coded switched MIM capacitors; each segment is
    # enabled by a temperature comparator (ohmic/2DEG divider into three DCFL inverters)
    nseg = len(band.get('segments', [])) if band else 0
    for k in range(nseg):
        sg = band['segments'][k]
        if 'force' in sg:                          # forced band bit (diagnostics only)
            net.add('vbf%d bnd%d 0 dc %g' % (k, k, VDD if sg['force'] else 0.0))
        else:
            net.r('vdd', 'tq%d' % k, rom(sg['Rt_om'], tc))
            net.r2deg('tq%d' % k, '0', sg['Rb_2deg'])
            # comparator: three resistor-load (RTL) inverters -- static, no bootstrap memory,
            # so the decision depends only on the divider tap and the temperature
            for a_, b_ in (('tq', 'tqa'), ('tqa', 'tqb'), ('tqb', 'bnd')):
                net.r2deg('vdd', '%s%d' % (b_, k), band.get('r_load', 200e3))
                net.hemt('%s%d' % (b_, k), '%s%d' % (a_, k), '0', band.get('w_cmp', 6.5e-6))
            net.c('tq%d' % k, '0', 2e-12)
    # ---- VCRO
    P = dict(vcro.BASE, W_hu=6.5e-6 * d['drive'], W_hd=32e-6 * d['drive'], W_hs=26e-6 * d['drive'],
             W_ht=26e-6 * d['drive'], Cb=3.5e-12 * d['drive'], W_var=150e-6 * d['s_var'],
             Cfix=1.2e-15, var_gate='node', charger='rail', Cw=cw_scale * 150e-15, Rw=2.9)
    vcro.add_vcro(net, P, vc='vc', vdd='vdd', vbn='vbn')
    for k in range(nseg):                          # band segment k on every stage
        for i in range(5):
            cbn = 'bk%d_%d' % (k, i)
            net.c('o%d' % i, cbn, d['C_u'] * 1e-12)
            net.hemt(cbn, 'bnd%d' % k, '0', 40e-6 * d['C_u'])
    # ---- buffer + divider: two cascaded 2/3 cells, N = 4 + 2 p1 + p0
    lgb = Logic(net, k=k_div1, kpu=logic_kpu)
    lgb.inv('ckb', 'o0'); lgb.inv('ck', 'ckb')
    divider23(net, k_div1, k_div2, mode, n_int, mash_seq)
    ic = {'lfx': ic_vc, 'vc': ic_vc, 'cp': ic_vc, 'lfs1': ic_vc, 'o0': 6, 'o1': 0}
    return net, ic

def mash2(frac, n_int, length, bits=20):
    """MASH 1-1 (second-order) division-ratio sequence for N = n_int + frac."""
    M = 1 << bits; k = int(round(frac * M)); a1 = a2 = 0; c2p = 0; out = []
    for _ in range(length):
        a1 += k; c1 = a1 >> bits; a1 &= M - 1
        a2 += a1; c2 = a2 >> bits; a2 &= M - 1
        out.append(n_int + c1 + c2 - c2p); c2p = c2
    return out, k / M
