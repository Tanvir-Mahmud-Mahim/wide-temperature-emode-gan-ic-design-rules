"""Five-stage bootstrapped current-starved VCRO exactly as drawn (Fig. S7/F8).

Stage i:  Hd (diode-connected, d=g=VDD, s=boot_i), Hu (pull-up, d=VDD,
g=boot_i, s=o_i), Cb between boot_i and o_i, Hs (pull-down, d=o_i,
g=o_{i-1}, s=x_i), Ht (tail, d=x_i, g=Vc, s=0), delay branch R from o_i to
n_i into a HEMT varactor (to ground or bias rail vb), fixed cap o_i->0.
"""
import numpy as np
import netlib

BASE = dict(W_hu=32e-6, W_hd=6.5e-6, W_hs=26e-6, W_ht=26e-6, Cb=3.5e-12,
            Cfix=15e-15, Rdel=150.0, W_var=150e-6, var_gate='node', vb=0.0,
            Cw=0.0, Rw=0.0, W_k=0.0, charger='diode', vboot=None,
            Cband=0.0, Wsw_per_pF=40e-6)

def add_vcro(net, p, vc='vc', vdd='vdd', vbn='vbn', prefix=''):
    o = lambda i: '%so%d' % (prefix, i % 5)
    for i in range(5):
        b, x, n = '%sb%d' % (prefix, i), '%sx%d' % (prefix, i), '%sn%d' % (prefix, i)
        inp = o(i - 1) if p['Rw'] == 0 else '%sg%d' % (prefix, i)
        if p.get('charger', 'diode') == 'diode':
            net.hemt(vdd, vdd, b, p['W_hd'])
        elif p['charger'] == 'rail':     # boosted precharge rail for the bootstrap charger only
            net.hemt('vboot', 'vboot', b, p['W_hd'])
        elif p['charger'] == 'gate_rail':  # charger gate on a 7 V bias rail, drain on VDD (rail carries ~no current)
            net.hemt(vdd, 'vboot', b, p['W_hd'])
        else:                            # self-timed: charger gate on the previous stage's boot node
            net.hemt(vdd, '%sb%d' % (prefix, (i - 1) % 5), b, p['W_hd'])
        net.hemt(vdd, b, o(i), p['W_hu'])
        net.c(b, o(i), p['Cb'])
        if p.get('W_k', 0) > 0:          # weak diode-connected keeper (static start-up path)
            net.hemt(vdd, vdd, o(i), p['W_k'])
        net.hemt(o(i), inp, x, p['W_hs'])
        net.hemt(x, vc, '0', p['W_ht'])
        net.r2deg(o(i), n, p['Rdel'])
        if p['var_gate'] == 'node':      # varactor gate on the delay node, S/D on bias rail
            net.varactor(n, vbn, p['W_var'])
        else:                            # varactor gate on bias rail, S/D on delay node
            net.varactor(vbn, n, p['W_var'])
        if p['Cfix'] > 0:
            net.c(o(i), '0', p['Cfix'])
        if p.get('Cband', 0) > 0:        # coarse band: MIM capacitor switched to ground by a HEMT
            cbn = '%scbk%d' % (prefix, i)
            net.c(o(i), cbn, p['Cband'])
            net.hemt(cbn, p.get('band_node', 'band'), '0', p['Wsw_per_pF'] * p['Cband'] * 1e12)
        if p['Rw'] > 0:                  # extracted inter-stage wire (pi section)
            j = (i + 1) % 5
            net.c(o(i), '0', p['Cw'] / 2)
            net.r(o(i), '%sg%d' % (prefix, j), p['Rw'])
            net.c('%sg%d' % (prefix, j), '0', p['Cw'] / 2)

def tb(p, tc, vcval, tstop=1.5e-6, tstep=20e-12):
    net = netlib.Net('VCRO T=%gC Vc=%g' % (tc, vcval), tc)
    net.add('vdd vdd 0 dc 6')
    net.add('vvc vc 0 dc %g' % vcval)
    net.add('vvb vbn 0 dc %g' % p['vb'])
    if p.get('vboot'):
        net.add('vvboot vboot 0 dc %g' % p['vboot'])
    if p.get('Cband', 0) > 0:
        net.add('vband band 0 dc %g' % p.get('band_v', 6.0))
    add_vcro(net, p)
    extra = ['let ibt = -i(vvboot)', 'let idd = -i(vdd)'] if p.get('vboot') else ['let idd = -i(vdd)']
    w = netlib.run(net.text(), ['o0', 'b0', 'o1', 'b4'], tstop, tstep,
                   ic={'o0': 6, 'o1': 0}, save_extra=['i(vdd)'] + (['i(vvboot)'] if p.get('vboot') else []), extra_ctl=['meas tran iddavg avg i(vdd) from=%g to=%g' % (tstop*0.5, tstop)] + (['meas tran ibtavg avg i(vvboot) from=%g to=%g' % (tstop*0.5, tstop)] if p.get('vboot') else []))
    f, sw = netlib.freq(w['t'], w['o0'], tstop * 0.5)
    m = w['t'] >= tstop * 0.5
    return dict(f=f, swing=sw, vmin=float(w['o0'][m].min()), vmax=float(w['o0'][m].max()),
                bmax=float(w['b0'][m].max()), vgs_max=float((w['b0'] - w['o0'])[m].max()),
                vgs_chg=float((w['b4'] - w['b0'])[m].max()),
                idd=-w['_meas'].get('iddavg', float('nan')), iboot=-w['_meas'].get('ibtavg', float('nan')))

if __name__ == '__main__':
    import sys, itertools, json, time
    tc = 27.0
    variants = {
        'I_node': dict(BASE),
        'I_rail': dict(BASE, var_gate='rail'),
        'II_node': dict(BASE, W_hu=6.5e-6, W_hd=32e-6),
        'II_rail': dict(BASE, W_hu=6.5e-6, W_hd=32e-6, var_gate='rail'),
    }
    for name, p in variants.items():
        for vc in (3.0, 3.5, 4.0, 4.5, 5.5):
            t0 = time.time()
            try:
                r = tb(p, tc, vc)
                print(f"{name:8s} Vc={vc:.1f} f={r['f']/1e6:7.2f} MHz swing={r['swing']:.2f} "
                      f"[{r['vmin']:.2f},{r['vmax']:.2f}] Vgs_up,max={r['vgs_max']:.2f} ({time.time()-t0:.0f}s)", flush=True)
            except Exception as e:
                print(name, vc, 'ERR', str(e)[-300:])
