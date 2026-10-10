"""Transistor-level test of the two-cell 2/3 divider (N = 4 + 2 p1 + p0), integer or
MASH-sequenced (XSPICE d_state clocked by the divider output)."""
import sys, json, numpy as np
import netlib, pllnet

def build_div(net, k1, k2, chg, mode, nword=5, seq=None, kfb=None, ff='e'):
    lg1 = pllnet.Logic(net, k=k1, chg=chg); lg2 = pllnet.Logic(net, k=k2, chg=chg)
    if mode == 'int':
        w = nword - 4
        net.add('vp0 p0 0 dc %g' % (6.0 if w & 1 else 0.0)); net.add('vp1 p1 0 dc %g' % (6.0 if w & 2 else 0.0))
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
    cell = 'cell23e' if ff == 'e' else 'cell23'
    getattr(lg1, cell)('ck', ['p0', 'a2'], 'a1', 'xa')
    getattr(lg2, cell)('a1', ['p1'], 'a2', 'xb')
    lgf = pllnet.Logic(net, k=kfb or k2, chg=chg)
    lgf.inv('fbn', 'a2'); lgf.inv('fb', 'fbn')

def divider_test(T_K, fin, mode='int', nword=5, seq=None, k1=1.0, k2=0.5, chg='vboot', tstop=2e-6, kin=1.0, ff='e'):
    net = netlib.Net('div23 test', T_K - 273.15)
    net.add('.options method=gear reltol=2e-3 abstol=1e-9 vntol=1e-5 itl4=100 gmin=1e-12 rshunt=1e9')
    net.add('vdd vdds 0 pwl(0 0 100n 6)'); net.add('rdds vdds vdd 0.5'); net.add('cdd vdd 0 100p')
    net.add('vbt vboots 0 pwl(0 0 100n 7)'); net.add('rbts vboots vboot 0.5'); net.add('cbt vboot 0 100p')
    p = 1e9 / fin
    net.add('vck ck0s 0 pulse(0 6 150n 0.4n 0.4n %gn %gn)' % (p / 2 - 0.4, p)); net.add('rck ck0s ck0 100')
    lgi = pllnet.Logic(net, k=kin, chg=chg); lgi.inv('ckb', 'ck0'); lgi.inv('ck', 'ckb')
    build_div(net, k1, k2, chg, mode, nword, seq, ff=ff)
    w = netlib.run(net.text(), ['fb', 'ck'], tstop, 20e-12, tmax=20e-12, save_extra=['i(vdd)', 'i(vbt)'],
                   extra_ctl=['meas tran iddavg avg i(vdd) from=%g to=%g' % (tstop / 2, tstop),
                              'meas tran ibtavg avg i(vbt) from=%g to=%g' % (tstop / 2, tstop)])
    t, v = w['t'], w['fb']
    s = v > 3.0; r = np.where(s[1:] & ~s[:-1])[0]
    te = t[r]; per = np.diff(te[te > 0.5e-6]) * fin
    return dict(T=T_K, fin_MHz=fin / 1e6, mode=mode, ratios=np.round(per).astype(int).tolist(),
                mean_ratio=float(np.mean(per)) if len(per) else 0.0,
                vmin=float(v[t > 0.5e-6].min()), vmax=float(v[t > 0.5e-6].max()),
                p_mW=float(-6 * w['_meas'].get('iddavg', 0) * 1e3 - 7 * w['_meas'].get('ibtavg', 0) * 1e3))

if __name__ == '__main__':
    from concurrent.futures import ProcessPoolExecutor
    cfg = json.loads(sys.argv[1])
    def job(c):
        r = divider_test(**c)
        if c.get('mode') == 'frac':
            seq = c['seq']; rat = r['ratios']
            best = max(range(0, 40), key=lambda o: sum(a == b for a, b in zip(rat, seq[o:o + len(rat)])))
            r['match'] = '%d/%d' % (sum(a == b for a, b in zip(rat, seq[best:best + len(rat)])), len(rat))
            r['miss_idx'] = [i for i, (a, b) in enumerate(zip(rat, seq[best:best + len(rat)])) if a != b]
            r['ratios'] = rat[:12]
        return {k: v for k, v in r.items()}
    with ProcessPoolExecutor(2) as ex:
        for c, r in zip(cfg, ex.map(job, cfg)):
            print(json.dumps({k: v for k, v in c.items() if k != 'seq'}), '->', json.dumps(r if 'match' in r else dict(r, ratios=r['ratios'][:12])), flush=True)
