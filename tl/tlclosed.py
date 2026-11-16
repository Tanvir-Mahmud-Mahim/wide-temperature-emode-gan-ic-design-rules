"""Closed-loop transistor-level verification of the co-designed GaN PLL.

usage: python3 tlclosed.py design.json T_K mode [n_int frac] [--vto dV] [--cw k] [--tag name]
Writes results/tl/<tag>_<T>_<mode>.json with the lock metrics and a decimated waveform.
"""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import netlib, pllnet
import os as _os
RES = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), 'results', 'tl')

def edges(t, v, th=3.0):
    s = v > th
    i = np.where(s[1:] & ~s[:-1])[0]
    return t[i] + (th - v[i]) / (v[i + 1] - v[i]) * (t[i + 1] - t[i])

def analyse(w, f_cmp, n_div, t_end):
    t = w['t']
    er = edges(t, w['ref']); ef = edges(t, w['fb']); eo = edges(t, w['ck'])
    out = {}
    # output frequency trace from buffered-oscillator edges
    if len(eo) > 10:
        fo = 1.0 / np.diff(eo); to = eo[1:]
    else:
        fo = np.array([0.0]); to = np.array([t_end])
    f_tgt = n_div * f_cmp
    tail = to > 0.8 * t_end
    et = eo[eo > 0.8 * t_end]
    f_avg = (len(et) - 1) / (et[-1] - et[0]) if len(et) > 2 else 0.0   # edge count / time
    out['f_tail_MHz'] = float(f_avg / 1e6)
    out['f_err_ppm'] = float((f_avg - f_tgt) / f_tgt * 1e6) if f_avg > 0 else -1e6
    out['f_inst_pp_pct'] = float((fo[tail].max() - fo[tail].min()) / f_tgt * 100) if tail.any() else 0.0
    # settling: last time the reference-cycle-averaged frequency leaves the 1 % band
    if len(eo) > 10:
        win = max(int(round(n_div)), 1) * 4
        fa = np.convolve(fo, np.ones(win) / win, mode='same')
        bad = np.where(np.abs(fa - f_tgt) / f_tgt > 0.01)[0]
        out['settle_us'] = float(to[bad[-1]] * 1e6) if len(bad) else float(to[0] * 1e6)
    # phase: pair each late reference edge with the nearest feedback edge
    re_ = er[er > 0.8 * t_end]
    if len(re_) > 4 and len(ef) > 4:
        idx = np.searchsorted(ef, re_)
        idx = np.clip(idx, 1, len(ef) - 1)
        cand = np.stack([ef[idx - 1], ef[idx]], 1)
        dt = cand - re_[:, None]
        k = np.argmin(np.abs(dt), 1)
        d = dt[np.arange(len(re_)), k]
        ph = d * f_cmp * 360.0
        out.update(phase_mean_deg=float(np.mean(ph)), phase_std_deg=float(np.std(ph)),
                   phase_pp_deg=float(np.ptp(ph)), tie_rms_ps=float(np.std(d) * 1e12),
                   n_ref_edges=int(len(re_)))
        # count of feedback edges per reference edge in the tail (cycle slips -> != 1)
        h = 0.5 / f_cmp                      # half a reference period: robust when the phase is near 0
        nf = np.sum((ef > re_[0] - h) & (ef <= re_[-1] - h))
        out['fb_per_ref'] = float(nf / (len(re_) - 1))
    # phase error at every reference edge over the whole run (for settling plots)
    if len(er) > 4 and len(ef) > 4:
        idx = np.clip(np.searchsorted(ef, er), 1, len(ef) - 1)
        cand = np.stack([ef[idx - 1], ef[idx]], 1); dt_ = cand - er[:, None]
        d_ = dt_[np.arange(len(er)), np.argmin(np.abs(dt_), 1)]
        out['phase_series'] = dict(t=er.tolist(), deg=(d_ * f_cmp * 360.0).tolist())
    m = t > 0.8 * t_end
    out['vc_tail'] = float(np.mean(w['vc'][m])); out['vc_pp_mV'] = float(np.ptp(w['vc'][m]) * 1e3)
    out['vb_tail'] = float(np.mean(w['vbn'][m]))
    out['vgs_pu_max'] = float(np.max((w['b0'] - w['o0'])[m]))
    out['vboot_node_max'] = float(np.max(w['b0'][m]))
    for k in [k for k in w if k.startswith('bnd')]:
        out[k] = float(np.mean(w[k][m]))
    # phase lock: one feedback edge per reference edge (no cycle slip) and a bounded phase
    # over the last 20 % of the run
    out['lock'] = bool(abs(out.get('fb_per_ref', 0) - 1) < 1e-6 and out.get('phase_pp_deg', 999) < 30)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('design'); ap.add_argument('T', type=float); ap.add_argument('mode')
    ap.add_argument('--nint', type=int, default=5); ap.add_argument('--frac', type=float, default=0.0)
    ap.add_argument('--vto', type=float, default=0.0); ap.add_argument('--vtovar', type=float, default=0.0); ap.add_argument('--cw', type=float, default=1.0)
    ap.add_argument('--tstop', type=float, default=30e-6); ap.add_argument('--tag', default='tl')
    ap.add_argument('--por', action='store_true', help='enable the power-on acquisition aid')
    ap.add_argument('--icvc', type=float, default=0.0); ap.add_argument('--tmax', type=float, default=50e-12)
    ap.add_argument('--force_band', default='')
    a = ap.parse_args()
    D = json.load(open(a.design))
    d, bias, band = D['design'], D['bias'], D.get('band')
    if a.force_band:
        band = dict(segments=[dict(force=int(c)) for c in a.force_band])
    f_cmp = D.get('f_cmp', 10e6)
    seq = None; n_div = float(a.nint)
    if a.mode == 'frac':
        seq, fr = pllnet.mash2(a.frac, a.nint, 4000)
        n_div = a.nint + fr
    net, ic = pllnet.build(a.T, d, mode=a.mode, n_int=a.nint, mash_seq=seq, f_cmp=f_cmp,
                           bias=bias, band=band, ic_vc=a.icvc, vto_shift=a.vto, cw_scale=a.cw, vto_var=a.vtovar, por=(dict(on=True) if a.por else None))
    nseg = len(band['segments']) if band else 0
    wv = ['ref', 'fb', 'ck', 'vc', 'o0', 'b0', 'vbn'] + ['bnd%d' % k for k in range(nseg)]
    t0 = time.time()
    w = netlib.run(net.text(), wv, a.tstop, 0.2e-9, tmax=a.tmax, ic=ic, timeout=40000,
                   save_extra=['i(vdd)', 'i(vbt)'],
                   extra_ctl=['meas tran iddavg avg i(vdd) from=%g to=%g' % (0.8 * a.tstop, a.tstop),
                              'meas tran ibtavg avg i(vbt) from=%g to=%g' % (0.8 * a.tstop, a.tstop)])
    for k in ['bnd%d' % k for k in range(nseg)]:
        pass
    res = analyse(w, f_cmp, n_div, a.tstop)
    res.update(T_K=a.T, mode=a.mode, n_div=n_div, por=bool(a.por), vto_shift=a.vto, vto_var=a.vtovar, cw_scale=a.cw, runtime_s=time.time() - t0,
               p_vdd_mW=-6.0 * w['_meas'].get('iddavg', np.nan) * 1e3,
               p_boot_mW=-7.0 * w['_meas'].get('ibtavg', np.nan) * 1e3, tstop_us=a.tstop * 1e6)
    os.makedirs(RES, exist_ok=True)
    name = RES + '/%s_%d_%s%s.json' % (a.tag, round(a.T), a.mode, '' if a.mode == 'int' else '_%g' % n_div)
    # decimated waveforms for figures
    t = w['t']; sel = np.linspace(0, len(t) - 1, min(len(t), 6000)).astype(int)
    res['wave'] = dict(t=t[sel].tolist(), vc=w['vc'][sel].tolist())
    eo = edges(t, w['ck'])
    if len(eo) > 10:
        fo = 1 / np.diff(eo); s2 = np.linspace(0, len(fo) - 1, min(len(fo), 6000)).astype(int)
        res['wave'].update(tf=eo[1:][s2].tolist(), f=fo[s2].tolist())
    json.dump(res, open(name, 'w'))
    print(json.dumps({k: v for k, v in res.items() if k != 'wave'}), flush=True)

if __name__ == '__main__':
    main()
