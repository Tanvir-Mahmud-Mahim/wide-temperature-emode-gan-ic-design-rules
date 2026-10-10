#!/usr/bin/env python3
"""18_tl_codesign.py -- co-design iteration 2 on the transistor-anchored model.

Iteration 1 (scripts 03, 05, 10) ran on the Spectre-calibrated behavioral model of the
manual design.  Transistor-level simulation (open flow) of that result exposed effects
the model could not represent (bootstrap start-up headroom at the hot corners, a ~4x
temperature spread of the ring frequency that the varactor bias alone cannot absorb,
real pump and filter elements, 2DEG temperature coefficients, gate leakage).  This
script re-optimizes the realizable design on the transistor-anchored model:

  frequency plan : f_cmp = 10 MHz, integer N = 5 (50 MHz); two cascaded 2/3 cells give
                   N = 4..7, so MASH-2 fractional operation covers N in [5, 6)
  coarse band    : B thermometer-coded segments of C_u per stage; band b (0..B) is
                   selected by on-chip temperature comparators
  design vector  : drive, s_var, bias-divider ratios (a2, ao), R_up (R_dn = 2 R_up), Rs1, Cs1,
                   Rs23, W_c2, W_c3, C_u
  objective      : dual-start lock at six corners, static phase error, K_VCO floor,
                   closed-loop jitter, VCRO power, pull-up gate stress (<= 5.2 V), and
                   band containment of the fractional range F_LO..F_HI with margin.
Writes results/tl_codesign.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp, optax
from ganpll import pinn, tl_anchor as ta, pll2

pst = pinn.load_state('results/pinn_state.json')
tp, ex = ta.load(os.environ.get('TL_VCRO', 'results/tl_vcro.json'))
F_CMP = pll2.F_CMP; N_INT = 5.0; F_TGT = N_INT * F_CMP
F_LO, F_HI = float(os.environ.get('F_LO', 50e6)), float(os.environ.get('F_HI', 55e6))
B = int(os.environ.get('NBAND', 4))
M_ON, V_HI, MARGIN = 0.15, 5.4, 0.05
TEMPS = (218., 253., 300., 348., 398., 423.)
TVER = (218., 233., 253., 273., 300., 323., 348., 373., 398., 423.)
W = dict(ferr=400., frac=20., ripple=10., phi=100., kv=40., jit=0.5, pw=1.0, vgs=50., band=200., pm=0.05, bw=20., on=50.)
ON_MARGIN = float(os.environ.get('ON_MARGIN', 0.4))
PM_MIN, BW_MAX = float(os.environ.get('PM_MIN', 50.0)), float(os.environ.get('BW_MAX', 1.0e6))
T_END, DT = float(os.environ.get('T_END', 30e-6)), 20e-9
V0 = (2.9, 5.4)        # start points: idle level of the pump (~3 V) and the top of the range

def windows(d, T, cb):
    vb = pll2.vb_of(d, T)
    von = ta.vc_on(tp, T, vb, d['drive'], d['s_var'], cb)
    flo = ta.f_model(tp, pst, T, jnp.minimum(von + M_ON, V_HI - 0.2), vb, d['drive'], d['s_var'], cb)
    fhi = ta.f_model(tp, pst, T, V_HI, vb, d['drive'], d['s_var'], cb)
    return flo, fhi

def band_margin(d, T, cb):
    flo, fhi = windows(d, T, cb)
    return jnp.minimum(jnp.log(fhi / F_HI), jnp.log(F_LO / flo))

def best_bands(d, temps):
    out = {}
    for T in temps:
        m = [float(band_margin(d, T, b * d['C_u'])) for b in range(B + 1)]
        out[T] = int(np.argmax(m))
    return out

def per_T(d, T, b):
    d = dict(d, cband=b * d['C_u'])
    cost = 0.0; vce = None
    for v0 in V0:
        tr = pll2.loop_transient(tp, pst, d, T, N_INT, vc0=v0, t_end=T_END, dt=DT)
        m = pll2.metrics(tr, F_TGT)
        cost += 0.5 * (W['ferr'] * m['ferr'] ** 2 + W['frac'] * (1 - m['frac']) ** 2
                       + W['ripple'] * m['ripple'] + W['phi'] * jnp.tanh(m['phi_ss'] / (2 * jnp.pi)) ** 2)
        if v0 == V0[0]:
            vce = m['vc_end']
    vg = jnp.linspace(vce - 0.3, vce + 0.3, 7)
    dfdv = jax.vmap(lambda v: jax.grad(lambda x: pll2.f_vco(tp, pst, d, x, T))(v))(vg)
    cost += W['kv'] * jnp.mean(jax.nn.relu(0.04 - dfdv / F_TGT) ** 2)
    vb = pll2.vb_of(d, T)
    pw = jnp.exp(ta.ridge_eval_jax(ex['power'], T, vce, vb, d['drive'], d['s_var'], d['cband']))
    sig, sp = pll2.jitter(tp, pst, d, T, vce, N_INT, 0.0, pw)
    cost += W['jit'] * (sig * 1e9) ** 2 + W['pw'] * (pw / 0.1) ** 2
    # small-signal stability: phase margin at the unity-gain frequency and a bandwidth cap
    # (the averaged loop model is valid for bandwidths well below f_cmp)
    L = sp['L']; mag = jnp.abs(L)
    i = jax.lax.stop_gradient(jnp.argmin(jnp.abs(jnp.log(mag))))
    pm = 180.0 + jnp.degrees(jnp.angle(L[i]))
    cost += W['pm'] * jax.nn.relu(PM_MIN - pm) ** 2
    lbw = jnp.interp(jnp.log(BW_MAX), jnp.log(sp['f']), jnp.log(mag))
    cost += W['bw'] * jax.nn.relu(lbw) ** 2
    vgs = ta.ridge_eval_jax(ex['vgs'], T, vce, vb, d['drive'], d['s_var'], d['cband'])
    cost += W['vgs'] * jax.nn.relu(vgs - 5.2) ** 2
    cost += W['band'] * jax.nn.relu(MARGIN - band_margin(d, T, d['cband'])) ** 2
    # keep the lock point away from the oscillation onset, where the oscillator gain is steep
    von = ta.vc_on(tp, T, vb, d['drive'], d['s_var'], d['cband'])
    cost += W['on'] * jax.nn.relu(von + ON_MARGIN - vce) ** 2
    return cost

def overlap_cost(d, bands):
    """Where the assigned band changes between neighbouring optimization temperatures,
    both bands must hold the target range at the midpoint temperature (with margin), so the
    comparator threshold has a tolerance window."""
    c = 0.0
    for T1, T2 in zip(TEMPS[:-1], TEMPS[1:]):
        b1, b2 = bands[T1], bands[T2]
        if b1 != b2:
            Tm = 0.5 * (T1 + T2)
            for b in range(min(b1, b2), max(b1, b2) + 1):
                c += jax.nn.relu(0.03 - band_margin(d, Tm, b * d['C_u'])) ** 2
    return c

def make_objective(bands):
    def objective(raw):
        d = pll2.raw_to_design(raw)
        J = sum((3.0 if T in (218., 423.) else 1.0) * per_T(d, T, bands[T]) for T in TEMPS) / len(TEMPS)
        return J + W['band'] * overlap_cost(d, bands)
    return objective

def evaluate(d, bands, n_div, frac_on, temps=TVER, t_end=60e-6):
    rows = []
    for T in temps:
        b = bands[T]; dd = dict(d, cband=b * d['C_u'])
        f_tgt = n_div * F_CMP; rec = dict(T_K=T, band=b)
        for v0, tag in zip(V0, ('lo', 'hi')):
            tr = pll2.loop_transient(tp, pst, dd, T, n_div, vc0=v0, t_end=t_end, dt=10e-9)
            m = {k: float(v) for k, v in pll2.metrics(tr, f_tgt).items()}
            lock = abs(m['ferr']) < 0.01 and m['exc'] < 2 * np.pi
            rec[f'f_end_{tag}_MHz'] = m['f_end'] / 1e6; rec[f'lock_{tag}'] = bool(lock)
            if tag == 'lo':
                ok = np.abs(np.asarray(tr['f']) - f_tgt) / f_tgt < 0.01
                bad = np.where(~ok)[0]
                rec.update(phi_ss_deg=m['phi_ss'] * 180 / np.pi, vc_end=m['vc_end'],
                           settle_us=float((bad[-1] + 1) * 10e-9 * 1e6) if len(bad) else 0.0)
        vb = float(pll2.vb_of(d, T)); cb = dd['cband']
        X = np.array([[T, rec['vc_end'], vb, float(d['drive']), float(d['s_var']), float(cb)]])
        pw = float(np.exp(ta.ridge_eval(ex['power'], X))[0])
        sig, sp = pll2.jitter(tp, pst, dd, T, rec['vc_end'], n_div, frac_on, pw)
        Lv = np.asarray(sp['L']); fv = np.asarray(sp['f']); iu = int(np.argmin(np.abs(np.log(np.abs(Lv)))))
        rec.update(f_ugb_kHz=float(fv[iu] / 1e3), pm_deg=float(180 + np.degrees(np.angle(Lv[iu]))))
        sigf, _ = pll2.jitter(tp, pst, dd, T, rec['vc_end'], 5.2, 1.0, pw); rec['jitter_frac_ns'] = float(sigf) * 1e9
        flo, fhi = windows(d, T, cb)
        rec.update(vb=vb, p_vco_mW=pw * 1e3, jitter_ns=float(sig) * 1e9, cband_pF=float(cb),
                   vgs_pu=float(ta.ridge_eval(ex['vgs'], X)[0]), win_lo_MHz=float(flo) / 1e6, win_hi_MHz=float(fhi) / 1e6,
                   lock=rec['lock_lo'] and rec['lock_hi'])
        rows.append(rec)
    return rows

def band_map(d, Tgrid=np.arange(218.0, 423.01, 5.0)):
    """Valid bands (positive margin) on a fine temperature grid -> comparator thresholds."""
    valid = {}
    for T in Tgrid:
        valid[float(T)] = [b for b in range(B + 1) if float(band_margin(d, T, b * d['C_u'])) > 0]
    thr = []
    for b in range(B, 0, -1):          # transition b -> b-1 with rising temperature
        both = [T for T, v in valid.items() if b in v and (b - 1) in v]
        thr.append(dict(from_band=b, to_band=b - 1, T_lo=min(both) if both else None,
                        T_hi=max(both) if both else None,
                        T_set=0.5 * (min(both) + max(both)) if both else None))
    return valid, thr

if __name__ == '__main__':
    d1 = dict(drive=1.5, s_var=1.05, a2=3.7, ao=2e-3, R_up=100e3, R_dn=300e3,
              Rs1=26e3, Cs1=40e-12, Rs23=90e3, W_c2=1000.0, W_c3=50.0, C_u=0.34)
    if os.environ.get('SEED'):
        d1.update(json.loads(os.environ['SEED']))
    raw = pll2.design_to_raw(d1)
    steps = int(os.environ.get('STEPS', 200)); every = 25
    opt = optax.adam(optax.cosine_decay_schedule(0.03, max(steps, 1), alpha=0.1)); st = opt.init(raw)
    hist = []; t0 = time.time(); bands = None
    for i in range(steps):
        if i % every == 0:
            nb = best_bands(pll2.raw_to_design(raw), TEMPS)
            if nb != bands:
                bands = nb; gf = jax.jit(jax.value_and_grad(make_objective(bands)))
                print('  bands', bands, flush=True)
        v, g = gf(raw); up, st = opt.update(g, st); raw = optax.apply_updates(raw, up)
        if i % 10 == 0 or i == steps - 1:
            hist.append((i, float(v), time.time() - t0)); print(f'  iter {i}: J={float(v):.5f} ({time.time()-t0:.0f}s)', flush=True)
    d = pll2.raw_to_design(raw) if steps > 0 else dict(pll2.raw_to_design(raw), **{k: jnp.asarray(v) for k, v in d1.items()})
    if steps == 0:      # evaluate a given design exactly (no rounding through the logistic map)
        d['W_ht'] = 26.0 * d['drive']; d['R_dn'] = pll2.R_DN_RATIO * d['R_up']
    dd = {k: float(v) for k, v in d.items()}
    print('design:', json.dumps({k: round(v, 6) for k, v in dd.items()}))
    valid, thr = band_map(d)
    print('thresholds:', json.dumps(thr))
    bands_v = best_bands(d, TVER)
    if os.environ.get('BANDS'):          # evaluate with the band settings of the implemented comparators
        bands_v = {float(k): int(v) for k, v in json.loads(os.environ['BANDS']).items()}
    rows = evaluate(d, bands_v, N_INT, 0.0)
    for r in rows:
        print(f"  T={r['T_K']:.0f} b={r['band']} win=[{r['win_lo_MHz']:.1f},{r['win_hi_MHz']:.1f}] f_lo={r['f_end_lo_MHz']:.3f} f_hi={r['f_end_hi_MHz']:.3f} lock={r['lock']} "
              f"phi={r['phi_ss_deg']:+.2f} vc={r['vc_end']:.2f} vb={r['vb']:.2f} set={r['settle_us']:.1f}us jit={r['jitter_ns']:.3f}/{r['jitter_frac_ns']:.2f}ns ugb={r['f_ugb_kHz']:.0f}k pm={r['pm_deg']:.0f} P={r['p_vco_mW']:.1f}mW vgs={r['vgs_pu']:.2f}", flush=True)
    json.dump(dict(design=dd, raw=np.asarray(raw).tolist(), hist=hist, integer=rows, bands=bands_v,
                   valid=valid, thresholds=thr, f_cmp=F_CMP, n_int=N_INT, F_LO=F_LO, F_HI=F_HI, B=B,
                   weights=W, m_on=M_ON, v_hi=V_HI), open(os.environ.get('OUT', 'results/tl_codesign.json'), 'w'), indent=1)
    print('saved', os.environ.get('OUT', 'results/tl_codesign.json'))
