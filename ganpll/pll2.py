"""Iteration-2 (transistor-anchored) differentiable loop model.

Differences from the iteration-1 model (pll.py), each traceable to the
transistor-level (open-flow) implementation:
  * frequency plan: comparison frequency f_cmp (default 12.5 MHz) and a
    feedback ratio N >= 2 realizable by a multi-modulus divider (MASH-2
    moduli N_int-1 ... N_int+2);
  * VCRO: transistor-anchored surrogate (tl_anchor) with the tracked
    varactor bias rail vb(T) = vb0 + s_T (T - 300 K) and the start-up
    boundary (no oscillation below Vc_on);
  * charge pump: e-mode switched 2DEG-resistor pump, I_up = (VDD - Vc)/R_up(T),
    I_dn = Vc/R_dn(T), plus the reset-overlap term (I_up - I_dn) t_rst f_cmp;
  * loop filter: drawn third-order topology, VCRO control taken after Rs23
    (Vc node carries C3 and the five tail gates); varactor C(V,T) per
    micron from the PINN shape mapped to the PDK model (scripts/24_cv_map.py);
  * 2DEG resistors carry the PDK temperature coefficients;
  * gate leakage of the varactors and tail gates from the PDK model.
"""
from __future__ import annotations
import json
import jax, jax.numpy as jnp
import numpy as np
from . import pinn, tl_anchor as ta, pdk_local

jax.config.update("jax_enable_x64", True)
VDD = 6.0
F_CMP = 10e6          # comparison frequency; N = 5 gives 50 MHz
# PINN shape -> foundry-model capacitance per micron, C = C_A + C_B * cap_unit (fitted by
# scripts/24_cv_map.py; PDK-derived, stored locally in results/cv_map.json)
T_RST = 4e-9                        # PFD reset-overlap width (from the transistor-level 300 K lock)
L_G = 1.3e-6

_DEV = None
def _dev():
    global _DEV
    if _DEV is None:
        d = pdk_local.devtables()
        V = np.asarray(d['V'], dtype=float); T = np.asarray(d['T'], dtype=float)
        lg = np.log(np.maximum(np.asarray([d['Ig_A_per_um'][str(t)] for t in d['T']]), 1e-18))
        _DEV = (V, T, lg)
    return _DEV

def ig_per_um(v, T):
    """Varactor/gate leakage per micron (A/um), bilinear interpolation in log space."""
    V, Ts, lg = _dev()
    V, Ts, lg = jnp.asarray(V), jnp.asarray(Ts), jnp.asarray(lg)
    v = jnp.clip(v, V[0], V[-1]); T = jnp.clip(T, Ts[0], Ts[-1])
    iv = jnp.clip(jnp.searchsorted(V, v) - 1, 0, len(V) - 2)
    it = jnp.clip(jnp.searchsorted(Ts, T) - 1, 0, len(Ts) - 2)
    av = (v - V[iv]) / (V[iv + 1] - V[iv]); at = (T - Ts[it]) / (Ts[it + 1] - Ts[it])
    g = ((1 - av) * (1 - at) * lg[it, iv] + av * (1 - at) * lg[it, iv + 1]
         + (1 - av) * at * lg[it + 1, iv] + av * at * lg[it + 1, iv + 1])
    return jnp.exp(g) * jnp.where(v > 0, 1.0, 0.0)

def c_per_um(pst, v, T):
    m = pdk_local.cv_map()
    return (m['C_A'] + m['C_B'] * pinn.cap_unit(pst, v, T, 0)) * 1e-15     # F/um

def r2deg(r25, T):
    """2DEG resistor at temperature T (K) with the PDK temperature coefficients."""
    c = pdk_local.constants()['res2deg']; d = T - 298.15
    return r25 * (1.0 + c['ptc1'] * d + c['ptc2'] * d * d)

def rom(r25, T):
    """Ohmic-metal resistor at temperature T (K) with the PDK temperature coefficients."""
    c = pdk_local.constants()['resom']; d = T - 298.15
    return r25 * (1.0 + c['ptc1'] * d + c['ptc2'] * d * d)

def vb_of(d, T):
    """Varactor bias rail.  Realizable law (default): resistor divider whose top leg
    is ohmic metal and whose bottom leg is a 2DEG resistor in series with ohmic
    metal; ratios a2 = R2deg,bot/Rom,top and ao = Rom,bot/Rom,top (values at 25 C).
    The different PDK temperature coefficients make vb rise with temperature.
    Legacy linear law vb0 + s_T (T - 300 K) is kept for the iteration-1 analysis."""
    if 'a2' in d:
        bot = d['a2'] * r2deg(1.0, T) + d['ao'] * rom(1.0, T)
        return VDD * bot / (bot + rom(1.0, T))
    return d['vb0'] + d['s_T'] * (T - 300.0)

def f_vco(tp, pst, d, vc, T):
    """d['cband'] (pF) is the band capacitance switched in at this temperature."""
    return ta.f_eff(tp, pst, T, vc, vb_of(d, T), d['drive'], d['s_var'], d.get('cband', 0.0))

R_DN_RATIO = 2.0   # R_dn / R_up (same 2DEG material, so the ratio holds at every temperature)
V_SW = 0.15        # high-side switch drop at the pump current (transistor-level probe)

def pump(d, vc, T):
    i_up = (VDD - V_SW - vc) / r2deg(d['R_up'], T)
    i_dn = vc / r2deg(d['R_dn'], T)
    return jnp.maximum(i_up, 0.0), jnp.maximum(i_dn, 0.0)

def caps(pst, d, vx, vc, T):
    cx = d['W_c2'] * c_per_um(pst, vx, T) + 0.3e-12                 # node X + CP-side parasitic
    cin = 5 * d['W_ht'] * c_per_um(pst, vc, T)                       # five tail gates
    cc = d['W_c3'] * c_per_um(pst, vc, T) + cin + 0.2e-12
    return cx, cc

def leak(d, vx, vc, T):
    return d['W_c2'] * ig_per_um(vx, T), (d['W_c3'] + 5 * d['W_ht']) * ig_per_um(vc, T)

def loop_transient(tp, pst, d, T, n_div, vc0=0.0, t_end=60e-6, dt=10e-9, f_cmp=F_CMP):
    rs1 = r2deg(d['Rs1'], T); rs23 = r2deg(d['Rs23'], T); cs1 = d['Cs1']
    n = int(t_end / dt)
    def step(s, _):
        phi, vx, v1, vc = s
        f = f_vco(tp, pst, d, vc, T)
        iu, idn = pump(d, vc, T)
        x = phi / (2 * jnp.pi)
        xp = 0.5 * (x + jnp.sqrt(x * x + 1e-6)); xn = 0.5 * (-x + jnp.sqrt(x * x + 1e-6))
        # UP is gated by the reference (start-up refresh of the bootstrapped high-side switch),
        # so its duty saturates at 1/2; DOWN saturates at 1
        i_cp = iu * 0.5 * jnp.tanh(2 * xp) - idn * jnp.tanh(xn) + (iu - idn) * T_RST * f_cmp
        lx, lc = leak(d, vx, vc, T)
        cx, cc = caps(pst, d, vx, vc, T)
        dvx = (i_cp - (vx - v1) / rs1 - (vx - vc) / rs23 - lx) / cx
        v1n = vx + (v1 - vx) * jnp.exp(-dt / (rs1 * cs1))
        tauc = rs23 * cc
        vc_inf = vx - lc * rs23
        vcn = vc_inf + (vc - vc_inf) * jnp.exp(-dt / tauc)
        vxn = jnp.clip(vx + dt * dvx, 0.0, VDD)
        phin = phi + dt * 2 * jnp.pi * (f_cmp - f / n_div)
        return (phin, vxn, v1n, jnp.clip(vcn, 0.0, VDD)), (vcn, f, phin)
    init = tuple(jnp.asarray(x, dtype=float) for x in (0.0, vc0, vc0, vc0))
    _, (vc_t, f_t, ph_t) = jax.lax.scan(step, init, None, length=n)
    return dict(t=jnp.arange(n) * dt, vc=vc_t, f=f_t, phi=ph_t)

def metrics(tr, f_tgt, tail=0.2):
    n = tr['f'].shape[0]; k = int(tail * n)
    f_end = jnp.mean(tr['f'][-k:])
    err = jnp.abs(tr['f'] - f_tgt) / f_tgt
    return dict(f_end=f_end, ferr=(f_end - f_tgt) / f_tgt,
                frac=jnp.mean(jax.nn.sigmoid((0.01 - err) / 0.002)),
                ripple=jnp.var(tr['f'][-k:]) / f_tgt ** 2,
                phi_ss=tr['phi'][-1], exc=jnp.max(tr['phi'][-k:]) - jnp.min(tr['phi'][-k:]),
                vc_end=jnp.mean(tr['vc'][-k:]))

# ------------------------------------------------------------------ noise
L_VCO_1M = -83.3       # dBc/Hz @1 MHz, 50 MHz carrier, 300 K, baseline VCRO (Spectre)
P_VCO_BASE = 32e-3     # baseline VCRO power (W), Spectre power breakdown (29-35 mW)
L_IB_50 = -89.1        # in-band floor at N = 1, 50 MHz comparison (Spectre, 300 K)

def zf(pst, d, vx, vc, T, w):
    s = 1j * w
    cx, cc = caps(pst, d, vx, vc, T)
    rs1 = r2deg(d['Rs1'], T); rs23 = r2deg(d['Rs23'], T)
    y23 = 1.0 / (rs23 + 1.0 / (s * cc))
    yx = s * cx + 1.0 / (rs1 + 1.0 / (s * d['Cs1'])) + y23
    return (1.0 / yx) * (1.0 / (1.0 + s * rs23 * cc))

def jitter(tp, pst, d, T, vc, n_div, frac_on, p_vco, f_cmp=F_CMP, m=2):
    """Closed-loop RMS jitter (s).  Delta-sigma noise: output-referred
    (2 pi)^2/(12 f_cmp) [2 sin(pi f/f_cmp)]^(2(m-1)), shaped by H (no N^2)."""
    f0 = n_div * f_cmp
    f = jnp.logspace(3.0, jnp.log10(f_cmp / 2), 200); w = 2 * jnp.pi * f
    kv = jax.grad(lambda v: f_vco(tp, pst, d, v, T))(vc)
    iu, idn = pump(d, vc, T)
    kpd = 0.5 * (iu + idn) / (2 * jnp.pi)
    vx = vc + leak(d, vc, vc, T)[1] * r2deg(d['Rs23'], T)
    L = kpd * zf(pst, d, vx, vc, T, w) * 2 * jnp.pi * kv / (1j * w * n_div)
    H = L / (1 + L); E = 1 / (1 + L)
    s_vco = (10 ** (L_VCO_1M / 10) * (1e6 / f) ** 2 * (f0 / 50e6) ** 2 * (T / 300.0) ** 1.5
             * (P_VCO_BASE / p_vco) * (1 + 2e5 / f))
    s_ib = 10 ** (L_IB_50 / 10) * (f_cmp / 50e6) * (T / 300.0)
    s_dsm = frac_on * (2 * jnp.pi) ** 2 / (12 * f_cmp) * (2 * jnp.sin(jnp.pi * f / f_cmp)) ** (2 * (m - 1))
    s_out = jnp.abs(E) ** 2 * s_vco + jnp.abs(H) ** 2 * (n_div ** 2 * s_ib + s_dsm)
    var = 2 * jnp.trapezoid(s_out, f)
    return jnp.sqrt(var) / (2 * jnp.pi * f0), dict(f=f, s_out=s_out, L=L, s_vco=s_vco, s_dsm=s_dsm)

# ------------------------------------------------------------------ design space
SPACE = [  # name, lo, hi, log?
    ('drive', 1.0, 4.0, True), ('s_var', 1.0, 6.0, True), ('a2', 0.02, 50.0, True),
    ('ao', 1e-3, 20.0, True), ('R_up', 5e3, 300e3, True),
    ('Rs1', 2e3, 200e3, True), ('Cs1', 1e-12, 50e-12, True), ('Rs23', 5e3, 500e3, True),
    ('W_c2', 2.0, 2000.0, True), ('W_c3', 2.0, 500.0, True), ('C_u', 0.1, 3.0, True)]

def raw_to_design(raw):
    d = {}
    for i, (k, lo, hi, lg) in enumerate(SPACE):
        u = jax.nn.sigmoid(raw[i])
        d[k] = jnp.exp(jnp.log(lo) + (jnp.log(hi) - jnp.log(lo)) * u) if lg else lo + (hi - lo) * u
    d['W_ht'] = 26.0 * d['drive']            # tail width (um) follows the drive scale
    d['R_dn'] = R_DN_RATIO * d['R_up']       # pump balance: I_up = I_dn near Vc ~ 3.9 V
    return d

def design_to_raw(d):
    raw = []
    for k, lo, hi, lg in SPACE:
        u = ((np.log(d[k]) - np.log(lo)) / (np.log(hi) - np.log(lo))) if lg else (d[k] - lo) / (hi - lo)
        u = np.clip(u, 1e-4, 1 - 1e-4); raw.append(np.log(u / (1 - u)))
    return jnp.asarray(raw)

def power(tp_extra, d, vc, T):
    X = np.array([[float(T), float(vc), float(vb_of(d, T)), float(d['drive']), float(d['s_var']),
                   float(d.get('cband', 0.0))]])
    return float(np.exp(ta.ridge_eval(tp_extra['power'], X))[0])
