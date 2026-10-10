"""Differentiable behavioral model of the monolithic GaN-on-SOI PLL.

Every block is written in JAX and is differentiable end-to-end, so
reverse-mode automatic differentiation provides exact discrete-adjoint
gradients of any transient or small-signal metric with respect to the design
parameters (charge-pump current, loop-filter elements, varactor geometry
scales, VCRO drive strength, varactor bias offset).

Blocks
------
VCRO   : five-stage ring oscillator; per-stage delay
             tau(Vc,T) = tau_g0*r(T)^g1 + kc0*r(T)^g2 * (rho + c(Vb0-Vc, T))
         with r(T)=T/300K, c = PINN varactor surrogate (normalized shape) and
         a bounded multiplicative NN residual gamma(Vc,T) calibrated to the
         Cadence Spectre tuning curves (captures DCFL swing/bootstrap effects
         the physical delay expression misses).
PFD+CP : phase-averaged tri-state model, i = Icp * tanh(phi_e / 2pi)
         (linear for small phase error, saturating at +-Icp).
LPF    : third-order passive filter from the manuscript: node capacitance
         C2(V,T) (varactor) + parasitic, branch1 Rs1-Cs1 (MIM), branch3
         Rs23-C3(V,T) (varactor); exponential (exact-relaxation) integration
         for the stiff branches.
Noise  : linearized loop transfer functions; VCO phase noise scaled with
         temperature; output phase-noise integral -> RMS jitter proxy.
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from . import pinn

jax.config.update("jax_enable_x64", True)

N_STAGES = 5
F_REF = 50e6
T_END = 100e-6
DT = 10e-9


# ----------------------------------------------------------------------
# VCRO tuning model
# ----------------------------------------------------------------------

def init_vcro_params(key=None):
    """Calibration parameters of the VCRO delay expression + NN residual."""
    key = key if key is not None else jax.random.PRNGKey(3)
    return dict(
        log_tau_g0=jnp.log(1.0e-9),   # intrinsic DCFL stage delay at 300 K (s)
        g1=jnp.asarray(1.5),          # mobility exponent of the gate delay
        log_kc0=jnp.log(0.6e-9),      # delay per unit normalized varactor cap (s)
        g2=jnp.asarray(1.0),
        rho=jnp.asarray(3.0),         # fixed MIM / parasitic in units of s_v
        vb0=jnp.asarray(6.0),         # effective varactor bias offset (V)
        net=pinn.init_mlp(key, (2, 24, 24, 1)),
    )


def _resid(vp, Vc, T):
    Vc, T = jnp.broadcast_arrays(jnp.asarray(Vc), jnp.asarray(T))
    x = jnp.stack([(Vc - 3.0) / 3.0, (T - 370.0) / 150.0], axis=-1)
    return 0.5 * jnp.tanh(pinn.mlp(vp["net"], x))  # bounded +-50%


def stage_delay(vp, pinn_state, Vc, T, design=None):
    """Per-stage delay (s).  design: optional dict of optimization scales."""
    d = design or {}
    r = jnp.asarray(T) / 300.0
    drive = d.get("drive", 1.0)          # gate width scale (speeds everything)
    s_cfix = d.get("s_cfix", 1.0)        # fixed delay-cap scale
    s_var = d.get("s_var", 1.0)          # delay varactor size scale
    dvb = d.get("dvb", 0.0)              # varactor bias offset shift (V)
    dvb_t = d.get("dvb_t", 0.0)          # thermally tracked bias slope (V/K),
                                         # realized by a diode-connected e-mode
                                         # HEMT sensor whose Vth shift tracks T
    dvb_eff = dvb + dvb_t * (jnp.asarray(T) - 300.0)
    c_par = d.get("c_par", 0.0)          # layout parasitic per-stage load,
                                         # in normalized capacitance units
                                         # (1 unit ~ 5.3 fF full-scale varactor)
    c = pinn.cap_unit(pinn_state, vp["vb0"] + dvb_eff - Vc, T, 0)
    tau_g = jnp.exp(vp["log_tau_g0"]) * r ** vp["g1"] / drive
    tau_c = jnp.exp(vp["log_kc0"]) * r ** vp["g2"] * (vp["rho"] * s_cfix + s_var * c + c_par) / drive
    gamma = 1.0 + _resid(vp, Vc, T)
    par_fac = d.get("par_fac", 1.0)      # layout parasitic slowdown factor,
                                         # back-annotated from transistor-level
                                         # (OpenVAF/ngspice) PEX runs
    return (tau_g + tau_c) * gamma * par_fac


def f_vcro(vp, pinn_state, Vc, T, design=None):
    return 1.0 / (N_STAGES * stage_delay(vp, pinn_state, Vc, T, design))


def kvco(vp, pinn_state, Vc, T, design=None):
    """dF/dVc (Hz/V) via automatic differentiation."""
    f = lambda v: f_vcro(vp, pinn_state, v, T, design)
    return jax.grad(lambda v: f(v).sum() if hasattr(f(v), "sum") else f(v))(Vc)


# ----------------------------------------------------------------------
# Loop-filter + loop transient (fully differentiable, lax.scan RK/exp)
# ----------------------------------------------------------------------

def default_circuit():
    """Baseline circuit values from the manuscript appendix."""
    return dict(
        icp=8e-6,        # effective average charge-pump current (A) [calibrated]
        rs1=20e3,        # branch-1 series resistance (Ohm)
        cs1=20e-12,      # branch-1 MIM capacitor (F)
        rs23=30e3,       # branch-3 series resistance (Ohm)
        c2_fF=520.0,     # node varactor full-scale capacitance (fF) [LPF C2]
        c3_fF=260.0,     # branch-3 varactor full-scale capacitance (fF)
        cpar=2e-12,      # CP output + VCRO control-node parasitic (F)
        vdd=6.0,
    )


def loop_transient(vp, pinn_state, circ, T, design=None, vc0=0.5,
                   t_end=T_END, dt=DT, f_ref=F_REF):
    """Phase-averaged transient of the full loop.  Returns dict of traces."""
    d = design or {}
    icp = circ["icp"] * d.get("s_icp", 1.0)
    rs1 = circ["rs1"] * d.get("s_rs1", 1.0)
    cs1 = circ["cs1"] * d.get("s_cs1", 1.0)
    rs23 = circ["rs23"] * d.get("s_rs23", 1.0)
    s_c2 = d.get("s_c2", 1.0)
    s_c3 = d.get("s_c3", 1.0)
    cpar = circ["cpar"]
    vdd = circ["vdd"]

    c2f = lambda v: (circ["c2_fF"] * 1e-15) * s_c2 * pinn.cap_unit(pinn_state, v, T, 0) + cpar
    c3f = lambda v: (circ["c3_fF"] * 1e-15) * s_c3 * pinn.cap_unit(pinn_state, v, T, 0) + 1e-14

    n_steps = int(t_end / dt)

    n_div = d.get("n_div", 1.0)          # feedback division ratio (fractional-N:
                                         # the averaged model uses the effective
                                         # ratio; delta-sigma quantization noise
                                         # is added analytically in the jitter
                                         # integral)
    i_leak = d.get("i_leak", 0.0)        # gate/varactor leakage at the CP node (A)
    cp_mis = d.get("cp_mis", 0.0)        # up/down current mismatch (fraction)
    cal = d.get("cal", 0.0)              # mismatch calibration depth (0..0.9),
                                         # regenerative current-copying CP
    cp_mis = cp_mis * (1.0 - cal)

    def step(state, _):
        phie, vx, v1, v3 = state
        f_out = f_vcro(vp, pinn_state, vx, T, design)
        dphie = 2.0 * jnp.pi * (f_ref - f_out / n_div)
        i_cp = icp * (jnp.tanh(phie / (2.0 * jnp.pi)) + cp_mis) - i_leak
        i1 = (vx - v1) / rs1
        i3 = (vx - v3) / rs23
        dvx = (i_cp - i1 - i3) / c2f(vx)
        # exact relaxation for the series branches (stiff-safe)
        tau1 = rs1 * cs1
        tau3 = rs23 * c3f(v3)
        v1n = vx + (v1 - vx) * jnp.exp(-dt / tau1)
        v3n = vx + (v3 - vx) * jnp.exp(-dt / tau3)
        phien = phie + dt * dphie
        vxn = jnp.clip(vx + dt * dvx, 0.0, vdd)
        return (phien, vxn, v1n, v3n), (vxn, f_out, phien)

    init = (jnp.asarray(0.0), jnp.asarray(vc0), jnp.asarray(vc0), jnp.asarray(vc0))
    _, (vc_t, f_t, phie_t) = jax.lax.scan(step, init, None, length=n_steps)
    t = jnp.arange(n_steps) * dt
    return dict(t=t, vc=vc_t, f=f_t, phie=phie_t)


def lock_metrics(traces, f_ref=F_REF, tail_frac=0.2):
    """Differentiable lock metrics from a transient."""
    n = traces["f"].shape[0]
    k = int(tail_frac * n)
    f_tail = traces["f"][-k:]
    f_end = jnp.mean(f_tail)
    ferr = (f_end - f_ref) / f_ref
    # settle time: first time |f-fref|<1% sustained (soft version for grad)
    err = jnp.abs(traces["f"] - f_ref) / f_ref
    locked = jax.nn.sigmoid((0.01 - err) / 0.002)
    frac_locked = jnp.mean(locked)
    f_ripple = jnp.var(f_tail) / f_ref**2  # variance-normalized (smooth at 0)
    return dict(f_end=f_end, ferr=ferr, frac_locked=frac_locked, ripple=f_ripple)


# ----------------------------------------------------------------------
# Small-signal loop analysis
# ----------------------------------------------------------------------

def zf(vp, pinn_state, circ, T, vc, w, design=None):
    """Loop-filter transimpedance Zf(jw) linearized at operating point vc."""
    d = design or {}
    s = 1j * w
    c2 = (circ["c2_fF"] * 1e-15) * d.get("s_c2", 1.0) * float(pinn.cap_unit(pinn_state, vc, T, 0)) + circ["cpar"]
    c3 = (circ["c3_fF"] * 1e-15) * d.get("s_c3", 1.0) * float(pinn.cap_unit(pinn_state, vc, T, 0)) + 1e-14
    rs1 = circ["rs1"] * d.get("s_rs1", 1.0)
    cs1 = circ["cs1"] * d.get("s_cs1", 1.0)
    rs23 = circ["rs23"] * d.get("s_rs23", 1.0)
    y = s * c2 + 1.0 / (rs1 + 1.0 / (s * cs1)) + 1.0 / (rs23 + 1.0 / (s * c3))
    return 1.0 / y


def open_loop(vp, pinn_state, circ, T, vc, design=None, n=400):
    """Open-loop gain L(jw) on a log grid; returns w, L."""
    d = design or {}
    icp = circ["icp"] * d.get("s_icp", 1.0)
    kpd = icp / (2.0 * jnp.pi)
    kv = float(kvco(vp, pinn_state, jnp.asarray(vc), T, design))
    w = np.logspace(2.5, 7.5, n) * 2 * np.pi
    z = np.array([complex(zf(vp, pinn_state, circ, T, vc, wi, design)) for wi in w])
    L = kpd * z * (2 * np.pi * kv) / (1j * w)
    return w, L


def stability_margins(vp, pinn_state, circ, T, vc, design=None):
    w, L = open_loop(vp, pinn_state, circ, T, vc, design)
    mag = np.abs(L)
    idx = np.where(mag >= 1.0)[0]
    if len(idx) == 0 or idx[-1] == len(mag) - 1 and mag[-1] > 1:
        return dict(pm_deg=np.nan, f_ugb_Hz=np.nan)
    i = idx[-1]
    # interpolate crossover
    lm = np.log10(mag)
    x = lm[i] / (lm[i] - lm[i + 1])
    wc = w[i] * (w[i + 1] / w[i]) ** x
    ph = np.angle(L[i]) + (np.angle(L[i + 1]) - np.angle(L[i])) * x
    pm = 180.0 + np.degrees(ph)
    return dict(pm_deg=float(pm), f_ugb_Hz=float(wc / 2 / np.pi))


# ----------------------------------------------------------------------
# Phase noise / jitter proxy
# ----------------------------------------------------------------------

def vco_phase_noise(T, f_off, f0, a_th=-83.3, f_flick=2e5, T_ref=300.0):
    """VCO open-loop phase noise (dBc/Hz), Leeson-like, thermal-noise ~ T.

    a_th: measured PN at 1 MHz offset, 50 MHz carrier, 300 K (manuscript)."""
    l_th = 10 ** (a_th / 10) * (1e6 / f_off) ** 2 * (f0 / 50e6) ** 2 * (T / T_ref) ** 1.5
    l_fl = l_th * (f_flick / f_off)
    return l_th + l_fl


def closed_loop_jitter(vp, pinn_state, circ, T, vc, design=None, f0=F_REF):
    """RMS jitter proxy: integrate |1/(1+L)|^2 * S_vco over offsets."""
    w, L = open_loop(vp, pinn_state, circ, T, vc, design, n=300)
    f = w / (2 * np.pi)
    Svco = vco_phase_noise(T, f, f0)
    Sout = np.abs(1.0 / (1.0 + L)) ** 2 * Svco
    var = 2.0 * np.trapezoid(Sout, f)
    return float(np.sqrt(var) / (2 * np.pi * f0)) * 1e9  # ns


# ----------------------------------------------------------------------
# Calibration persistence
# ----------------------------------------------------------------------

def save_vcro(vp, path):
    ser = {k: (np.asarray(v).tolist() if k != "net" else
               [(np.asarray(w).tolist(), np.asarray(b).tolist()) for w, b in v])
           for k, v in vp.items()}
    with open(path, "w") as f:
        json.dump(ser, f)


def load_vcro(path):
    with open(path) as f:
        ser = json.load(f)
    out = {}
    for k, v in ser.items():
        out[k] = ([(jnp.asarray(w), jnp.asarray(b)) for w, b in v] if k == "net"
                  else jnp.asarray(v))
    return out
