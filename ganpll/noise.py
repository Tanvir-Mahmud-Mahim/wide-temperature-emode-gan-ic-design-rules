"""Differentiable closed-loop phase-noise and jitter model (JAX).

Extends the loop model with the three noise paths of a (fractional-)N
synthesizer, all differentiable so that jitter can sit directly inside the
adjoint co-design objective:

  S_out(f) = |1/(1+L)|^2 S_vco(f)
           + |L/(1+L)|^2 N^2 S_inband(f)
           + |L/(1+L)|^2 N^2 S_dsm(f)

  * S_vco  : Leeson-like VCO noise anchored to the transistor-level
             -83.3 dBc/Hz at 1 MHz offset (50 MHz carrier, 300 K), scaled
             (f0/50MHz)^2 (ring FoM), (T/300K)^1.5 (thermal), with a flicker
             corner.
  * S_inband: reference + PFD/CP in-band floor, anchored to the
             transistor-level -89.1 dBc/Hz close-in noise at N=1, multiplied
             by N^2 as in any divider-based synthesizer.
  * S_dsm  : MASH-2 delta-sigma quantization noise of the fractional divider,
             S = (2 pi)^2/(12 f_comp) (2 sin(pi f/f_comp))^2, low-pass
             filtered by the loop.  Zero for integer N.

  L(jw) = Kpd Zf(jw) (2 pi Kvco) / (jw N),  Kpd = Icp/2pi.

RMS jitter: sigma_t = sqrt(2 integral S_out df) / (2 pi f0).
The static phase error under charge-pump leakage I_leak and mismatch m is
also exposed: at lock the pump must cancel the disturbance, giving
phi_ss ~ 2 pi (I_leak/Icp - m) inside the linear detector range; the
transient model reproduces it exactly, and the objective can penalize it.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from . import pinn, pll

L_VCO_1M = -83.3    # dBc/Hz at 1 MHz offset, 50 MHz carrier, 300 K
L_INBAND = -89.1    # dBc/Hz in-band floor at N=1, 300 K
F_FLICKER = 2e5     # Hz


def _zf_jax(circ, d, pinn_state, T, vc, w):
    s = 1j * w
    c2 = (circ["c2_fF"] * 1e-15) * d.get("s_c2", 1.0) * pinn.cap_unit(pinn_state, vc, T, 0) + circ["cpar"]
    c3 = (circ["c3_fF"] * 1e-15) * d.get("s_c3", 1.0) * pinn.cap_unit(pinn_state, vc, T, 0) + 1e-14
    rs1 = circ["rs1"] * d.get("s_rs1", 1.0)
    cs1 = circ["cs1"] * d.get("s_cs1", 1.0)
    rs23 = circ["rs23"] * d.get("s_rs23", 1.0)
    y = s * c2 + 1.0 / (rs1 + 1.0 / (s * cs1)) + 1.0 / (rs23 + 1.0 / (s * c3))
    return 1.0 / y


def output_jitter(vp, pinn_state, circ, T, vc, design=None, f_ref=50e6,
                  n_grid=160):
    """Differentiable RMS jitter (seconds) and phase-noise pieces at the
    settled control voltage vc.  Returns (sigma_t, aux dict)."""
    d = design or {}
    n_div = d.get("n_div", 1.0)
    frac = d.get("frac_on", 0.0)   # 1.0 => fractional mode (DSM noise active)
    icp = circ["icp"] * d.get("s_icp", 1.0)
    kpd = icp / (2.0 * jnp.pi)

    f0 = n_div * f_ref
    f = jnp.logspace(3.0, 7.5, n_grid)
    w = 2.0 * jnp.pi * f

    kv = jax.grad(lambda v: pll.f_vcro(vp, pinn_state, v, T, d))(vc)
    zf = _zf_jax(circ, d, pinn_state, T, vc, w)
    L = kpd * zf * (2.0 * jnp.pi * kv) / (1j * w * n_div)
    s_err = 1.0 / (1.0 + L)          # VCO high-pass
    s_low = L / (1.0 + L)            # input-referred low-pass

    # VCO open-loop noise (referred to the output carrier f0)
    l_th = 10.0 ** (L_VCO_1M / 10.0) * (1e6 / f) ** 2 * (f0 / 50e6) ** 2 \
        * (jnp.asarray(T) / 300.0) ** 1.5
    s_vco = l_th * (1.0 + F_FLICKER / f)

    # in-band reference/PFD/CP floor, xN^2
    s_ib = 10.0 ** (L_INBAND / 10.0) * (jnp.asarray(T) / 300.0) * n_div ** 2

    # MASH-2 delta-sigma quantization noise (fractional mode only), xN^2
    f_comp = f_ref
    s_dsm = frac * (2.0 * jnp.pi) ** 2 / (12.0 * f_comp) \
        * (2.0 * jnp.sin(jnp.pi * f / f_comp)) ** 2 * n_div ** 2

    s_out = (jnp.abs(s_err) ** 2 * s_vco
             + jnp.abs(s_low) ** 2 * (s_ib + s_dsm))
    var = 2.0 * jnp.trapezoid(s_out, f)
    sigma_t = jnp.sqrt(var) / (2.0 * jnp.pi * f0)
    return sigma_t, dict(f=f, s_out=s_out, s_vco=s_vco,
                         s_ib=jnp.full_like(f, s_ib), s_dsm=s_dsm, L=L)


def pn_at(vp, pinn_state, circ, T, vc, design, foffs=(1e3, 1e4, 1e5, 1e6),
          f_ref=50e6):
    """Closed-loop phase noise (dBc/Hz) at given offsets."""
    _, aux = output_jitter(vp, pinn_state, circ, T, vc, design, f_ref)
    out = {}
    for fo in foffs:
        idx = jnp.argmin(jnp.abs(aux["f"] - fo))
        out[fo] = 10.0 * jnp.log10(aux["s_out"][idx])
    return out
