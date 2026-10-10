"""Analytic charge-control physics prior for the e-mode p-GaN gate HEMT varactor.

The varactor is a two-terminal device: gate (terminal A) versus shorted
source/drain (terminal B).  Below the threshold voltage the 2DEG under the
gate is depleted and the capacitance is the series combination of the p-GaN /
AlGaN barrier capacitance and the depletion capacitance (C_min).  Above
threshold the 2DEG re-forms and connects to the access regions, so the
capacitance saturates towards the barrier capacitance times the gated area
(C_max).  As in virtual-source GaN HEMT models (MVSG), the channel charge is
taken as a smooth function n*Vt*ln(1 + exp((V - Vth)/(n*Vt))), whose derivative
is a logistic-shaped C-V transition:

    u(V,T)   = (V - Vth(T)) / (n(T) * Vt(T)),   Vt = k_B T / q
    C(V,T)   = C_min(T) + (C_max(T) - C_min(T)) * sigma(u)

with sigma the logistic function.  Temperature enters through
  * Vth(T) = Vth0 + kvt * (T - T0)          (linear threshold drift)
  * the thermal voltage Vt(T)  (transition broadening at high T)
  * n(T) = n0 * exp(kn*(T - T0))            (subthreshold-slope degradation)
  * C_max(T)-C_min(T) = dC0 * exp(kcm*(T - T0))  (weak barrier drift)
  * C_min(T) = Cmin0 * exp(kci*(T - T0))    (buffer/depletion drift)

All functions are written in JAX and are differentiable in every argument, so
the same code serves as (i) the physics prior of the PINN loss, (ii) the
generator of dense synthetic pre-training data, and (iii) a fully analytic
fallback varactor model inside the differentiable PLL.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

KB_Q = 8.617333262e-5  # k_B/q in V/K
T0 = 300.0             # reference temperature (K)


def default_params():
    """Physics-prior parameters in *unconstrained* (raw) space.

    Positivity and ordering constraints are enforced inside `resolve`:
      n0, cmin0, dc0 via softplus; temperature rates via exp scaling with
      raw values expressed in milli-units so all raws live on ~unit scale.
    """
    import numpy as np

    inv_softplus = lambda y: float(np.log(np.expm1(y)))
    return dict(
        vth0=2.05,                      # threshold at 300 K (V)
        kvt_m=-3.2,                     # dVth/dT in mV/K
        n0_raw=inv_softplus(18.0),      # ideality x smoothing at 300 K
        kn_m=2.4,                       # 1e-3/K transition broadening rate
        cmin0_raw=inv_softplus(1.0),    # normalized C_min
        dc0_raw=inv_softplus(2.4),      # normalized (C_max - C_min)
        kcm_m=0.15,                     # 1e-3/K C_max drift rate
        kci_m=0.60,                     # 1e-3/K C_min drift rate
    )


def resolve(p, T):
    """Map raw params -> physical, guaranteed-valid quantities at T."""
    dT = jnp.asarray(T) - T0
    n = jax.nn.softplus(p["n0_raw"]) * jnp.exp(1e-3 * p["kn_m"] * dT)
    cmin = jax.nn.softplus(p["cmin0_raw"]) * jnp.exp(1e-3 * p["kci_m"] * dT)
    dc = jax.nn.softplus(p["dc0_raw"]) * jnp.exp(1e-3 * p["kcm_m"] * dT)
    vth_T = p["vth0"] + 1e-3 * p["kvt_m"] * dT
    return n, cmin, dc, vth_T


def vth(T, p):
    return p["vth0"] + 1e-3 * p["kvt_m"] * (jnp.asarray(T) - T0)


def cv_analytic(V, T, p):
    """Normalized analytic C(V,T) prior. V, T broadcastable arrays."""
    Vt = KB_Q * jnp.asarray(T)
    n, cmin, dc, vth_T = resolve(p, T)
    u = (V - vth_T) / (n * Vt)
    return cmin + dc * jax.nn.sigmoid(u)


def dcdv_analytic(V, T, p):
    return jax.grad(lambda v: cv_analytic(v, T, p).sum())(V)


def charge_analytic(V, T, p):
    """Q(V) = integral of C dV (closed form for the logistic transition)."""
    Vt = KB_Q * jnp.asarray(T)
    n, cmin, dc, vth_T = resolve(p, T)
    nvt = n * Vt
    u = (V - vth_T) / nvt
    return cmin * V + dc * nvt * jax.nn.softplus(u)


def mobility_factor(T, gamma=1.6):
    """Normalized 2DEG mobility ~ (T/300K)^-gamma (phonon-limited)."""
    return (jnp.asarray(T) / T0) ** (-gamma)
