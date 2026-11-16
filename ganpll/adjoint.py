"""Adjoint-gradient (reverse-mode AD) co-design of the GaN PLL across
temperature.

The design vector collects multiplicative scales on the realizable circuit
quantities (charge-pump current, loop-filter R/C, LPF varactor sizes, VCRO
drive width, fixed delay capacitor, delay-varactor size) plus an additive
varactor bias offset.  Bounds are enforced by a tanh reparameterization, so
the optimizer works in an unconstrained space while every candidate remains
physically realizable.

The objective is evaluated on full loop transients at a set of temperature
corners; reverse-mode differentiation through the `lax.scan` time-stepper is
mathematically the discrete adjoint method (one backward sweep of the
linearized dynamics per objective evaluation), giving the exact gradient with
respect to *all* design parameters at a cost independent of their number
(Director & Rohrer 1969; Chen et al. 2018).
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import pll

# name, lo, hi (multiplicative unless noted), init 1.0 (0.0 for dvb)
DESIGN_SPACE = [
    ("s_icp", 0.25, 8.0),
    ("s_rs1", 0.2, 5.0),
    ("s_cs1", 0.2, 5.0),
    ("s_rs23", 0.2, 5.0),
    ("s_c2", 0.25, 20.0),
    ("s_c3", 0.25, 20.0),
    ("drive", 0.5, 6.0),
    ("s_cfix", 0.08, 3.0),
    ("s_var", 0.3, 6.0),
    ("dvb", -2.0, 2.0),  # additive (V)
    ("dvb_t", -8e-3, 8e-3),  # thermally tracked bias slope (V/K)
]


def raw_to_design(raw):
    d = {}
    for i, (name, lo, hi) in enumerate(DESIGN_SPACE):
        if name in ("dvb", "dvb_t"):
            d[name] = lo + (hi - lo) * jax.nn.sigmoid(raw[i])
        else:
            # log-space mapping: raw=0 -> 1.0 (baseline)
            llo, lhi = jnp.log(lo), jnp.log(hi)
            frac0 = (0.0 - llo) / (lhi - llo)
            u = jax.nn.sigmoid(raw[i] + jnp.log(frac0 / (1 - frac0)))
            d[name] = jnp.exp(llo + (lhi - llo) * u)
    return d


def init_raw():
    raw = np.zeros(len(DESIGN_SPACE))
    # dvb: sigmoid(raw)=(0-lo)/(hi-lo)=0.5 -> raw=0 gives dvb=0
    return jnp.asarray(raw)


def make_objective(vp, pst, circ, temps, weights=None,
                   t_end=60e-6, dt=20e-9, f_ref=50e6):
    w = dict(ferr=300.0, lockfrac=1.0, ripple=10.0, kvco=50.0, power=0.02,
             kmin=0.04)
    if weights:
        w.update(weights)

    Vgrid = jnp.linspace(1.0, 5.5, 40)
    tw = {T: 1.0 for T in temps}
    tw[max(temps)] = 3.0  # emphasize the hottest spec corner
    if 300.0 in tw:
        tw[300.0] = 2.0   # nominal corner must not regress

    def per_temp(design, T):
        # dual-start transients: acquisition from below and from above the
        # lock point must both converge (kills spurious equilibria)
        cost = 0.0
        m = None
        for vc0 in (1.0, 5.5):
            tr = pll.loop_transient(vp, pst, circ, T, design=design,
                                    vc0=vc0, t_end=t_end, dt=dt, f_ref=f_ref)
            mm = pll.lock_metrics(tr, f_ref=f_ref)
            cost = cost + 0.5 * (w["ferr"] * mm["ferr"] ** 2
                                 + w["lockfrac"] * (1.0 - mm["frac_locked"]) ** 2
                                 + w["ripple"] * mm["ripple"])
            if vc0 == 1.0:
                m = mm
        # normalized slope (fraction of f_ref per volt); require a minimum
        # positive K_VCRO across the window (no inversions, no stall pockets)
        f_of_v = lambda v: pll.f_vcro(vp, pst, v, T, design)
        dfdv = jax.vmap(jax.grad(f_of_v))(Vgrid)
        kvco_neg = jnp.mean(jax.nn.relu(w["kmin"] - dfdv / f_ref) ** 2)
        cost = cost + w["kvco"] * kvco_neg
        return cost, m

    def objective(raw):
        design = raw_to_design(raw)
        tot = 0.0
        for T in temps:
            c, _ = per_temp(design, T)
            tot = tot + tw[T] * c
        # power proxy: wider gates and larger CP current burn more power
        tot = tot + w["power"] * ((design["drive"] - 1.0) ** 2
                                  + 0.2 * (design["s_icp"] - 1.0) ** 2)
        return tot / len(temps)

    return objective, per_temp


def optimize(objective, raw0, steps=120, lr=0.05, log_every=10):
    sched = optax.cosine_decay_schedule(lr, steps, alpha=0.1)
    opt = optax.adam(sched)
    grad_fn = jax.jit(jax.value_and_grad(objective))
    raw = raw0
    ostate = opt.init(raw)
    hist = []
    t0 = time.time()
    for i in range(steps):
        val, g = grad_fn(raw)
        up, ostate = opt.update(g, ostate)
        raw = optax.apply_updates(raw, up)
        if i % log_every == 0 or i == steps - 1:
            hist.append((i, float(val), time.time() - t0))
            print(f"  iter {i:4d}: J = {float(val):.5f}  ({time.time()-t0:.0f}s)")
    return raw, hist


def fd_gradient(objective, raw, h=1e-3):
    """Central finite-difference gradient (baseline for cost comparison)."""
    raw = np.asarray(raw)
    g = np.zeros_like(raw)
    obj = jax.jit(objective)
    for i in range(len(raw)):
        e = np.zeros_like(raw); e[i] = h
        g[i] = (float(obj(jnp.asarray(raw + e))) - float(obj(jnp.asarray(raw - e)))) / (2 * h)
    return g


def random_search(objective, n, key=0, sigma=1.5):
    """Random-search baseline with the same evaluation budget."""
    rng = np.random.default_rng(key)
    obj = jax.jit(objective)
    best, best_raw, hist = np.inf, None, []
    for i in range(n):
        cand = jnp.asarray(rng.normal(0, sigma, len(DESIGN_SPACE)))
        v = float(obj(cand))
        if v < best:
            best, best_raw = v, cand
        hist.append((i, best))
    return best_raw, best, hist
