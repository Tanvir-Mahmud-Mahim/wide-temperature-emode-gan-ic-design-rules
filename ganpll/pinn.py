"""Physics-informed neural network (PINN) surrogate of the GaN HEMT varactor.

Architecture (physics-embedded, cf. Tung et al., IEEE T-ED 2025; Luo et al.,
IEEE TMTT 2026):

    C(V, T; g) = A_g * C_phys(V, T; theta_p) * (1 + eps * tanh(delta_NN(x)))

  * C_phys       : analytic charge-control prior (ganpll.physics), with its
                   parameters theta_p *learned jointly* with the network,
  * delta_NN     : small MLP correction on normalized inputs
                   x = [(V-3)/3, (T-370)/150, g], bounded to +-eps (default
                   15%) so the physical asymptotics (C_min/C_max saturation)
                   can never be destroyed by the network,
  * A_g          : per-device area scale (fF), g in {0,1} device flag.

Physics-informed loss terms (collocation grid, no labels needed):
  L_mono  : penalty on negative dC/dV  (2DEG charge control => C monotone)
  L_Tsmooth: penalty on d2C/dT2        (no spurious temperature resonances)
  L_prior : weak anchor of the correction to zero outside the data window
            (controls extrapolation)

Total: L = L_data + lam_mono*L_mono + lam_T*L_Tsmooth + lam_prior*L_prior.
Setting all lambdas to zero and eps -> large recovers a plain black-box NN,
used as the ablation baseline in the paper.
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from . import physics

jax.config.update("jax_enable_x64", True)


def init_mlp(key, sizes=(3, 32, 32, 1)):
    params = []
    for i in range(len(sizes) - 1):
        key, k1, k2 = jax.random.split(key, 3)
        w = jax.random.normal(k1, (sizes[i], sizes[i + 1])) * jnp.sqrt(2.0 / sizes[i])
        b = jnp.zeros(sizes[i + 1])
        params.append((w, b))
    return params


def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return (x @ w + b).squeeze(-1)


def init_state(key, eps=0.15):
    p0 = physics.default_params()
    return dict(
        net=init_mlp(key),
        phys={k: jnp.asarray(v) for k, v in p0.items()},
        logA=jnp.log(jnp.asarray([5.3, 1060.0])),  # fF per device (small, large)
        eps=eps,
    )


def _norm_inputs(V, T, g):
    V, T, g = jnp.broadcast_arrays(jnp.asarray(V), jnp.asarray(T), jnp.asarray(g))
    return jnp.stack([(V - 3.0) / 3.0, (T - 370.0) / 150.0, g], axis=-1)


def cap_fF(state, V, T, g):
    """Surrogate capacitance in fF. g: device flag (0 small, 1 large)."""
    cshape = physics.cv_analytic(V, T, state["phys"])
    delta = mlp(state["net"], _norm_inputs(V, T, g))
    A = jnp.exp(state["logA"])[g.astype(int)] if hasattr(g, "astype") else jnp.exp(state["logA"])[int(g)]
    return A * cshape / 3.4 * (1.0 + state["eps"] * jnp.tanh(delta))


def cap_unit(state, V, T, g=0):
    """Normalized capacitance shape (used by the PLL model, scaled there)."""
    gv = jnp.full(jnp.shape(V), float(g))
    cshape = physics.cv_analytic(V, T, state["phys"])
    delta = mlp(state["net"], _norm_inputs(V, T, gv))
    return cshape / 3.4 * (1.0 + state["eps"] * jnp.tanh(delta))


def make_loss(data, lam_mono=2.0, lam_T=0.5, lam_prior=0.05, physics_on=True):
    """data: dict with arrays V, T, g, C_fF (training points)."""
    Vd, Td, gd, Cd = (jnp.asarray(data[k]) for k in ("V", "T", "g", "C_fF"))
    # collocation grid for physics terms (covers extrapolation region too)
    Vc_, Tc_ = jnp.meshgrid(jnp.linspace(-0.5, 7.0, 40), jnp.linspace(200.0, 540.0, 35))
    Vc_, Tc_ = Vc_.ravel(), Tc_.ravel()
    gc_ = jnp.zeros_like(Vc_)

    def loss(state):
        pred = cap_fF(state, Vd, Td, gd)
        l_data = jnp.mean(((pred - Cd) / Cd) ** 2)
        if not physics_on:
            return l_data, dict(data=l_data)
        hV, hT = 0.02, 2.0  # central differences on the collocation grid
        dCdV = (cap_unit(state, Vc_ + hV, Tc_, 0) - cap_unit(state, Vc_ - hV, Tc_, 0)) / (2 * hV)
        l_mono = jnp.mean(jax.nn.relu(-dCdV) ** 2) * 1e2
        c0 = cap_unit(state, Vc_, Tc_, 0)
        d2d = (cap_unit(state, Vc_, Tc_ + hT, 0) - 2 * c0 + cap_unit(state, Vc_, Tc_ - hT, 0)) / hT**2
        l_T = jnp.mean(d2d ** 2) * 1e6
        delta = mlp(state["net"], _norm_inputs(Vc_, Tc_, gc_))
        l_prior = jnp.mean(delta ** 2)
        tot = l_data + lam_mono * l_mono + lam_T * l_T + lam_prior * l_prior
        return tot, dict(data=l_data, mono=l_mono, tsmooth=l_T, prior=l_prior)

    return loss


def adam_train(loss_fn, state, steps=4000, lr=3e-3, log_every=500, trainable=("net", "phys", "logA")):
    import optax

    flat_keys = [k for k in ("net", "phys", "logA") if k in trainable]
    opt = optax.adam(lr)

    train_state = {k: state[k] for k in flat_keys}
    static = {k: v for k, v in state.items() if k not in flat_keys}

    def full(ts):
        s = dict(static)
        s.update(ts)
        return s

    @jax.jit
    def step(ts, opt_state):
        (l, aux), grads = jax.value_and_grad(lambda t: loss_fn(full(t)), has_aux=True)(ts)
        updates, opt_state = opt.update(grads, opt_state)
        ts = optax.apply_updates(ts, updates)
        return ts, opt_state, l, aux

    opt_state = opt.init(train_state)
    hist = []
    for i in range(steps):
        train_state, opt_state, l, aux = step(train_state, opt_state)
        if i % log_every == 0 or i == steps - 1:
            hist.append((i, float(l), {k: float(v) for k, v in aux.items()}))
    return full(train_state), hist


def build_training_arrays(cv_small, cv_large, hold_out_T=None):
    """Assemble (V,T,g,C_fF) arrays from digitized curves; optionally hold out
    one temperature for interpolation testing."""
    V, T, g, C = [], [], [], []
    Vh, Th, gh, Ch = [], [], [], []
    for dev, curves, flag, scale in (("small", cv_small, 0.0, 1e3), ("large", cv_large, 1.0, 1e3)):
        for c in curves:
            tgt = (Vh, Th, gh, Ch) if (hold_out_T and abs(c["T"] - hold_out_T) < 1) else (V, T, g, C)
            tgt[0].extend(c["V"]); tgt[1].extend([c["T"]] * len(c["V"]))
            tgt[2].extend([flag] * len(c["V"])); tgt[3].extend(c["C_pF"] * scale)
    train = dict(V=np.array(V), T=np.array(T), g=np.array(g), C_fF=np.array(C))
    test = dict(V=np.array(Vh), T=np.array(Th), g=np.array(gh), C_fF=np.array(Ch))
    return train, test


def save_state(state, path):
    ser = dict(
        net=[(np.asarray(w).tolist(), np.asarray(b).tolist()) for w, b in state["net"]],
        phys={k: float(v) for k, v in state["phys"].items()},
        logA=np.asarray(state["logA"]).tolist(),
        eps=state["eps"],
    )
    with open(path, "w") as f:
        json.dump(ser, f)


def load_state(path):
    with open(path) as f:
        ser = json.load(f)
    return dict(
        net=[(jnp.asarray(w), jnp.asarray(b)) for w, b in ser["net"]],
        phys={k: jnp.asarray(v) for k, v in ser["phys"].items()},
        logA=jnp.asarray(ser["logA"]),
        eps=ser["eps"],
    )
