"""Physics-constrained WGAN-GP over the varactor physics-parameter space.

Generative model of device-to-device variability of the GaN HEMT varactor,
expressed in the 8 raw (unconstrained) parameters of the analytic physics
prior (`ganpll.physics`): vth0, kvt_m, n0_raw, kn_m, cmin0_raw, dc0_raw,
kcm_m, kci_m.  A Wasserstein GAN with gradient penalty (Gulrajani et al.,
NeurIPS 2017) is trained on parameter vectors obtained by re-fitting the
physics prior to the measured C-V curves; sampling the generator then gives
plausible device *variants* whose C(V,T) can be pushed through the
differentiable PLL for Monte-Carlo lock-yield analysis.

HONEST SCOPE STATEMENT
----------------------
The training set is built from only 10 measured C-V curves (5 temperatures
x 2 device geometries, Fig. F2 of the manuscript), each re-fit individually,
and augmented to 200 vectors by bootstrap refits under 1% multiplicative
measurement noise.  The learned distribution therefore reflects
*characterization scatter* (curve-to-curve residuals of the compact model
across temperature and geometry, plus measurement-noise-induced fit
uncertainty) -- NOT true fab process statistics, for which no wafer-level
data is available.  Three further modeling choices are deliberate and
should be kept in mind when interpreting yields:

  1. Vth inflation.  The raw fit scatter of vth0 (~0.05-0.06 V) understates
     realistic p-GaN gate Vth process spread; the vth0 deltas are inflated
     so the training/generated sigma is ~VTH_TARGET_SIGMA = 0.1 V, a
     representative value for p-GaN gate technology.
  2. Rate-parameter jitter.  The four temperature-rate parameters (kvt_m,
     kn_m, kcm_m, kci_m) are not identifiable from a single-temperature
     curve fit and are held fixed during fitting; a small seeded Gaussian
     jitter (a few % of nominal) is injected into the training vectors so
     the 8-D distribution is full rank.
  3. Frozen NN correction.  When a generated parameter vector is used as a
     device variant inside the PLL, only the physics-prior parameters of
     the PINN state are replaced; the bounded NN correction (trained on the
     nominal device) stays fixed.  Variants therefore perturb the physical
     backbone of the surrogate while retaining the nominal shape correction.
  4. Re-centering on the nominal parameters.  Because the per-curve prior
     fits absorb part of what the PINN's NN correction represents, their
     mean is systematically offset from the nominal PINN parameters; that
     offset is surrogate-decomposition ambiguity, not device variability.
     Only the fit *scatter* (deltas about the fit mean) is kept, and the
     generated distribution is centered on the nominal parameter vector.

ARCHITECTURE
------------
Generator : z in R^4 ~ N(0,I) -> MLP 4-32-32-8 -> tanh -> x in (-1,1)^8;
            theta = theta_nom + S * x: bounded deltas around the *nominal*
            PINN physics parameters, S = 3 x per-parameter training sigma
            (vth0 sigma inflated to 0.1 V).  The tanh bound plus the softplus
            reparameterization inside `physics.resolve` guarantee C_min,
            C_max - C_min > 0 for every sample (C ratios stay positive).
Critic    : MLP (8+3)-32-32-1 on the normalized parameter vector
            concatenated with 3 physics features of the generated 300 K
            curve (C_min, C_max, V_t50 = bias at half transition), computed
            through the full PINN surrogate.  Gradient penalty weight 10.
Physics constraint: a differentiable generator penalty (and a hard
            rejection test at sampling time) on (i) non-monotonic C(V) at
            any T in {218, 300, 423} K and (ii) Cmax/Cmin at 300 K outside
            [2, 6].

Training  : ~3000 critic iterations, 5 critic steps per generator step,
            batch 64, Adam(1e-4, beta1=0.5, beta2=0.9). CPU, a few minutes.
"""
from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import physics, pinn

jax.config.update("jax_enable_x64", True)

# canonical ordering of the 8 raw physics parameters
PKEYS = ("vth0", "kvt_m", "n0_raw", "kn_m", "cmin0_raw", "dc0_raw", "kcm_m", "kci_m")
# the 4 parameters fitted per curve (rates are not single-T identifiable)
FIT4 = ("vth0", "n0_raw", "cmin0_raw", "dc0_raw")
RATE4 = ("kvt_m", "kn_m", "kcm_m", "kci_m")

VTH_TARGET_SIGMA = 0.1  # V, representative p-GaN gate Vth process spread
# seeded jitter (std) injected on the non-identifiable rate parameters
RATE_JITTER = dict(kvt_m=0.10, kn_m=0.07, kcm_m=0.005, kci_m=0.008)

VGRID = jnp.linspace(0.0, 6.0, 41)
CON_TEMPS = (218.0, 300.0, 423.0)
RATIO_LO, RATIO_HI = 2.0, 6.0
# Monotonicity deadband: the frozen NN correction carries ~0.1%-of-range
# ripple in the flat C-V tails (even every *training-set* vector shows it),
# so "non-monotonic" means a per-step decrease exceeding 1% of the curve's
# full C_max - C_min swing -- a genuine inversion, not correction ripple.
MONO_TOL = 0.01


def phys_dict(pv):
    """8-vector (PKEYS order) -> raw physics-parameter dict."""
    return {k: pv[i] for i, k in enumerate(PKEYS)}


def phys_vec(p):
    """Raw physics-parameter dict -> 8-vector (PKEYS order)."""
    return jnp.asarray([p[k] for k in PKEYS])


def variant_state(pst, pv):
    """PINN state for a device variant: replace the physics-prior params,
    keep the (nominal) NN correction and every other field fixed."""
    pstv = dict(pst)
    pstv["phys"] = phys_dict(pv)
    return pstv


def variant_curve(pst, pv, T, V=VGRID):
    """Normalized C(V) of a variant through the full PINN surrogate."""
    return pinn.cap_unit(variant_state(pst, pv), V, T, 0)


# ----------------------------------------------------------------------
# 1. Training set: per-curve refits of the physics prior
# ----------------------------------------------------------------------

def fit_training_set(pst, n_boot=20, noise=0.01, steps=800, lr=0.02, seed=0):
    """Fit the physics prior to each measured C-V curve individually.

    For each of the 10 digitized curves (5 temperatures x 2 devices) the four
    single-temperature-identifiable parameters (vth0, n0_raw, cmin0_raw,
    dc0_raw) are fit with Adam on that curve alone, with the temperature-rate
    parameters frozen at their nominal (PINN) values; temperature effects are
    captured by evaluating the prior at the curve's own T.  Each curve is
    additionally re-fit `n_boot` times under 1% multiplicative measurement
    noise (bootstrap), yielding 10 clean + 10*n_boot bootstrap vectors.

    Amplitudes are made comparable across devices by calibrating each device's
    measured curves to the nominal prior at 300 K (median pointwise ratio), so
    that fitted cmin0/dc0 capture curve-shape scatter, not the (already
    NN-absorbed) geometric area difference between the two devices.

    Returns dict(theta_boot (10*n_boot, 8), theta_clean (10, 8),
                 loss_clean, loss_boot, curves meta).
    """
    from . import data

    nom = {k: float(v) for k, v in pst["phys"].items()}
    rates = {k: jnp.asarray(nom[k]) for k in RATE4}
    p4_0 = jnp.asarray([nom[k] for k in FIT4])
    rng = np.random.default_rng(seed)
    opt = optax.adam(lr)

    def fit_batch(V, T, Ctar):
        def loss(p4, C):
            p = dict(vth0=p4[0], n0_raw=p4[1], cmin0_raw=p4[2], dc0_raw=p4[3],
                     **rates)
            return jnp.mean(((physics.cv_analytic(V, T, p) - C) / C) ** 2)

        def fit_one(C):
            def step(carry, _):
                p4, os_ = carry
                l, g = jax.value_and_grad(loss)(p4, C)
                up, os_ = opt.update(g, os_)
                return (optax.apply_updates(p4, up), os_), l

            (p4f, _), ls = jax.lax.scan(step, (p4_0, opt.init(p4_0)), None,
                                        length=steps)
            return p4f, ls[-1]

        return jax.jit(jax.vmap(fit_one))(Ctar)

    theta_clean, theta_boot, loss_clean, loss_boot, meta = [], [], [], [], []
    for dev in ("small", "large"):
        curves = data.load_varactor_cv(dev)
        # per-device amplitude calibration against the nominal prior at 300 K
        c300 = [c for c in curves if abs(c["T"] - 300.0) < 1][0]
        prior300 = np.asarray(physics.cv_analytic(jnp.asarray(c300["V"]),
                                                  300.0, pst["phys"]))
        k_dev = float(np.median(prior300 / c300["C_pF"]))
        for c in curves:
            Cn = np.asarray(c["C_pF"]) * k_dev
            reps = [Cn] + [Cn * (1.0 + noise * rng.standard_normal(Cn.shape))
                           for _ in range(n_boot)]
            p4, ls = fit_batch(jnp.asarray(c["V"]), c["T"],
                               jnp.asarray(np.stack(reps)))
            p4 = np.asarray(p4)

            def full8(p4row):
                v = np.array([nom[k] for k in PKEYS])
                for j, k in enumerate(FIT4):
                    v[PKEYS.index(k)] = p4row[j]
                return v

            theta_clean.append(full8(p4[0]))
            loss_clean.append(float(ls[0]))
            for r in range(1, n_boot + 1):
                theta_boot.append(full8(p4[r]))
                loss_boot.append(float(ls[r]))
            meta.append(dict(device=dev, T=float(c["T"]), k_dev=k_dev,
                             loss=float(ls[0])))

    theta_boot = np.stack(theta_boot)
    theta_clean = np.stack(theta_clean)
    # full-rank regularization: seeded jitter on the non-identifiable rates
    for k, sig in RATE_JITTER.items():
        theta_boot[:, PKEYS.index(k)] += sig * rng.standard_normal(len(theta_boot))
    return dict(theta_boot=theta_boot, theta_clean=theta_clean,
                loss_clean=loss_clean, loss_boot=loss_boot, meta=meta)


def prepare_gan_space(theta_boot, pv_nom):
    """Deltas about the fit mean, re-centered on the nominal parameters.

    The training deltas are the fit vectors minus their own mean (the
    systematic fit-vs-nominal offset is surrogate-decomposition ambiguity
    and is discarded, see module docstring); the vth0 column is inflated to
    the target p-GaN sigma.  Returns (mu, S, X, inflation, offset):
    theta = mu + S*x with mu = the *nominal* vector, X the normalized
    training deltas, S = 3 x per-parameter sigma after inflation, and
    offset = nominal - fit mean (recorded for transparency)."""
    theta = np.asarray(theta_boot, dtype=float).copy()
    mean_fit = theta.mean(axis=0)
    d = theta - mean_fit
    s_vth = d[:, PKEYS.index("vth0")].std()
    inflation = max(1.0, VTH_TARGET_SIGMA / max(s_vth, 1e-9))
    d[:, PKEYS.index("vth0")] *= inflation
    sig = np.maximum(d.std(axis=0), 1e-6)
    S = 3.0 * sig
    mu = np.asarray(pv_nom, dtype=float)
    return (jnp.asarray(mu), jnp.asarray(S), jnp.asarray(d / S),
            float(inflation), mu - mean_fit)


# ----------------------------------------------------------------------
# 2. Physics features / constraint
# ----------------------------------------------------------------------

def _mlp_apply(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def make_physics_fns(pst):
    """Build (features, penalty, accept) closures over the nominal PINN state.

    features(pv) -> (3,): (Cmin, Cmax, Vt50) of the variant curve at 300 K.
    penalty(pv)  -> scalar >= 0, differentiable: monotonicity violation of
                    C(V) at T in {218, 300, 423} K + Cmax/Cmin at 300 K
                    outside [2, 6].
    accept(pv)   -> bool: hard version of the same constraint (rejection).
    """

    def features(pv):
        C = variant_curve(pst, pv, 300.0)
        cmin, cmax = C[0], C[-1]
        vt50 = jnp.interp(0.5 * (cmin + cmax), C, VGRID)
        return jnp.stack([cmin, cmax, vt50])

    def penalty(pv):
        pen = 0.0
        for T in CON_TEMPS:
            C = variant_curve(pst, pv, T)
            tol = MONO_TOL * (C[-1] - C[0])
            pen = pen + 1e4 * jnp.sum(jax.nn.relu(-(C[1:] - C[:-1]) - tol) ** 2)
        C0 = variant_curve(pst, pv, 300.0)
        r = C0[-1] / C0[0]
        pen = pen + jax.nn.relu(RATIO_LO - r) ** 2 + jax.nn.relu(r - RATIO_HI) ** 2
        return pen

    def accept(pv):
        ok = True
        for T in CON_TEMPS:
            C = variant_curve(pst, pv, T)
            tol = MONO_TOL * (C[-1] - C[0])
            ok = jnp.logical_and(ok, jnp.all(C[1:] - C[:-1] > -tol))
        C0 = variant_curve(pst, pv, 300.0)
        r = C0[-1] / C0[0]
        return jnp.logical_and(ok, jnp.logical_and(r >= RATIO_LO, r <= RATIO_HI))

    return features, penalty, accept


# ----------------------------------------------------------------------
# 3. WGAN-GP
# ----------------------------------------------------------------------

LATENT = 4


def init_generator(key):
    return pinn.init_mlp(key, (LATENT, 32, 32, 8))


def init_critic(key):
    return pinn.init_mlp(key, (8 + 3, 32, 32, 1))


def gen_apply(gp, z):
    """z (B, LATENT) -> normalized parameter vector x in (-1, 1)^8."""
    return jnp.tanh(_mlp_apply(gp, z))


def gen_theta(gp, z, mu, S):
    return mu + S * gen_apply(gp, z)


def train_wgan(X_real, pst, mu, S, key, critic_steps=3000, n_critic=5,
               batch=64, lr=1e-4, gp_weight=10.0, lam_phys=30.0, log_every=100):
    """WGAN-GP training loop (Gulrajani 2017, Alg. 1). Returns state + hist."""
    features, penalty, _ = make_physics_fns(pst)
    f_nom = features(phys_vec(pst["phys"]))
    f_scale = jnp.asarray([0.1, 0.2, 0.2])

    def crit_scalar(cp, x):
        theta = mu + S * x
        fn = (features(theta) - f_nom) / f_scale
        return _mlp_apply(cp, jnp.concatenate([x, fn]))[0]

    def crit_batch(cp, X):
        return jax.vmap(lambda x: crit_scalar(cp, x))(X)

    kg, kc, key = jax.random.split(key, 3)
    gparams, cparams = init_generator(kg), init_critic(kc)
    opt_g = optax.adam(lr, b1=0.5, b2=0.9)
    opt_c = optax.adam(lr, b1=0.5, b2=0.9)
    og, oc = opt_g.init(gparams), opt_c.init(cparams)
    n_real = X_real.shape[0]

    def critic_loss(cp, gpar, k):
        ki, kz, ke = jax.random.split(k, 3)
        xr = X_real[jax.random.randint(ki, (batch,), 0, n_real)]
        xf = gen_apply(gpar, jax.random.normal(kz, (batch, LATENT)))
        eps = jax.random.uniform(ke, (batch, 1))
        xh = eps * xr + (1.0 - eps) * xf
        gnorm = jax.vmap(lambda x: jnp.linalg.norm(
            jax.grad(lambda xx: crit_scalar(cp, xx))(x)))(xh)
        w_est = jnp.mean(crit_batch(cp, xr)) - jnp.mean(crit_batch(cp, xf))
        gp_term = gp_weight * jnp.mean((gnorm - 1.0) ** 2)
        return -w_est + gp_term, (w_est, gp_term)

    def gen_loss(gpar, cp, k):
        xf = gen_apply(gpar, jax.random.normal(k, (batch, LATENT)))
        pen = jnp.mean(jax.vmap(penalty)(mu + S * xf))
        return -jnp.mean(crit_batch(cp, xf)) + lam_phys * pen, pen

    @jax.jit
    def critic_step(cp, oc, gpar, k):
        (l, aux), g = jax.value_and_grad(critic_loss, has_aux=True)(cp, gpar, k)
        up, oc = opt_c.update(g, oc)
        return optax.apply_updates(cp, up), oc, l, aux

    @jax.jit
    def gen_step(gpar, og, cp, k):
        (l, pen), g = jax.value_and_grad(gen_loss, has_aux=True)(gpar, cp, k)
        up, og = opt_g.update(g, og)
        return optax.apply_updates(gpar, up), og, l, pen

    hist = []
    for i in range(critic_steps):
        key, kc_ = jax.random.split(key)
        cparams, oc, lc, (w_est, gp_term) = critic_step(cparams, oc, gparams, kc_)
        if (i + 1) % n_critic == 0:
            key, kg_ = jax.random.split(key)
            gparams, og, lg, pen = gen_step(gparams, og, cparams, kg_)
        if i % log_every == 0 or i == critic_steps - 1:
            hist.append(dict(step=i, w_est=float(w_est), gp=float(gp_term),
                             gen_loss=float(lg) if i >= n_critic else None,
                             phys_pen=float(pen) if i >= n_critic else None))
    return dict(gen=gparams, crit=cparams, mu=mu, S=S), hist


def sample_thetas(gan, pst, key, n, reject=True, max_rounds=40):
    """Sample n raw-parameter vectors; rejection-filter physics violators.

    Returns (thetas (n, 8), acceptance_rate)."""
    _, _, accept = make_physics_fns(pst)
    accept_v = jax.jit(jax.vmap(accept))
    gen = jax.jit(lambda z: gen_theta(gan["gen"], z, gan["mu"], gan["S"]))
    out, n_gen, n_ok = [], 0, 0
    for _ in range(max_rounds):
        key, kz = jax.random.split(key)
        th = gen(jax.random.normal(kz, (max(n, 128), LATENT)))
        m = np.asarray(accept_v(th)) if reject else np.ones(len(th), bool)
        n_gen += len(th)
        n_ok += int(m.sum())
        out.append(np.asarray(th)[m])
        if sum(len(o) for o in out) >= n:
            break
    thetas = np.concatenate(out)[:n]
    if len(thetas) < n:
        raise RuntimeError(f"only {len(thetas)}/{n} samples accepted")
    return thetas, n_ok / n_gen


def summary_stats(thetas, pst):
    """(Vt50 at 300 K, Cmax/Cmin at 300 K) arrays for a set of variants."""
    features, _, _ = make_physics_fns(pst)
    F = np.asarray(jax.jit(jax.vmap(features))(jnp.asarray(thetas)))
    return F[:, 2], F[:, 1] / F[:, 0]


# ----------------------------------------------------------------------
# 4. Persistence
# ----------------------------------------------------------------------

def save_gan(gan, path, extra=None):
    ser = dict(
        gen=[(np.asarray(w).tolist(), np.asarray(b).tolist()) for w, b in gan["gen"]],
        crit=[(np.asarray(w).tolist(), np.asarray(b).tolist()) for w, b in gan["crit"]],
        mu=np.asarray(gan["mu"]).tolist(),
        S=np.asarray(gan["S"]).tolist(),
        pkeys=list(PKEYS), latent=LATENT,
    )
    if extra:
        ser["extra"] = extra
    with open(path, "w") as f:
        json.dump(ser, f)


def load_gan(path):
    with open(path) as f:
        ser = json.load(f)
    return dict(
        gen=[(jnp.asarray(w), jnp.asarray(b)) for w, b in ser["gen"]],
        crit=[(jnp.asarray(w), jnp.asarray(b)) for w, b in ser["crit"]],
        mu=jnp.asarray(ser["mu"]), S=jnp.asarray(ser["S"]),
    )
