"""Transistor-anchored VCRO surrogate (co-design iteration 2).

The open-flow transistor-level VCRO (bootstrapped current-starved stage,
PDK MVSG devices, 7 V precharge rail) is sampled by a design of experiments
over (T, Vc, vb, drive, s_var).  This module fits a physics-structured,
differentiable surrogate to those samples:

    tau = A_g r^g1 / (drive u^a)
          + A_c r^g2 (rho/drive + s_var c(vx - vb, T) + k_b C_band) / (drive u^b)
    f   = [1 / (2 N_stage tau)] (1 + gamma_NN)            (gamma bounded +-30 %)
    u   = softplus((Vc - vt(T)) / w) w                     (tail-transistor overdrive)
    P(osc) = sigmoid((Vc - Vc_on(T, drive, vb, s_var, C_band)) / s)

C_band (pF) is the switched MIM band capacitor per stage (0 when the band is off).

with c(.) the frozen PINN varactor shape.  Power (supply current) and the
peak pull-up gate-source stress are fitted with small ridge regressions.
"""
from __future__ import annotations
import json
import jax, jax.numpy as jnp
import numpy as np
import optax
from . import pinn

jax.config.update("jax_enable_x64", True)
N_STAGE = 5

def load_doe(path):
    rs = [json.loads(l) for l in open(path)]
    rs = [r for r in rs if r.get('f') == r.get('f')]          # drop NaN (sim errors)
    X = np.array([[r['T'], r['vc'], r['vb'], r['drive'], r['s_var'], 1e12 * r.get('cband', 0.0)] for r in rs])
    f = np.array([r['f'] for r in rs]); osc = (f > 1e6).astype(float)
    idd = np.array([r.get('idd', np.nan) for r in rs])
    vgs = np.array([r.get('vgs_max', np.nan) for r in rs])
    return X, f, osc, idd, vgs

def init_params(key=jax.random.PRNGKey(11)):
    return dict(logAg=jnp.log(1.2e-9), g1=jnp.asarray(1.0), a=jnp.asarray(0.6),
                logAc=jnp.log(0.6e-9), g2=jnp.asarray(0.5), b=jnp.asarray(0.6),
                rho=jnp.asarray(0.3), vx=jnp.asarray(3.0),
                vt0=jnp.asarray(3.3), kt=jnp.asarray(2e-3), logw=jnp.log(0.25),
                c0=jnp.asarray(3.5), c1=jnp.asarray(3e-3), c2=jnp.asarray(0.0),
                c3=jnp.asarray(0.0), c4=jnp.asarray(0.0), c5=jnp.asarray(0.0), logs=jnp.log(0.1),
                logkb=jnp.log(0.5), net=pinn.init_mlp(key, (6, 16, 16, 1)))

def _xn(T, vc, vb, drive, svar, cb):
    return jnp.stack([(T - 320.) / 100., (vc - 4.5) / 1.5, (vb - 3.0) / 3.0,
                      jnp.log(drive) - 0.9, jnp.log(svar) - 1.0, cb / 2.0 - 0.5], axis=-1)

def f_model(p, pst, T, vc, vb, drive, svar, cb=0.0):
    T, vc, vb, drive, svar, cb = jnp.broadcast_arrays(*map(jnp.asarray, (T, vc, vb, drive, svar, cb)))
    r = T / 300.0
    w = jnp.exp(p['logw'])
    u = jax.nn.softplus((vc - p['vt0'] - p['kt'] * (T - 300.0)) / w) * w + 1e-3
    c = pinn.cap_unit(pst, p['vx'] - vb, T, 0)
    tau = (jnp.exp(p['logAg']) * r ** p['g1'] / (drive * u ** p['a'])
           + jnp.exp(p['logAc']) * r ** p['g2'] * (p['rho'] / drive + svar * c + jnp.exp(p['logkb']) * cb)
           / (drive * u ** p['b']))
    gam = 0.3 * jnp.tanh(pinn.mlp(p['net'], _xn(T, vc, vb, drive, svar, cb)))
    return (1.0 / (2 * N_STAGE * tau)) * (1.0 + gam)

def vc_on(p, T, vb, drive, svar, cb=0.0):
    return (p['c0'] + p['c1'] * (T - 300.0) + p['c2'] * jnp.log(drive) + p['c3'] * vb
            + p['c4'] * jnp.log(svar) + p.get('c5', 0.0) * cb)

def p_osc(p, T, vc, vb, drive, svar, cb=0.0):
    return jax.nn.sigmoid((vc - vc_on(p, T, vb, drive, svar, cb)) / jnp.exp(p['logs']))

def f_eff(p, pst, T, vc, vb, drive, svar, cb=0.0):
    """Smooth oscillation frequency used inside the loop model (0 below start-up)."""
    return f_model(p, pst, T, vc, vb, drive, svar, cb) * p_osc(p, T, vc, vb, drive, svar, cb)

def fit(X, f, osc, pst, steps=6000, lr=8e-3, p0=None, verbose=False):
    Xj = jnp.asarray(X); lf = jnp.log(jnp.maximum(jnp.asarray(f), 1.0)); oj = jnp.asarray(osc)
    def loss(p):
        fm = f_model(p, pst, *[Xj[:, i] for i in range(6)])
        e = (jnp.log(fm) - lf) ** 2
        lf_loss = jnp.sum(e * oj) / jnp.maximum(jnp.sum(oj), 1.0)
        po = p_osc(p, *[Xj[:, i] for i in range(6)])
        bce = -jnp.mean(oj * jnp.log(po + 1e-9) + (1 - oj) * jnp.log(1 - po + 1e-9))
        reg = 1e-4 * sum(jnp.sum(w_ ** 2) for w_, _ in p['net'])
        return lf_loss + 0.05 * bce + reg
    p = p0 if p0 is not None else init_params()
    opt = optax.adam(optax.cosine_decay_schedule(lr, steps, alpha=0.05)); st = opt.init(p)
    vg = jax.jit(jax.value_and_grad(loss))
    for i in range(steps):
        v, g = vg(p)
        up, st = opt.update(g, st); p = optax.apply_updates(p, up)
        if verbose and i % 1000 == 0:
            print('  fit', i, float(v))
    return p, float(loss(p))

def ridge_fit(X, y, lam=1e-3):
    """Quadratic-feature ridge regression (log-space power / stress models)."""
    Z = _quad(X)
    mu, sd = Z.mean(0), Z.std(0) + 1e-12
    Zn = (Z - mu) / sd
    A = Zn.T @ Zn + lam * len(y) * np.eye(Zn.shape[1])
    w = np.linalg.solve(A, Zn.T @ (y - y.mean()))
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=w.tolist(), b=float(y.mean()))

def _quad(X):
    X = np.atleast_2d(X)
    T, vc, vb, d, s, cb = X.T
    base = np.stack([(T - 320) / 100, vc, vb, np.log(d), np.log(s), cb], 1)
    q = [base] + [base[:, i:i + 1] * base[:, j:j + 1] for i in range(6) for j in range(i, 6)]
    return np.concatenate(q, 1)

def ridge_eval(m, X):
    Z = (_quad(X) - np.array(m['mu'])) / np.array(m['sd'])
    return Z @ np.array(m['w']) + m['b']

def save(p, path, extra=None):
    ser = {k: ([(np.asarray(w).tolist(), np.asarray(b).tolist()) for w, b in v] if k == 'net'
               else float(v)) for k, v in p.items()}
    if extra:
        ser['_extra'] = extra
    json.dump(ser, open(path, 'w'))

def load(path):
    ser = json.load(open(path))
    extra = ser.pop('_extra', None)
    p = {k: ([(jnp.asarray(w), jnp.asarray(b)) for w, b in v] if k == 'net' else jnp.asarray(v))
         for k, v in ser.items()}
    return p, extra

def ridge_eval_jax(m, T, vc, vb, d, s, cb=0.0):
    base = jnp.stack([(T - 320) / 100, vc, vb, jnp.log(d), jnp.log(s), jnp.asarray(cb, dtype=float)])
    q = [base] + [base[i:i + 1] * base[j:j + 1] for i in range(6) for j in range(i, 6)]
    z = (jnp.concatenate(q) - jnp.asarray(m['mu'])) / jnp.asarray(m['sd'])
    return z @ jnp.asarray(m['w']) + m['b']
