#!/usr/bin/env python3
"""21_pinn_uq.py -- uncertainty and validation of the PINN varactor surrogate.

(a) Deep ensemble (5 seeds, identical protocol, 300 K held out): ensemble
    mean error, predictive spread, and empirical coverage of the +-2 sigma
    band on the held-out curve (Lakshminarayanan et al., NeurIPS 2017).
(b) Leave-one-temperature-out: each of the five characterized temperatures
    is held out in turn (218 and 423 K are then extrapolation tests).
Writes results/pinn_uq.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from ganpll import data, pinn

cv_s = data.load_varactor_cv('small'); cv_l = data.load_varactor_cv('large')

def train_pinn(train, seed):
    st = pinn.init_state(jax.random.PRNGKey(seed), eps=0.6)
    st, _ = pinn.adam_train(pinn.make_loss(train, physics_on=False), st, steps=4000, lr=5e-3, trainable=('phys', 'logA'))
    st, _ = pinn.adam_train(pinn.make_loss(train, lam_mono=1.0, lam_T=0.05, lam_prior=0.01, physics_on=True), st, steps=8000, lr=1e-3)
    return st

def pred(st, d):
    return np.asarray(pinn.cap_fF(st, jnp.asarray(d['V']), jnp.asarray(d['T']), jnp.asarray(d['g'])))

out = {}
# (a) deep ensemble with 300 K held out
train, test = pinn.build_training_arrays(cv_s, cv_l, hold_out_T=300.0)
P = []
for seed in range(5):
    t0 = time.time(); st = train_pinn(train, 100 + seed); P.append(pred(st, test))
    print('ensemble member', seed, 'rmse %.3f%%' % (100 * np.sqrt(np.mean(((P[-1] - test['C_fF']) / test['C_fF']) ** 2))), '%.0fs' % (time.time() - t0), flush=True)
P = np.array(P); mu, sd = P.mean(0), P.std(0, ddof=1)
rel = (mu - test['C_fF']) / test['C_fF']
member = [float(100 * np.sqrt(np.mean(((p - test['C_fF']) / test['C_fF']) ** 2))) for p in P]
out['ensemble'] = dict(member_rmse_pct=member, mean_rmse_pct=float(100 * np.sqrt(np.mean(rel ** 2))),
                       mean_rel_sigma_pct=float(100 * np.mean(sd / mu)), max_rel_sigma_pct=float(100 * np.max(sd / mu)),
                       coverage_2sigma=float(np.mean(np.abs(mu - test['C_fF']) <= 2 * sd + 1e-12)),
                       coverage_2sigma_with_floor=float(np.mean(np.abs(mu - test['C_fF']) <= 2 * np.sqrt(sd ** 2 + (0.0035 * mu) ** 2))))
print('ensemble:', json.dumps(out['ensemble']), flush=True)
np.savez('results/pinn_uq_pred.npz', V=test['V'], T=test['T'], g=test['g'], C=test['C_fF'], mu=mu, sd=sd, members=P)
# (b) leave-one-temperature-out
out['loto'] = {}
for Th in (218.0, 273.0, 300.0, 398.0, 423.0):
    tr, te = pinn.build_training_arrays(cv_s, cv_l, hold_out_T=Th)
    st = train_pinn(tr, 7); p = pred(st, te)
    r = float(100 * np.sqrt(np.mean(((p - te['C_fF']) / te['C_fF']) ** 2)))
    out['loto'][str(int(Th))] = dict(rmse_pct=r, n_test=int(len(te['V'])), extrapolation=Th in (218.0, 423.0))
    print('LOTO', Th, 'rmse %.3f%%' % r, flush=True)
json.dump(out, open('results/pinn_uq.json', 'w'), indent=1)
print('saved results/pinn_uq.json')
