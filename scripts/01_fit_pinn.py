"""Fit the PINN varactor surrogate (two-stage: physics warm-up, then joint
physics + bounded-NN-correction training with physics-informed losses).

Ablations: plain black-box NN (no physics loss, unbounded correction) and
analytic-prior-only fit.  Held-out test: the full 300 K C-V curves (both
devices), never seen in training.  Also reports monotonicity violations on a
dense V-T grid extending beyond the training window (extrapolation physics
check) and the learned physics parameters.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from ganpll import data, pinn, physics

cv_s = data.load_varactor_cv('small'); cv_l = data.load_varactor_cv('large')
train, test = pinn.build_training_arrays(cv_s, cv_l, hold_out_T=300.0)
print('train pts', len(train['V']), 'test pts', len(test['V']))

def rmse(st):
    pred = pinn.cap_fF(st, jnp.asarray(test['V']), jnp.asarray(test['T']), jnp.asarray(test['g']))
    return float(np.sqrt(np.mean(((np.asarray(pred)-test['C_fF'])/test['C_fF'])**2)))*100

def mono_viol(st):
    Vg = jnp.linspace(-0.5, 7.0, 200); worst = 0.0
    for T in (218., 300., 423., 523.):
        c = pinn.cap_unit(st, Vg, jnp.full_like(Vg, T), 0)
        d = np.diff(np.asarray(c))
        worst = max(worst, float(np.maximum(-d, 0).sum() / np.abs(d).sum() * 100))
    return worst

results = {}

# --- PINN: stage 1 physics warm-up, stage 2 joint with physics losses
t0 = time.time()
st = pinn.init_state(jax.random.PRNGKey(0), eps=0.6)
loss1 = pinn.make_loss(train, physics_on=False)
st, h1 = pinn.adam_train(loss1, st, steps=4000, lr=5e-3, trainable=('phys', 'logA'))
loss2 = pinn.make_loss(train, lam_mono=1.0, lam_T=0.05, lam_prior=0.01, physics_on=True)
st, h2 = pinn.adam_train(loss2, st, steps=8000, lr=1e-3)
results['pinn'] = dict(rmse_pct_300K=rmse(st), mono_viol_pct=mono_viol(st),
                       train_s=time.time()-t0, hist=h1+h2)
pinn.save_state(st, 'results/pinn_state.json')
print('PINN: held-out 300K RMSE %.3f%%, mono-viol %.4f%% (%.0fs)' %
      (results['pinn']['rmse_pct_300K'], results['pinn']['mono_viol_pct'], results['pinn']['train_s']))
n, cmin, dc, vth = physics.resolve(st['phys'], 300.0)
results['pinn']['learned_phys'] = dict(
    vth0_V=float(st['phys']['vth0']), dvth_dT_mVK=float(st['phys']['kvt_m']),
    n300=float(n), cratio=float((cmin+dc)/cmin))
print('learned physics @300K:', results['pinn']['learned_phys'])

# --- plain black-box NN ablation
t0 = time.time()
st_nn = pinn.init_state(jax.random.PRNGKey(1), eps=3.0)
loss_nn = pinn.make_loss(train, physics_on=False)
st_nn, _ = pinn.adam_train(loss_nn, st_nn, steps=12000, lr=3e-3, trainable=('net', 'logA'))
results['plain_nn'] = dict(rmse_pct_300K=rmse(st_nn), mono_viol_pct=mono_viol(st_nn),
                           train_s=time.time()-t0)
pinn.save_state(st_nn, 'results/plainnn_state.json')
print('plain NN: RMSE %.3f%%, mono-viol %.2f%%' %
      (results['plain_nn']['rmse_pct_300K'], results['plain_nn']['mono_viol_pct']))

# --- analytic-prior-only ablation
t0 = time.time()
st_an = pinn.init_state(jax.random.PRNGKey(2), eps=0.0)
st_an, _ = pinn.adam_train(pinn.make_loss(train, physics_on=False),
                           st_an, steps=8000, lr=5e-3, trainable=('phys', 'logA'))
results['analytic'] = dict(rmse_pct_300K=rmse(st_an), mono_viol_pct=mono_viol(st_an),
                           train_s=time.time()-t0)
pinn.save_state(st_an, 'results/analytic_state.json')
print('analytic: RMSE %.3f%%, mono-viol %.4f%%' %
      (results['analytic']['rmse_pct_300K'], results['analytic']['mono_viol_pct']))

# --- extrapolation stress test: evaluate all three at 523 K (no data there)
Vg = jnp.linspace(0.0, 6.0, 100); T523 = jnp.full_like(Vg, 523.0)
ext = {}
for name, s in (('pinn', st), ('plain_nn', st_nn), ('analytic', st_an)):
    c = np.asarray(pinn.cap_fF(s, Vg, T523, jnp.ones_like(Vg)))
    ext[name] = dict(cmin_pF=float(c.min()/1e3), cmax_pF=float(c.max()/1e3),
                     negative=bool((c < 0).any()),
                     nonmono_pct=float(np.maximum(-np.diff(c), 0).sum()/np.abs(np.diff(c)).sum()*100))
results['extrap_523K'] = ext
print('523K extrapolation:', json.dumps(ext, indent=1))

json.dump(results, open('results/pinn_results.json', 'w'), indent=1)
print('saved results/pinn_results.json')
