"""Physics-constrained WGAN-GP varactor variability + PLL lock-yield study.

Pipeline
--------
a. Fit the physics prior to each measured C-V curve (10 curves; 20 bootstrap
   refits each under 1% multiplicative noise -> 200 training vectors), train
   the WGAN-GP of ganpll.wgan, save the generator and 500 sampled variants.
b. Sanity metrics: generated vs training marginals of (Vt50 at 300 K,
   Cmax/Cmin at 300 K), means/stds and a KS-style overlap number.
c. Monte-Carlo LOCK YIELD: 300 generated device variants, each pushed through
   the differentiable PLL (physics params of the PINN state replaced, NN
   correction frozen) at corners {218, 300, 423} K with dual-start transients
   (vc0 = 1.0 and 5.5 V, 60 us, dt = 20 ns); a variant passes iff
   |f_end - 50 MHz| < 0.5 MHz at every corner for both starts.  Evaluated for
   the baseline design (None) and the thermally tracked co-design raw_opt_tt.
d. ROBUST REFINEMENT: 60 Adam steps (lr 0.02) from raw_opt_tt on the expected
   adjoint objective over 8 fixed sampled variants (held out from the 300
   evaluation variants; mean-balanced greedy subset for variance reduction),
   temps (218, 300, 423) K, ferr weight 1200.  The refined design is saved
   to results/raw_opt_robust.npy only if its variant yield is at least the
   tracked design's AND it still locks 10/10 nominal corners (218-423 K
   standard list, dual-start, full 100 us transients).
e. Everything saved to results/wgan_results.json.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp, optax
from ganpll import pinn, pll, adjoint, wgan

SEED = 0
np.random.seed(SEED)

pst = pinn.load_state('results/pinn_state.json')
vp = pll.load_vcro('results/vcro_params.json')
circ = pll.default_circuit()
circ['icp'] = json.load(open('results/circuit_cal.json'))['icp']
raw_tt = jnp.asarray(np.load('results/raw_opt_tt.npy'))

PKEYS = wgan.PKEYS
DKEYS = [n for n, _, _ in adjoint.DESIGN_SPACE]
CORNERS = (218.0, 300.0, 423.0)
TEMPS_10 = (218., 233., 253., 273., 300., 323., 348., 373., 398., 423.)
N_EVAL, N_ROBUST_FIT = 300, 8

results, clocks = {}, {}
t_all = time.time()

# ======================================================================
# (a) training set, WGAN training, sampling
# ======================================================================
t0 = time.time()
fits = wgan.fit_training_set(pst, n_boot=20, noise=0.01, seed=SEED)
theta_boot = fits['theta_boot']            # (200, 8)
pv_nom = wgan.phys_vec(pst['phys'])
mu, S, X_real, inflation, offset = wgan.prepare_gan_space(theta_boot, pv_nom)
clocks['fits_s'] = time.time() - t0
print(f"[a] {len(theta_boot)} training vectors from {len(fits['theta_clean'])} "
      f"curve fits in {clocks['fits_s']:.0f}s "
      f"(worst clean fit loss {max(fits['loss_clean']):.2e}, "
      f"vth inflation x{inflation:.2f})")
print(f"[a] discarded nominal-vs-fit-mean offset (see wgan.py docstring): "
      f"{dict(zip(PKEYS, np.round(offset, 3).tolist()))}")

t0 = time.time()
gan, gan_hist = wgan.train_wgan(X_real, pst, mu, S, jax.random.PRNGKey(SEED),
                                critic_steps=3000, n_critic=5, batch=64)
clocks['wgan_train_s'] = time.time() - t0
print(f"[a] WGAN-GP trained (3000 critic steps) in {clocks['wgan_train_s']:.0f}s; "
      f"final W-estimate {gan_hist[-1]['w_est']:.3f}, "
      f"phys penalty {gan_hist[-1]['phys_pen']:.2e}")

samples, acc_rate = wgan.sample_thetas(gan, pst, jax.random.PRNGKey(SEED + 1), 500)
assert np.isfinite(samples).all(), 'NaN/inf in generated samples'
wgan.save_gan(gan, 'results/wgan_state.json',
              extra=dict(inflation=inflation, acceptance_rate=acc_rate,
                         seed=SEED, critic_steps=3000))
np.save('results/wgan_samples.npy', samples)
print(f"[a] 500 samples saved (physics-rejection acceptance rate {acc_rate:.3f})")

# ======================================================================
# (b) sanity metrics: generated vs training marginals
# ======================================================================
def ks_stat(a, b):
    """Two-sample Kolmogorov-Smirnov statistic (no scipy)."""
    a, b = np.sort(a), np.sort(b)
    allv = np.concatenate([a, b])
    ca = np.searchsorted(a, allv, side='right') / len(a)
    cb = np.searchsorted(b, allv, side='right') / len(b)
    return float(np.max(np.abs(ca - cb)))

vt_tr, cr_tr = wgan.summary_stats(theta_boot, pst)
vt_ge, cr_ge = wgan.summary_stats(samples, pst)
# training distribution the GAN is actually asked to match: deltas about the
# fit mean, vth inflated, re-centered on the nominal parameter vector
d_tr = theta_boot - theta_boot.mean(0)
d_tr[:, PKEYS.index('vth0')] *= inflation
theta_matched = np.asarray(pv_nom) + d_tr
vt_tri, cr_tri = wgan.summary_stats(theta_matched, pst)

stats = dict(
    train_raw=dict(vt50_mean=float(vt_tr.mean()), vt50_std=float(vt_tr.std()),
                   cratio_mean=float(cr_tr.mean()), cratio_std=float(cr_tr.std())),
    train_matched=dict(vt50_mean=float(vt_tri.mean()), vt50_std=float(vt_tri.std()),
                       cratio_mean=float(cr_tri.mean()), cratio_std=float(cr_tri.std())),
    nominal_offset=dict(zip(PKEYS, offset.tolist())),
    generated=dict(vt50_mean=float(vt_ge.mean()), vt50_std=float(vt_ge.std()),
                   cratio_mean=float(cr_ge.mean()), cratio_std=float(cr_ge.std())),
    ks=dict(vt50_vs_raw=ks_stat(vt_ge, vt_tr),
            vt50_vs_matched=ks_stat(vt_ge, vt_tri),
            cratio_vs_raw=ks_stat(cr_ge, cr_tr),
            cratio_vs_matched=ks_stat(cr_ge, cr_tri)),
    # overlap vs the matched (re-centered + vth-inflated) training set --
    # the distribution the GAN is actually trained on
    overlap=dict(vt50=1.0 - ks_stat(vt_ge, vt_tri),
                 cratio=1.0 - ks_stat(cr_ge, cr_tri)),
    vth_inflation=inflation, acceptance_rate=acc_rate,
    raw_param_train_mean=dict(zip(PKEYS, theta_boot.mean(0).tolist())),
    raw_param_train_std=dict(zip(PKEYS, theta_boot.std(0).tolist())),
    raw_param_gen_mean=dict(zip(PKEYS, samples.mean(0).tolist())),
    raw_param_gen_std=dict(zip(PKEYS, samples.std(0).tolist())),
)
results['stats'] = stats
print(f"[b] Vt50@300K: train(raw) {vt_tr.mean():.3f}+-{vt_tr.std():.3f} V, "
      f"train(matched) {vt_tri.mean():.3f}+-{vt_tri.std():.3f} V, "
      f"generated {vt_ge.mean():.3f}+-{vt_ge.std():.3f} V")
print(f"[b] Cmax/Cmin@300K: train(raw) {cr_tr.mean():.3f}+-{cr_tr.std():.3f}, "
      f"train(matched) {cr_tri.mean():.3f}+-{cr_tri.std():.3f}, "
      f"generated {cr_ge.mean():.3f}+-{cr_ge.std():.3f}")
print(f"[b] KS overlap (1-KS): Vt50 {stats['overlap']['vt50']:.3f}, "
      f"Cratio {stats['overlap']['cratio']:.3f}")

# ======================================================================
# (c) lock yield: 300 variants x {baseline, tracked} x 3 corners x 2 starts
# ======================================================================
def _fend_one(pv, dv, T, vc0):
    pstv = dict(pst)
    pstv['phys'] = {k: pv[i] for i, k in enumerate(PKEYS)}
    des = {k: dv[i] for i, k in enumerate(DKEYS)}
    tr = pll.loop_transient(vp, pstv, circ, T, design=des, vc0=vc0,
                            t_end=60e-6, dt=20e-9)
    return jnp.mean(tr['f'][-600:])

fend_batch = jax.jit(jax.vmap(_fend_one, in_axes=(0, None, None, None)))

def design_vec(design):
    if design is None:  # baseline: all scales 1, additive offsets 0
        return jnp.asarray([1.0] * 9 + [0.0, 0.0])
    return jnp.stack([jnp.asarray(design[k], dtype=jnp.float64) for k in DKEYS])

def yield_eval(dv, thetas, label):
    pvs = jnp.asarray(thetas)
    ok = np.ones(len(thetas), bool)
    fends = {}
    for T in CORNERS:
        for vc0 in (1.0, 5.5):
            f = np.asarray(fend_batch(pvs, dv, T, vc0))
            assert np.isfinite(f).all(), f'NaN f_end at {label} T={T} vc0={vc0}'
            ok &= np.abs(f - 50e6) < 0.5e6
            fends[f'{T:.0f}K_vc{vc0}'] = f
    y = 100.0 * ok.mean()
    per_corner = {f'{T:.0f}K': float(100.0 * np.mean(
        (np.abs(fends[f"{T:.0f}K_vc1.0"] - 50e6) < 0.5e6) &
        (np.abs(fends[f"{T:.0f}K_vc5.5"] - 50e6) < 0.5e6))) for T in CORNERS}
    print(f"[c] {label}: yield {y:.1f}% ({int(ok.sum())}/{len(thetas)}); "
          f"per-corner pass % {per_corner}")
    return y, ok, per_corner

eval_thetas = samples[:N_EVAL]
t0 = time.time()
d_tt = adjoint.raw_to_design(raw_tt)
y_base, ok_base, pc_base = yield_eval(design_vec(None), eval_thetas, 'baseline')
y_tt, ok_tt, pc_tt = yield_eval(design_vec(d_tt), eval_thetas, 'tracked (tt)')
clocks['yield_base_tt_s'] = time.time() - t0

# ======================================================================
# (d) robust refinement on 8 held-out variants
# ======================================================================
t0 = time.time()
# 8 fixed sampled variants, held out of the 300-variant evaluation set.
# Variance reduction: with only 8 Monte-Carlo variants the subset mean can sit
# a fair fraction of a sigma away from the nominal device, which drags the
# refined design off nominal frequency; greedily pick the 8-sample subset of
# the held-out pool whose mean best matches the nominal parameter vector
# (deterministic, standardized greedy selection).
pool = samples[N_EVAL:]
sd_all = samples.std(0)
sel, avail = [], list(range(len(pool)))
for _ in range(N_ROBUST_FIT):
    scores = [(np.linalg.norm((pool[sel + [i]].mean(0) - np.asarray(pv_nom)) / sd_all), i)
              for i in avail]
    _, bi = min(scores)
    sel.append(bi); avail.remove(bi)
phys8 = jnp.asarray(pool[sel])
print(f"[d] robust-fit variants: pool indices {[N_EVAL + i for i in sel]}, "
      f"mean offset (in sigmas) "
      f"{np.round((pool[sel].mean(0) - np.asarray(pv_nom)) / sd_all, 2).tolist()}")

# Emphasize frequency centering (ferr weight 1200 vs default 300): with
# variant scatter present, a softer ferr lets the optimizer trade ~1 MHz of
# nominal frequency offset for hot-corner robustness, which violates the
# nominal 10-corner gate; the stiffer weight keeps f_end centered at 50 MHz.
ROBUST_WEIGHTS = dict(ferr=1200.0)

def _one_variant_obj(pv, raw):
    pstv = dict(pst)
    pstv['phys'] = {k: pv[i] for i, k in enumerate(PKEYS)}
    obj, _ = adjoint.make_objective(vp, pstv, circ, CORNERS, weights=ROBUST_WEIGHTS)
    return obj(raw)

def robust_obj(raw):
    return jnp.mean(jax.vmap(_one_variant_obj, in_axes=(0, None))(phys8, raw))

grad_fn = jax.jit(jax.value_and_grad(robust_obj))
opt = optax.adam(0.02)
raw, ostate, trace = raw_tt, opt.init(raw_tt), []
for i in range(60):
    v, g = grad_fn(raw)
    assert np.isfinite(float(v)), 'NaN in robust objective'
    up, ostate = opt.update(g, ostate)
    raw = optax.apply_updates(raw, up)
    trace.append(float(v))
    if i % 10 == 0 or i == 59:
        print(f"[d] robust step {i:3d}: E[J] = {float(v):.5f}")
raw_rob = raw
clocks['robust_refine_s'] = time.time() - t0
print(f"[d] refinement {clocks['robust_refine_s']:.0f}s: "
      f"E[J] {trace[0]:.4f} -> {trace[-1]:.4f}")

# ======================================================================
# (d cont.) robust yield on the same 300 variants + nominal 10-corner gate
# ======================================================================
d_rob = adjoint.raw_to_design(raw_rob)
t0 = time.time()
y_rob, ok_rob, pc_rob = yield_eval(design_vec(d_rob), eval_thetas, 'robust')
clocks['yield_robust_s'] = time.time() - t0

# nominal 10-corner gate: full 100 us transients, nominal device
fend100 = jax.jit(lambda dv, T, vc0: jnp.mean(pll.loop_transient(
    vp, pst, circ, T, design={k: dv[i] for i, k in enumerate(DKEYS)},
    vc0=vc0, t_end=100e-6, dt=10e-9)['f'][-2000:]))

def nominal_10corner(dv, label):
    locks = {}
    for T in TEMPS_10:
        f1, f2 = float(fend100(dv, T, 1.0)), float(fend100(dv, T, 5.5))
        locks[f'{T:.0f}K'] = bool(abs(f1 - 50e6) < 0.5e6 and abs(f2 - 50e6) < 0.5e6)
    n = sum(locks.values())
    print(f"[d] nominal 10-corner lock ({label}): {n}/10 {locks}")
    return n, locks

t0 = time.time()
n_tt10, locks_tt10 = nominal_10corner(design_vec(d_tt), 'tracked')
n_rob10, locks_rob10 = nominal_10corner(design_vec(d_rob), 'robust')
clocks['nominal_10corner_s'] = time.time() - t0

robust_saved = bool(y_rob >= y_tt and n_rob10 == 10)
if robust_saved:
    np.save('results/raw_opt_robust.npy', np.asarray(raw_rob))
    print(f"[d] SAVED results/raw_opt_robust.npy "
          f"(yield {y_rob:.1f}% >= {y_tt:.1f}%, nominal 10/10)")
else:
    print(f"[d] robust design NOT saved (yield {y_rob:.1f}% vs tt {y_tt:.1f}%, "
          f"nominal {n_rob10}/10) -- keeping raw_opt_tt as the design of record")

# ======================================================================
# (e) results file + generated curves for figures
# ======================================================================
Vg = np.linspace(0.0, 6.0, 41)
curves300 = np.asarray(jax.vmap(
    lambda pv: wgan.variant_curve(pst, pv, 300.0))(jnp.asarray(samples[:30])))

results['yields'] = dict(baseline=y_base, tracked=y_tt, robust=y_rob,
                         n_variants=N_EVAL,
                         per_corner=dict(baseline=pc_base, tracked=pc_tt,
                                         robust=pc_rob))
results['robust'] = dict(saved=robust_saved, trace=trace,
                         n_fit_variants=N_ROBUST_FIT, lr=0.02, steps=60,
                         fit_variant_indices=[N_EVAL + i for i in sel],
                         weights=ROBUST_WEIGHTS,
                         nominal_10corner=dict(tracked=n_tt10, robust=n_rob10,
                                               tracked_detail=locks_tt10,
                                               robust_detail=locks_rob10),
                         design_tt={k: float(v) for k, v in d_tt.items()},
                         design_robust={k: float(v) for k, v in d_rob.items()})
results['gan_hist'] = gan_hist
results['fit_meta'] = fits['meta']
results['curves_300K'] = dict(V=Vg.tolist(), C_unit=curves300.tolist())
clocks['total_s'] = time.time() - t_all
results['clocks'] = clocks
json.dump(results, open('results/wgan_results.json', 'w'), indent=1, default=float)

print('\n================ SUMMARY ================')
print(f"training vectors: {len(theta_boot)} (10 curve fits x 20 bootstraps); "
      f"vth inflation x{inflation:.2f}")
print(f"generated Vt50@300K {vt_ge.mean():.3f}+-{vt_ge.std():.3f} V "
      f"(train matched {vt_tri.mean():.3f}+-{vt_tri.std():.3f}); "
      f"Cratio {cr_ge.mean():.2f}+-{cr_ge.std():.2f} (train matched {cr_tri.mean():.2f}+-{cr_tri.std():.2f})")
print(f"KS overlap: Vt50 {stats['overlap']['vt50']:.3f}, Cratio {stats['overlap']['cratio']:.3f}")
print(f"LOCK YIELD over {N_EVAL} variants (3 corners x dual start): "
      f"baseline {y_base:.1f}%  tracked {y_tt:.1f}%  robust {y_rob:.1f}%")
print(f"robust design saved: {robust_saved}")
print(f"wall clocks (s): { {k: round(v, 1) for k, v in clocks.items()} }")
print('saved: results/wgan_state.json, results/wgan_samples.npy, '
      'results/wgan_results.json' + (', results/raw_opt_robust.npy' if robust_saved else ''))
