"""Adjoint-gradient co-design of the GaN PLL across temperature.

Optimizes 10 bounded design parameters (CP current, loop-filter R/C scales,
LPF varactor sizes, VCRO drive width, fixed delay cap, delay-varactor size,
varactor bias offset) against transient lock quality + K_VCRO-monotonicity +
power at six temperature corners spanning 218-423 K.

Also produces:
  * adjoint vs central-finite-difference gradient cost and agreement,
  * a random-search baseline with an identical evaluation budget,
  * before/after evaluation at 14 verification temperatures (218-523 K).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
from ganpll import pinn, pll, adjoint

pst = pinn.load_state('results/pinn_state.json')
vp = pll.load_vcro('results/vcro_params.json')
circ = pll.default_circuit()
circ['icp'] = json.load(open('results/circuit_cal.json'))['icp']

TEMPS_OPT = (218.0, 258.0, 300.0, 348.0, 398.0, 423.0)
TEMPS_VERIF = (218., 233., 253., 273., 300., 323., 348., 373., 398., 423., 448., 473., 498., 523.)

obj, per_temp = adjoint.make_objective(vp, pst, circ, TEMPS_OPT)
raw0 = adjoint.init_raw()
results = {}

# ---------------- gradient cost: adjoint vs finite differences ---------
gf = jax.jit(jax.value_and_grad(obj))
of = jax.jit(obj)
_ = gf(raw0); _ = of(raw0)  # compile
t0 = time.time(); v0 = float(of(raw0)); t_fwd = time.time() - t0
t0 = time.time(); val, g_adj = gf(raw0); np.asarray(g_adj); t_adj = time.time() - t0
t0 = time.time(); g_fd = adjoint.fd_gradient(obj, raw0); t_fd = time.time() - t0
cos = float(np.dot(g_fd, np.asarray(g_adj)) /
            (np.linalg.norm(g_fd) * np.linalg.norm(np.asarray(g_adj)) + 1e-30))
results['gradient_cost'] = dict(t_forward_s=t_fwd, t_adjoint_s=t_adj, t_fd_s=t_fd,
                                n_params=len(adjoint.DESIGN_SPACE),
                                adjoint_overhead_x=t_adj/t_fwd, fd_cost_x=t_fd/t_fwd,
                                speedup_x=t_fd/t_adj, cosine_agreement=cos)
print('gradient: fwd %.2fs, adjoint %.2fs (%.1fx fwd), FD %.2fs (%.1fx fwd), '
      'speedup %.1fx, cosine %.4f' %
      (t_fwd, t_adj, t_adj/t_fwd, t_fd, t_fd/t_fwd, t_fd/t_adj, cos))

# ---------------- optimization ----------------------------------------
print('J(baseline) =', v0)
raw_opt, hist = adjoint.optimize(obj, raw0, steps=200, lr=0.05, log_every=20)
design = {k: float(v) for k, v in adjoint.raw_to_design(raw_opt).items()}
results['optimization'] = dict(J0=v0, J_final=hist[-1][1], hist=hist, design=design)
print('optimized design:', json.dumps(design, indent=1))

# random-search baseline, same wall-clock-equivalent budget (200 grad steps
# ~ 200*(t_adj/t_fwd) forward evals)
budget = int(200 * t_adj / t_fwd) + 1
rs_raw, rs_best, rs_hist = adjoint.random_search(obj, budget)
results['random_search'] = dict(budget=budget, best=rs_best,
                                hist=[(i, float(b)) for i, b in rs_hist[::10]])
print(f'random search ({budget} evals): best J = {rs_best:.4f} '
      f'vs adjoint {hist[-1][1]:.4f}')

# ---------------- before/after verification at 14 temperatures ---------
def evaluate(design, label):
    rows = []
    for T in TEMPS_VERIF:
        tr = pll.loop_transient(vp, pst, circ, T, design=design, vc0=1.0,
                                t_end=100e-6, dt=10e-9)
        f = np.asarray(tr['f']); vc = np.asarray(tr['vc'])
        f_end = float(np.mean(f[-2000:]))
        err = np.abs(f - 50e6) / 50e6
        ok = err < 0.01
        idx = np.where(~ok)[0]
        settle = (idx[-1]+1) * 10e-9 * 1e6 if len(idx) and idx[-1] < len(ok)-1 else np.inf
        vce = float(vc[-1])
        ms = pll.stability_margins(vp, pst, circ, T, vce, design)
        jit = pll.closed_loop_jitter(vp, pst, circ, T, vce, design)
        # tuning range + worst normalized Kvco inversion over window
        Vg = jnp.linspace(1.0, 5.5, 60)
        fr = np.asarray(pll.f_vcro(vp, pst, Vg, T, design))
        dfdv = np.gradient(fr, np.asarray(Vg))
        rows.append(dict(T_K=T, f_end_MHz=f_end/1e6, lock=bool(abs(f_end-50e6)/50e6 < 0.01),
                         settle_us=float(settle), pm_deg=ms['pm_deg'],
                         f_ugb_kHz=ms['f_ugb_Hz']/1e3 if np.isfinite(ms['f_ugb_Hz']) else None,
                         jitter_ns=jit, vc_end=vce,
                         f_min_MHz=float(fr.min()/1e6), f_max_MHz=float(fr.max()/1e6),
                         kvco_min_MHzV=float(dfdv.min()/1e6)))
        print(f"  [{label}] T={T:.0f}K f_end={f_end/1e6:6.2f} MHz lock={rows[-1]['lock']} "
              f"settle={settle:6.1f}us PM={ms['pm_deg']:.0f} jit={jit:.2f}ns "
              f"range=[{fr.min()/1e6:.0f},{fr.max()/1e6:.0f}]")
    return rows

print('baseline verification:')
base_rows = evaluate(None, 'base')
print('optimized verification:')
opt_rows = evaluate(adjoint.raw_to_design(raw_opt), 'opt')
results['verify_baseline'] = base_rows
results['verify_optimized'] = opt_rows

nb = sum(r['lock'] for r in base_rows if 218 <= r['T_K'] <= 423)
no = sum(r['lock'] for r in opt_rows if 218 <= r['T_K'] <= 423)
print(f'true 50MHz lock corners in 218-423K: baseline {nb}/10, optimized {no}/10')

json.dump(results, open('results/adjoint_results.json', 'w'), indent=1, default=float)
np.save('results/raw_opt.npy', np.asarray(raw_opt))
print('saved results/adjoint_results.json')
