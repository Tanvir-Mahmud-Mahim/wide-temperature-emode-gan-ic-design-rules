"""Calibrate the differentiable PLL against Cadence Spectre data and validate.

Step 1: fit the VCRO delay expression (+ bounded NN residual) to the five
        digitized Spectre tuning curves (218-423 K).
Step 2: validate the composed PINN->VCRO model at the Spectre transient
        corners up to 423 K: predict f from the simulated steady-state control
        voltage and compare with the simulated plateau frequency.  The 253 K
        corner is never used in any fitting step (blind test).
Step 3: calibrate the charge-pump current to the 300 K lock transient and
        compare model vs Cadence Vc(t)/f(t).
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp, optax
from ganpll import data, pinn, pll

pst = pinn.load_state('results/pinn_state.json')
tuning = data.load_vcro_tuning()
corners = data.load_corners()
results = {}

# ---------------- Step 1: fit tuning surface --------------------------
V_all = np.concatenate([t['V'] for t in tuning])
T_all = np.concatenate([np.full(len(t['V']), t['T']) for t in tuning])
f_all = np.concatenate([t['f_MHz'] for t in tuning]) * 1e6
Vj, Tj, fj = map(jnp.asarray, (V_all, T_all, f_all))

vp0 = pll.init_vcro_params()

# semi-supervised corner anchors: transient corners at temperatures with no
# tuning curve.  Three corners enter the fit; 253 K is withheld as a blind test.
CORNER_TRAIN_K = (233.0, 323.0, 373.0)
EXCLUDE_K = (348.0,)  # post-lock-loss collapsed state: not a steady tuning point
T_MAX = 423.0  # specification ceiling: nothing above is fitted or reported
corners = [r for r in corners if r['T_K'] <= T_MAX + 0.5 and r['T_K'] not in EXCLUDE_K]
anchors = []
for r in corners:
    if r['T_K'] in CORNER_TRAIN_K and 'f_lock_MHz' in r:
        tw, vw = r['t_Vc'], r['v_Vc']
        t0_, t1_ = r['plateau_window']
        sel = (tw >= t0_) & (tw <= t1_)
        anchors.append((jnp.asarray(vw[sel]), r['T_K'], r['f_lock_MHz'] * 1e6))

# T-smoothness collocation for the residual gamma in the data-sparse
# low-voltage region (the 300 K sweep only covers 3.05-5.81 V, so gamma is
# unconstrained there and can invent phantom plateaus between neighbors)
Vc_lowv, Tc_lowv = jnp.meshgrid(jnp.linspace(0.5, 3.2, 12),
                                jnp.linspace(240.0, 360.0, 13))
Vc_lowv, Tc_lowv = Vc_lowv.ravel(), Tc_lowv.ravel()

def tuning_loss(vp):
    fm = pll.f_vcro(vp, pst, Vj, Tj)
    l = jnp.mean((jnp.log(fm) - jnp.log(fj)) ** 2)
    for vw, Tk, fmeas in anchors:
        fpred = jnp.mean(pll.f_vcro(vp, pst, vw, Tk))
        l = l + 0.05 * (jnp.log(fpred) - jnp.log(fmeas)) ** 2
    hT = 10.0
    g0 = pll._resid(vp, Vc_lowv, Tc_lowv)
    gp = pll._resid(vp, Vc_lowv, Tc_lowv + hT)
    gm = pll._resid(vp, Vc_lowv, Tc_lowv - hT)
    l = l + 3.0 * jnp.mean(((gp - 2 * g0 + gm) / hT**2) ** 2) * 1e4
    return l

opt = optax.adam(3e-3)
vstate = vp0
ostate = opt.init(vstate)

@jax.jit
def step(vs, os_):
    l, g = jax.value_and_grad(tuning_loss)(vs)
    up, os_ = opt.update(g, os_)
    return optax.apply_updates(vs, up), os_, l

t0 = time.time()
for i in range(8000):
    vstate, ostate, l = step(vstate, ostate)
    if i % 2000 == 0:
        print(f'  tuning fit iter {i}: loss {float(l):.5f}')
vp = vstate
fm = np.asarray(pll.f_vcro(vp, pst, Vj, Tj))
rel = np.abs(fm - f_all) / f_all
results['tuning_fit'] = dict(rmse_pct=float(np.sqrt(np.mean(rel**2)))*100,
                             mean_abs_pct=float(np.mean(rel))*100,
                             fit_s=time.time()-t0)
print('tuning-surface fit: RMSE %.2f%%, MAE %.2f%%' %
      (results['tuning_fit']['rmse_pct'], results['tuning_fit']['mean_abs_pct']))
pll.save_vcro(vp, 'results/vcro_params.json')

# ---------------- Step 2: corner validation ------------------
rows = []
for r in corners:
    if 'f_lock_MHz' not in r or r['Vc_mean'] is None:
        continue
    # ripple-aware: average model frequency over the measured Vc(t) samples
    # taken from the same plateau window used for the frequency measurement
    tw = r['t_Vc']; vw = r['v_Vc']
    t0_, t1_ = r['plateau_window']
    sel = (tw >= t0_) & (tw <= t1_)
    if sel.sum() < 5:
        sel = tw > 0.6 * tw[-1]
    fpred = float(np.mean(np.asarray(pll.f_vcro(vp, pst, jnp.asarray(vw[sel]), r['T_K'])))) / 1e6
    r['vc_plateau'] = float(np.mean(vw[sel]))
    rows.append(dict(T_K=r['T_K'], vc=r.get('vc_plateau', r['Vc_mean']), f_meas=r['f_lock_MHz'],
                     f_pred=fpred, err_pct=100*(fpred-r['f_lock_MHz'])/r['f_lock_MHz'],
                     in_fit_set=(r['T_K'] in (218., 273., 300., 398., 423.)
                                 or r['T_K'] in CORNER_TRAIN_K)))
for row in rows:
    print(f"  T={row['T_K']:.0f}K Vc={row['vc']:.2f}V meas {row['f_meas']:.1f} pred {row['f_pred']:.1f} MHz ({row['err_pct']:+.1f}%){' *unseen*' if not row['in_fit_set'] else ''}")
unseen = [abs(r['err_pct']) for r in rows if not r['in_fit_set']]
seen = [abs(r['err_pct']) for r in rows if r['in_fit_set']]
results['corner_validation'] = dict(rows=rows,
    mae_pct_seen=float(np.mean(seen)), mae_pct_unseen=float(np.mean(unseen)))
print('corner validation MAE: seen %.1f%%, unseen %.1f%%' %
      (results['corner_validation']['mae_pct_seen'], results['corner_validation']['mae_pct_unseen']))

# ---------------- Step 3: 300 K transient calibration ------------------
c300 = [r for r in corners if r['name'] == 'norm_27'][0]
circ = pll.default_circuit()
best = None
for icp in (2e-6, 4e-6, 8e-6, 16e-6, 32e-6):
    circ_i = dict(circ, icp=icp)
    tr = pll.loop_transient(vp, pst, circ_i, 300.0, vc0=0.5)
    m = pll.lock_metrics(tr)
    # settle: first index where |f-50M|<1% and stays
    err = np.abs(np.asarray(tr['f']) - 50e6) / 50e6
    ok = err < 0.01
    idx = np.where(~ok)[0]
    settle = (idx[-1] + 1) * pll.DT * 1e6 if len(idx) and idx[-1] < len(ok) - 1 else np.inf
    print(f'  icp={icp*1e6:.0f}uA: f_end {float(m["f_end"])/1e6:.2f} MHz, settle {settle:.1f} us')
    if best is None or (np.isfinite(settle) and settle < best[1]):
        best = (icp, settle)
# measured settle time at 300 K from the digitized lock transient (Fig. F9)
import glob
f9 = sorted(glob.glob('data/digitized/F9_axes_1_c*_300_K.csv'))
d9 = np.loadtxt(f9[0], delimiter=',', skiprows=1)
i9 = np.argsort(d9[:, 0]); t9, ff9 = d9[i9, 0], d9[i9, 1]  # us, MHz
merr = np.abs(ff9 - 50.0) / 50.0
bad = np.where(merr > 0.02)[0]
settle_meas = t9[min(bad[-1] + 1, len(t9) - 1)] if len(bad) else 0.0
e_t, e_f = t9 * 1e-6, ff9 * 1e6
# pick icp whose settle time is closest to measured
cands = []
for icp in (2e-6, 4e-6, 8e-6, 16e-6, 32e-6):
    tr = pll.loop_transient(vp, pst, dict(circ, icp=icp), 300.0, vc0=0.5)
    err = np.abs(np.asarray(tr['f']) - 50e6) / 50e6
    ok = err < 0.01
    idx = np.where(~ok)[0]
    settle = (idx[-1] + 1) * pll.DT * 1e6 if len(idx) and idx[-1] < len(ok) - 1 else np.inf
    cands.append((abs(settle - settle_meas) if np.isfinite(settle) else 1e9, icp, settle))
cands.sort()
icp_cal = cands[0][1]
circ['icp'] = icp_cal
results['transient_cal'] = dict(icp_uA=icp_cal*1e6, settle_model_us=cands[0][2],
                                settle_meas_us=float(settle_meas))
print(f'calibrated icp = {icp_cal*1e6:.0f} uA; settle model {cands[0][2]:.1f} vs meas {settle_meas:.1f} us')
json.dump({'icp': icp_cal}, open('results/circuit_cal.json', 'w'))

# save model traces for figures
tr = pll.loop_transient(vp, pst, circ, 300.0, vc0=0.5)
np.savez('results/transient_300K.npz', t=np.asarray(tr['t']), vc=np.asarray(tr['vc']),
         f=np.asarray(tr['f']), meas_t=e_t, meas_f=e_f,
         meas_tvc=c300['t_Vc'], meas_vvc=c300['v_Vc'])

# small-signal + jitter at 300 K
ms = pll.stability_margins(vp, pst, circ, 300.0, float(np.asarray(tr['vc'])[-1]))
jit = pll.closed_loop_jitter(vp, pst, circ, 300.0, float(np.asarray(tr['vc'])[-1]))
results['smallsignal_300K'] = dict(**ms, jitter_ns=jit)
print('300K small-signal:', results['smallsignal_300K'])

json.dump(results, open('results/calibration_results.json', 'w'), indent=1, default=float)
print('saved results/calibration_results.json')
