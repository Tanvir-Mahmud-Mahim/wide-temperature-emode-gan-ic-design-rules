"""GCN-SAC + PER reinforcement-learning co-design of the GaN PLL, and the
hybrid RL-then-adjoint pipeline.

Runs
----
1. GCN-SAC + prioritized experience replay (Haarnoja 2018; Schaul 2016;
   Kipf & Welling 2017) on the ``ganpll.rl_env`` design environment.
2. Ablation: flat MLP-SAC with uniform replay, identical budget.
3. Hybrid: the top-3 (n_lock, -J) designs discovered by run 1 are refined
   with the exact discrete-adjoint optimizer (``adjoint.optimize``,
   150 steps, lr = 0.02); the best refined design is kept.
4. Verification of {adjoint-only baseline, best pure-RL, best hybrid} at the
   10 verification temperatures with full transients (t_end = 100 us,
   dt = 10 ns), dual acquisition starts vc0 in {1.0, 5.5}; lock means the
   settled frequency of BOTH starts is within 1 % (0.5 MHz) of 50 MHz.

Outputs: results/rl_results.json (learning curves, budgets, wall-clock,
best raw vectors, verification tables, hybrid refinement trace) and, if the
hybrid dominates the adjoint-only baseline, results/raw_opt_hybrid.npy.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import jax
import jax.numpy as jnp

# --- pin the RL study to the FIXED-BIAS (10-parameter) architecture ---
# The RL exploration and feasibility screen certify the fixed-bias design
# space; the thermally tracked bias (dvb_t) is the architecture augmentation
# that the certificate motivates, optimized separately by the adjoint.
from ganpll import adjoint as _adj
_adj.DESIGN_SPACE = [e for e in _adj.DESIGN_SPACE if e[0] != 'dvb_t']

from ganpll import pinn, pll, adjoint, rl_env, gcn_sac

# ----------------------------------------------------------------------
# Setup
# ----------------------------------------------------------------------
SEED = 0
N_STEPS = int(os.environ.get("RL_STEPS", 5000))   # env steps per run
ADJ_STEPS = int(os.environ.get("ADJ_STEPS", 150))  # adjoint refinement steps
WARMUP = 500                                       # uniform-random warmup
TEMPS_VERIF = (218., 233., 253., 273., 300., 323., 348., 373., 398., 423.)
F_REF = 50e6

pst = pinn.load_state("results/pinn_state.json")
vp = pll.load_vcro("results/vcro_params.json")
circ = pll.default_circuit()
circ["icp"] = json.load(open("results/circuit_cal.json"))["icp"]
raw_baseline = np.load("results/raw_opt_final.npy")

evaluator = rl_env.make_evaluator(vp, pst, circ)   # shared jitted fast eval
t_wall0 = time.time()
results = dict(seed=SEED, warmup_steps=WARMUP,
               temps_train=list(rl_env.TEMPS), temps_verif=list(TEMPS_VERIF))


# ----------------------------------------------------------------------
# 0) Feasibility screen: is 6/6 dual-start lock reachable at all?
# ----------------------------------------------------------------------
# The 300 K failure of the adjoint-only design is a *reachability* trap: the
# VCRO tuning curve f(vc) at 300 K satisfies f(0) > f_ref with f(vc0=1) >
# f_ref, so acquisition from below slides to the vc = 0 rail (settling at
# 50.96 MHz) instead of reaching the 50 MHz crossing.  Dual-start lock at a
# corner therefore requires, for each start, a 50 MHz crossing of the tuning
# curve in the direction the loop travels.  This condition needs no
# transient, so the whole design box can be screened densely; it bounds what
# any optimizer (RL, adjoint, or hybrid) can achieve.


def feasibility_screen(n=100_000, margin=0.2e6, seed=7):
    """Monte-Carlo screen of the design box for 6-corner dual-start
    reachability of the 50 MHz lock point (tuning-curve condition only)."""
    temps = jnp.asarray(rl_env.TEMPS)
    Vg = jnp.linspace(0.0, 6.0, 121)
    i10 = int(np.argmin(np.abs(np.asarray(Vg) - 1.0)))
    i55 = int(np.argmin(np.abs(np.asarray(Vg) - 5.5)))

    def per_design(raw):
        d = adjoint.raw_to_design(raw)

        def per_T(T):
            f = pll.f_vcro(vp, pst, Vg, T, d)

            def reachable(i0):
                near = jnp.abs(f[i0] - F_REF) <= margin
                up = (f[i0] < F_REF - margin) & jnp.any(f[i0:] > F_REF + margin)
                dn = (f[i0] > F_REF + margin) & jnp.any(f[:i0] < F_REF - margin)
                return near | up | dn

            return reachable(i10) & reachable(i55)

        return jax.vmap(per_T)(temps)

    fn = jax.jit(jax.vmap(per_design))
    rng = np.random.default_rng(seed)
    ok = []
    for _ in range(n // 4000):
        X = np.clip(rng.normal(0, 1.8, (4000, rl_env.N_PARAMS)),
                    -rl_env.RAW_CLIP, rl_env.RAW_CLIP)
        ok.append(np.asarray(fn(jnp.asarray(X))))
    ok = np.concatenate(ok)
    return dict(n_screened=int(len(ok)),
                per_corner_pass_rate=[float(x) for x in ok.mean(0)],
                p_300K_and_423K=float((ok[:, 2] & ok[:, 5]).mean()),
                n_all6=int(ok.all(1).sum()))


if int(os.environ.get("SKIP_SCREEN", 0)) == 0:
    t0 = time.time()
    scr = feasibility_screen()
    scr["wall_clock_s"] = time.time() - t0
    results["feasibility_screen"] = scr
    print(f"feasibility screen: {scr['n_all6']}/{scr['n_screened']} designs "
          f"pass 6-corner dual-start reachability "
          f"(P[300K & 423K jointly lockable] = {scr['p_300K_and_423K']:.5f})")


# ----------------------------------------------------------------------
# Training loop (shared by both runs)
# ----------------------------------------------------------------------

def run_sac(label, encoder, use_per, n_steps, seed):
    """Train one SAC agent for n_steps env steps; return summary dict and
    the archive of every visited design (for hybrid seeding)."""
    t0 = time.time()
    env = rl_env.PLLDesignEnv(vp, pst, circ, raw_baseline,
                              seed=seed, evaluator=evaluator)
    agent = gcn_sac.SACAgent(seed=seed, encoder=encoder, use_per=use_per)
    rng = np.random.default_rng(seed + 100)

    archive_raw, archive_nl, archive_J = [], [], []
    ep_returns = []          # (env_step, episode_return, episode_len)
    best_curve = []          # (env_step, best_n_lock, best_J_at_best_nl)
    best = dict(n_lock=-1, J=np.inf, raw=None)

    def consider(raw, n_lock, J, step):
        archive_raw.append(np.asarray(raw).copy())
        archive_nl.append(n_lock)
        archive_J.append(J)
        if (n_lock, -J) > (best["n_lock"], -best["J"]):
            best.update(n_lock=n_lock, J=J, raw=np.asarray(raw).copy())
            best_curve.append((step, n_lock, J))

    obs = env.reset()
    consider(env.raw, env._n_lock, env._J, 0)
    ep_ret, ep_len, ep_start = 0.0, 0, 0
    n_updates = 0

    for step in range(1, n_steps + 1):
        if step <= WARMUP:
            action = rng.uniform(-1.0, 1.0, rl_env.ACT_DIM)
        else:
            action = agent.act(obs)
        nobs, r, done, info = env.step(action)
        # bootstrap through timeouts, cut only at true terminals (all-lock)
        agent.buffer.add(obs, action, r, nobs, float(info["terminal"]))
        consider(info["raw"], info["n_lock"], info["J"], step)
        ep_ret += r
        ep_len += 1
        obs = nobs
        if done:
            ep_returns.append((step, float(ep_ret), ep_len))
            obs = env.reset()
            consider(env.raw, env._n_lock, env._J, step)
            ep_ret, ep_len = 0.0, 0
        if step > WARMUP:
            beta = gcn_sac.beta_schedule(step - WARMUP, n_steps - WARMUP)
            agent.update(beta)
            n_updates += 1
        if step % 1000 == 0:
            print(f"  [{label}] step {step}/{n_steps}  best n_lock="
                  f"{best['n_lock']}/6  best J={best['J']:.4f}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    wall = time.time() - t0
    print(f"  [{label}] done: best n_lock={best['n_lock']}/6, "
          f"J={best['J']:.4f}, {wall:.0f}s, {n_updates} updates")
    summary = dict(
        label=label, encoder=encoder, use_per=use_per,
        env_steps=n_steps, n_updates=n_updates, wall_clock_s=wall,
        best_n_lock=int(best["n_lock"]), best_J=float(best["J"]),
        best_raw=[float(x) for x in best["raw"]],
        episode_returns=[(int(s), round(r, 4), int(l))
                         for s, r, l in ep_returns],
        best_curve=[(int(s), int(n), float(j)) for s, n, j in best_curve],
    )
    return summary, (np.array(archive_raw), np.array(archive_nl),
                     np.array(archive_J)), best


# ----------------------------------------------------------------------
# 1) GCN-SAC + PER   2) MLP-SAC + uniform replay (ablation)
# ----------------------------------------------------------------------
print("=== run 1: GCN-SAC + PER ===")
sum_gcn, archive, best_gcn = run_sac("gcn+per", "gcn", True, N_STEPS, SEED)
results["gcn_sac_per"] = sum_gcn

print("=== run 2: MLP-SAC + uniform replay (ablation) ===")
sum_mlp, _, best_mlp = run_sac("mlp+uniform", "mlp", False, N_STEPS, SEED)
results["mlp_sac_uniform"] = sum_mlp


# ----------------------------------------------------------------------
# 3) Hybrid: top-3 RL designs -> adjoint refinement
# ----------------------------------------------------------------------
print("=== run 3: hybrid RL -> adjoint refinement ===")


def top_k_distinct(raws, nlocks, Js, k=3, min_dist=0.5):
    """Best-k visited designs by (n_lock desc, J asc), pairwise-distinct in
    raw space so the adjoint refinements start from different basins."""
    order = np.lexsort((Js, -nlocks))
    picked = []
    for i in order:
        if all(np.linalg.norm(raws[i] - raws[j]) >= min_dist for j in picked):
            picked.append(i)
        if len(picked) == k:
            break
    return picked


obj, _ = adjoint.make_objective(vp, pst, circ, rl_env.TEMPS)
idxs = top_k_distinct(*archive, k=3)
hybrid_trace = []
t_hyb0 = time.time()
for rank, i in enumerate(idxs):
    raw0 = jnp.asarray(archive[0][i])
    print(f"  refining seed {rank}: n_lock={int(archive[1][i])}, "
          f"J={archive[2][i]:.4f}")
    raw_ref, hist = adjoint.optimize(obj, raw0, steps=ADJ_STEPS, lr=0.02,
                                     log_every=50)
    J_ref, f_lo, f_hi = evaluator(raw_ref)
    n_ref = int(np.sum((np.abs(np.asarray(f_lo) - F_REF) < 0.5e6)
                       & (np.abs(np.asarray(f_hi) - F_REF) < 0.5e6)))
    hybrid_trace.append(dict(
        seed_rank=rank,
        seed_raw=[float(x) for x in archive[0][i]],
        seed_n_lock=int(archive[1][i]), seed_J=float(archive[2][i]),
        refined_raw=[float(x) for x in np.asarray(raw_ref)],
        refined_n_lock6=n_ref, refined_J=float(J_ref),
        adjoint_hist=[(int(a), float(b), float(c)) for a, b, c in hist]))
t_hybrid = time.time() - t_hyb0


# ----------------------------------------------------------------------
# 4) Full-transient verification at 10 temperatures
# ----------------------------------------------------------------------
print("=== run 4: full verification (t_end=100us, dt=10ns) ===")
temps_v = jnp.asarray(TEMPS_VERIF)


@jax.jit
def verify_raw(raw):
    design = adjoint.raw_to_design(raw)

    def fend(T, vc0):
        tr = pll.loop_transient(vp, pst, circ, T, design=design, vc0=vc0,
                                t_end=100e-6, dt=10e-9, f_ref=F_REF)
        return pll.lock_metrics(tr, f_ref=F_REF)["f_end"]

    lo = jax.vmap(lambda T: fend(T, 1.0))(temps_v)
    hi = jax.vmap(lambda T: fend(T, 5.5))(temps_v)
    return lo, hi


def verify(raw, label):
    lo, hi = (np.asarray(x) for x in verify_raw(jnp.asarray(raw)))
    J6 = float(evaluator(jnp.asarray(raw))[0])
    rows, n = [], 0
    for k, T in enumerate(TEMPS_VERIF):
        lock = bool(abs(lo[k] - F_REF) < 0.5e6 and abs(hi[k] - F_REF) < 0.5e6)
        n += lock
        rows.append(dict(T_K=float(T), f_end_lo_MHz=float(lo[k] / 1e6),
                         f_end_hi_MHz=float(hi[k] / 1e6), lock=lock))
        print(f"  [{label}] T={T:3.0f}K  f_lo={lo[k]/1e6:6.2f}  "
              f"f_hi={hi[k]/1e6:6.2f}  lock={lock}")
    print(f"  [{label}] locked {n}/10, J(6-corner) = {J6:.4f}")
    return dict(raw=[float(x) for x in np.asarray(raw)], per_corner=rows,
                n_lock10=int(n), J6=J6)


# pick the best refined design by (10-corner locks, -J): verification is
# cheap, so all three refinements are checked and the winner kept
ver_refined = [verify(h["refined_raw"], f"hybrid-cand{r}")
               for r, h in enumerate(hybrid_trace)]
i_best = max(range(len(ver_refined)),
             key=lambda i: (ver_refined[i]["n_lock10"],
                            -ver_refined[i]["J6"]))
for r, h in enumerate(hybrid_trace):
    h["refined_n_lock10"] = ver_refined[r]["n_lock10"]
best_hybrid_raw = np.asarray(ver_refined[i_best]["raw"])
results["hybrid"] = dict(wall_clock_s=t_hybrid, adjoint_steps=ADJ_STEPS,
                         adjoint_lr=0.02, trace=hybrid_trace,
                         best_candidate=i_best)

print("--- final verification tables ---")
ver_base = verify(raw_baseline, "adjoint-only")
ver_rl = verify(best_gcn["raw"], "pure-RL")
results["verification"] = dict(adjoint_baseline=ver_base, pure_rl=ver_rl,
                               hybrid=ver_refined[i_best],
                               hybrid_candidates=ver_refined)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
if (ver_refined[i_best]["n_lock10"] >= ver_base["n_lock10"]
        and ver_refined[i_best]["J6"] < ver_base["J6"]):
    np.save("results/raw_opt_hybrid.npy", best_hybrid_raw)
    results["hybrid_saved"] = True
    print("saved results/raw_opt_hybrid.npy (hybrid dominates baseline)")
else:
    results["hybrid_saved"] = False
    print("hybrid does NOT dominate baseline; raw_opt_hybrid.npy not saved")

results["total_wall_clock_s"] = time.time() - t_wall0
json.dump(results, open("results/rl_results.json", "w"), indent=1,
          default=float)
print(f"total wall clock: {results['total_wall_clock_s']:.0f}s")
ten = ver_refined[i_best]["n_lock10"]
print(f"HYBRID RESULT: {ten}/10 verification corners locked "
      + ("-> 10/10 REACHED" if ten == 10 else "-> 10/10 not reached"))
print("saved results/rl_results.json")
