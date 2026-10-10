"""RL co-design environment around the differentiable GaN PLL model.

The environment exposes the 10-parameter bounded design space of
``ganpll.adjoint.DESIGN_SPACE`` as a continuous-control task whose reward
mixes the *non-smooth* global design goal (number of temperature corners with
dual-start frequency lock) with the smooth adjoint objective J as a dense
shaping term.  It is intended for a graph-neural soft actor-critic agent
(see ``ganpll.gcn_sac``): the observation is a small circuit graph
(Kipf & Welling, ICLR 2017 style message passing on a fixed adjacency)
plus a global feature vector of per-corner transient outcomes.

Observation
-----------
* 12 nodes: one per design parameter
  (CP, Rs1, Cs1, Rs23, C2var, C3var, drive, Cfix, VCROvar, dVb)
  plus 2 summary nodes (loop-filter node, VCRO node).
  Node features (14): one-hot component type (10) | current raw value |
  log physical design value | log lower bound | log upper bound
  (the additive dVb parameter uses its plain value/bounds since it may be
  negative; summary nodes carry zeros and are shaped purely by message
  passing over the circuit topology).
* Edges (fixed circuit connectivity): CP->LF; Rs1,Cs1,Rs23,C2,C3->LF;
  LF->VCRO; drive,Cfix,VCROvar,dVb->VCRO; VCRO->CP (feedback).  Stored as a
  row-normalized symmetric adjacency with self-loops (mean aggregation).
* Global features (25): per-corner tail frequency errors (f_end-f_ref)/f_ref
  for both acquisition starts vc0 in {1.0, 5.5} at the 6 optimization
  temperatures (12 numbers), the corresponding dual-per-start lock flags
  (12 numbers), and log1p of the current smooth objective J.

The observation handed to the agent is the flat concatenation
(12*14 + 25 = 193); ``ganpll.gcn_sac`` re-splits it into (nodes, globals).

Action / dynamics
-----------------
Action a in [-1,1]^10; raw <- clip(raw + 0.35*a, -4, 4).

Reward
------
r = 2.0*(n_lock' - n_lock) + clip(J - J', -10, 10) - 0.02
with n_lock the number of TEMPS corners where BOTH starts settle within
0.5 MHz of 50 MHz, and J the exact smooth objective of
``adjoint.make_objective`` (replicated here inside one jitted, temperature-
vmapped evaluator so a full env step costs a single fast forward pass,
~0.07 s on CPU).  Episodes run at most 30 steps; locking all 6 corners
terminates early with a +5 bonus.  The Delta-J term is clipped to +-10 as a
numerical guard: far-from-lock designs have J of order 10^2, and unclipped
differences destabilize the critics without changing the ordering of good
designs.

Resets alternate (50/50) between the adjoint-only optimum
``results/raw_opt_final.npy`` perturbed by N(0, 0.3) and ``adjoint.init_raw``
perturbed by N(0, 0.8), so the agent both refines the known basin and
explores globally.
"""
from __future__ import annotations

import numpy as np
import jax
import jax.numpy as jnp

from . import pll, adjoint

# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------
TEMPS = (218.0, 253.0, 300.0, 348.0, 398.0, 423.0)
F_REF = 50e6
LOCK_TOL_HZ = 0.5e6          # |f_end - 50 MHz| < 0.5 MHz (1 %) per start

N_PARAMS = len(adjoint.DESIGN_SPACE)   # 10
N_NODES = 12                            # 10 components + LF node + VCRO node
NODE_DIM = 14                           # onehot(10) + raw + logval + loglo + loghi
GLOBAL_DIM = 2 * len(TEMPS) * 2 + 1     # 12 ferr + 12 lock flags + log1p(J) = 25
OBS_DIM = N_NODES * NODE_DIM + GLOBAL_DIM
ACT_DIM = N_PARAMS

ACTION_SCALE = 0.35
RAW_CLIP = 4.0
MAX_STEPS = 30
STEP_PENALTY = 0.02
LOCK_REWARD = 2.0
DJ_CLIP = 10.0
ALL_LOCK_BONUS = 5.0

IDX_LF, IDX_VCRO = 10, 11
# component index -> (name in DESIGN_SPACE order)
#  0 CP(s_icp) 1 Rs1 2 Cs1 3 Rs23 4 C2var 5 C3var 6 drive 7 Cfix 8 VCROvar 9 dVb
EDGES = [
    (0, IDX_LF),                       # charge pump feeds the loop filter
    (1, IDX_LF), (2, IDX_LF), (3, IDX_LF), (4, IDX_LF), (5, IDX_LF),
    (IDX_LF, IDX_VCRO),                # control voltage drives the VCRO
    (6, IDX_VCRO), (7, IDX_VCRO), (8, IDX_VCRO), (9, IDX_VCRO),
    (IDX_VCRO, 0),                     # divider/PFD feedback closes the loop
]


def adjacency():
    """Row-normalized symmetric adjacency with self-loops (12x12).

    Mean aggregation over the fixed circuit graph: A_hat = D^-1 (A + A^T + I),
    the propagation operator used by the GCN encoder (Kipf & Welling 2017).
    """
    A = np.zeros((N_NODES, N_NODES))
    for i, j in EDGES:
        A[i, j] = 1.0
        A[j, i] = 1.0
    A += np.eye(N_NODES)
    A /= A.sum(axis=1, keepdims=True)
    return jnp.asarray(A)


# ----------------------------------------------------------------------
# Jitted design evaluator (transients + smooth objective in one pass)
# ----------------------------------------------------------------------

def make_evaluator(vp, pst, circ, t_end=60e-6, dt=20e-9):
    """Return a jitted ``eval_raw(raw) -> (J, f_end_lo(6,), f_end_hi(6,))``.

    Replicates ``adjoint.make_objective(vp, pst, circ, TEMPS)`` exactly
    (verified to 1e-13 relative) but vmaps the per-temperature transients so
    one call returns both the smooth objective and the settled frequencies
    needed for the discrete lock count.  Compiled once; subsequent calls run
    in ~0.07 s on CPU.
    """
    temps = jnp.asarray(TEMPS)
    # per-temperature weights of adjoint.make_objective: hottest corner 3x,
    # nominal 300 K corner 2x
    tw = np.ones(len(TEMPS))
    tw[np.argmax(np.asarray(TEMPS))] = 3.0
    tw[list(TEMPS).index(300.0)] = 2.0
    tw = jnp.asarray(tw)
    Vgrid = jnp.linspace(1.0, 5.5, 40)
    w_ferr, w_lockfrac, w_ripple, w_kvco, w_power, kmin = \
        300.0, 1.0, 10.0, 50.0, 0.02, 0.04

    def eval_raw(raw):
        design = adjoint.raw_to_design(raw)

        def one(T, vc0):
            tr = pll.loop_transient(vp, pst, circ, T, design=design,
                                    vc0=vc0, t_end=t_end, dt=dt, f_ref=F_REF)
            m = pll.lock_metrics(tr, f_ref=F_REF)
            return m["f_end"], m["ferr"], m["frac_locked"], m["ripple"]

        lo = jax.vmap(lambda T: one(T, 1.0))(temps)
        hi = jax.vmap(lambda T: one(T, 5.5))(temps)

        def cost_part(res):
            _, ferr, fl, rp = res
            return 0.5 * (w_ferr * ferr ** 2
                          + w_lockfrac * (1.0 - fl) ** 2 + w_ripple * rp)

        c = cost_part(lo) + cost_part(hi)

        def kpen(T):
            f_of_v = lambda v: pll.f_vcro(vp, pst, v, T, design)
            dfdv = jax.vmap(jax.grad(f_of_v))(Vgrid)
            return jnp.mean(jax.nn.relu(kmin - dfdv / F_REF) ** 2)

        c = c + w_kvco * jax.vmap(kpen)(temps)
        J = jnp.sum(tw * c)
        J = J + w_power * ((design["drive"] - 1.0) ** 2
                           + 0.2 * (design["s_icp"] - 1.0) ** 2)
        return J / len(TEMPS), lo[0], hi[0]

    return jax.jit(eval_raw)


# ----------------------------------------------------------------------
# Static node-feature scaffolding
# ----------------------------------------------------------------------

def _static_node_features():
    """One-hot component type + (log-)bounds; filled once at import."""
    feats = np.zeros((N_NODES, NODE_DIM))
    for i, (name, lo, hi) in enumerate(adjoint.DESIGN_SPACE):
        feats[i, i] = 1.0                          # one-hot type
        if name == "dvb":                          # additive parameter
            feats[i, 12] = lo
            feats[i, 13] = hi
        else:                                      # multiplicative -> log
            feats[i, 12] = np.log(lo)
            feats[i, 13] = np.log(hi)
    return feats


_STATIC_FEATS = _static_node_features()


def node_features(raw, design_vals):
    """(12,14) node feature matrix for the current design point.

    raw: np (10,) unconstrained vector; design_vals: np (10,) physical values
    in DESIGN_SPACE order (log-transformed here for multiplicative scales).
    """
    feats = _STATIC_FEATS.copy()
    for i, (name, _, _) in enumerate(adjoint.DESIGN_SPACE):
        feats[i, 10] = raw[i]
        feats[i, 11] = design_vals[i] if name == "dvb" else np.log(design_vals[i])
    return feats


# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

class PLLDesignEnv:
    """Episodic design-tuning environment (see module docstring).

    Parameters
    ----------
    vp, pst, circ : loaded model states (VCRO params, PINN state, circuit).
    raw_anchor    : np (10,) adjoint-only optimum used for warm-start resets.
    seed          : deterministic reset/perturbation stream.
    """

    def __init__(self, vp, pst, circ, raw_anchor, seed=0,
                 max_steps=MAX_STEPS, evaluator=None):
        self._eval = evaluator if evaluator is not None else \
            make_evaluator(vp, pst, circ)
        self.raw_anchor = np.asarray(raw_anchor, dtype=float)
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.n_evals = 0
        self.raw = None
        self._t = 0
        self._J = None
        self._n_lock = None

    # -- internals ------------------------------------------------------
    def _evaluate(self, raw):
        J, f_lo, f_hi = self._eval(jnp.asarray(raw))
        self.n_evals += 1
        J = float(np.nan_to_num(np.asarray(J), nan=1e3, posinf=1e3, neginf=1e3))
        f_lo = np.nan_to_num(np.asarray(f_lo), nan=0.0)
        f_hi = np.nan_to_num(np.asarray(f_hi), nan=0.0)
        lock_lo = np.abs(f_lo - F_REF) < LOCK_TOL_HZ
        lock_hi = np.abs(f_hi - F_REF) < LOCK_TOL_HZ
        n_lock = int(np.sum(lock_lo & lock_hi))
        return J, f_lo, f_hi, lock_lo, lock_hi, n_lock

    def _observe(self, raw, J, f_lo, f_hi, lock_lo, lock_hi):
        d = adjoint.raw_to_design(jnp.asarray(raw))
        dvals = np.array([float(d[name]) for name, _, _ in adjoint.DESIGN_SPACE])
        nodes = node_features(np.asarray(raw), dvals)
        g = np.concatenate([
            (f_lo - F_REF) / F_REF,
            (f_hi - F_REF) / F_REF,
            lock_lo.astype(float),
            lock_hi.astype(float),
            [np.log1p(max(J, 0.0))],
        ])
        return np.concatenate([nodes.ravel(), g]).astype(np.float64)

    # -- gym-like API -----------------------------------------------------
    def reset(self):
        """50 %: anchor + N(0,0.3); 50 %: init_raw + N(0,0.8)."""
        if self.rng.random() < 0.5:
            raw = self.raw_anchor + self.rng.normal(0.0, 0.3, N_PARAMS)
        else:
            raw = np.asarray(adjoint.init_raw()) + self.rng.normal(0.0, 0.8, N_PARAMS)
        self.raw = np.clip(raw, -RAW_CLIP, RAW_CLIP)
        self._t = 0
        J, f_lo, f_hi, llo, lhi, n_lock = self._evaluate(self.raw)
        self._J, self._n_lock = J, n_lock
        return self._observe(self.raw, J, f_lo, f_hi, llo, lhi)

    def step(self, action):
        """Apply delta, re-evaluate, shape reward.  Returns
        (obs, reward, done, info) with info carrying raw/J/n_lock."""
        a = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        self.raw = np.clip(self.raw + ACTION_SCALE * a, -RAW_CLIP, RAW_CLIP)
        self._t += 1
        J, f_lo, f_hi, llo, lhi, n_lock = self._evaluate(self.raw)
        r = (LOCK_REWARD * (n_lock - self._n_lock)
             + float(np.clip(self._J - J, -DJ_CLIP, DJ_CLIP))
             - STEP_PENALTY)
        terminal = n_lock == len(TEMPS)
        if terminal:
            r += ALL_LOCK_BONUS
        timeout = self._t >= self.max_steps
        done = terminal or timeout
        self._J, self._n_lock = J, n_lock
        obs = self._observe(self.raw, J, f_lo, f_hi, llo, lhi)
        info = dict(raw=self.raw.copy(), J=J, n_lock=n_lock,
                    terminal=terminal, timeout=timeout)
        return obs, r, done, info
