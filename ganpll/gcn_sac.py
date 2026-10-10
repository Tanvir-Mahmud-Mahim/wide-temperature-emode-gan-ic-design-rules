"""Graph-neural soft actor-critic with prioritized experience replay (JAX).

Agent for the PLL co-design environment ``ganpll.rl_env``:

* Encoder: 2-layer graph convolutional network (Kipf & Welling, "Semi-
  Supervised Classification with Graph Convolutional Networks", ICLR 2017)
  with mean aggregation over the fixed 12-node circuit adjacency, hidden 64,
  tanh; node embeddings are mean-pooled, concatenated with the 25 global
  transient features, and passed through a 2-layer MLP trunk.  An ``'mlp'``
  encoder variant (plain MLP on the flat observation) provides the
  graph-ablation baseline.
* Actor-critic: soft actor-critic (Haarnoja et al., "Soft Actor-Critic:
  Off-Policy Maximum Entropy Deep RL with a Stochastic Actor", ICML 2018)
  with a tanh-Gaussian 10-dim policy head, twin Q critics (clipped double-Q),
  Polyak-averaged target critics (tau = 0.005), and automatic temperature
  tuning against a target entropy of -10 (= -|A|).
* Replay: proportional prioritized experience replay (Schaul et al.,
  "Prioritized Experience Replay", ICLR 2016), alpha = 0.6, beta annealed
  0.4 -> 1.0, importance-sampling weights applied to the critic loss,
  priorities |TD error| + 1e-3, capacity 50 000.  ``use_per=False`` degrades
  to uniform sampling with unit weights (ablation).

Networks and updates are pure JAX + optax (jitted); the replay buffer is
numpy.  All randomness is seeded; the tanh-Gaussian log-density is guarded
against saturation (log-std clipped to [-5, 2], Jacobian term
log(1 - tanh(u)^2 + 1e-6)) so no NaNs can be produced.
"""
from __future__ import annotations

from functools import partial

import numpy as np
import jax
import jax.numpy as jnp
import optax

from . import rl_env

HIDDEN = 64
LR = 3e-4
GAMMA = 0.97
TAU = 0.005
TARGET_ENTROPY = -10.0
LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
PER_ALPHA = 0.6
PER_BETA0, PER_BETA1 = 0.4, 1.0
PER_EPS = 1e-3
CAPACITY = 50_000


# ----------------------------------------------------------------------
# Parameter initialization / network application
# ----------------------------------------------------------------------

def _linear(key, nin, nout, scale=1.0):
    """LeCun-normal linear layer parameters."""
    w = jax.random.normal(key, (nin, nout)) * scale * jnp.sqrt(1.0 / nin)
    return dict(w=w, b=jnp.zeros(nout))


def _apply(p, x):
    return x @ p["w"] + p["b"]


def init_encoder(key, encoder="gcn"):
    """GCN (2 message-passing layers + trunk) or flat-MLP encoder params."""
    ks = jax.random.split(key, 4)
    if encoder == "gcn":
        return dict(
            g1=_linear(ks[0], rl_env.NODE_DIM, HIDDEN),
            g2=_linear(ks[1], HIDDEN, HIDDEN),
            t1=_linear(ks[2], HIDDEN + rl_env.GLOBAL_DIM, HIDDEN),
            t2=_linear(ks[3], HIDDEN, HIDDEN),
        )
    # ablation: parameter-comparable plain MLP on the flat observation
    return dict(
        t0=_linear(ks[0], rl_env.OBS_DIM, 128),
        t1=_linear(ks[1], 128, HIDDEN),
        t2=_linear(ks[2], HIDDEN, HIDDEN),
    )


def apply_encoder(p, obs, A, encoder="gcn"):
    """obs: (..., OBS_DIM) flat observation -> (..., HIDDEN) embedding."""
    if encoder == "gcn":
        nd = rl_env.N_NODES * rl_env.NODE_DIM
        nodes = obs[..., :nd].reshape(*obs.shape[:-1],
                                      rl_env.N_NODES, rl_env.NODE_DIM)
        glob = obs[..., nd:]
        h = jnp.tanh(A @ _apply(p["g1"], nodes))     # message passing 1
        h = jnp.tanh(A @ _apply(p["g2"], h))         # message passing 2
        pooled = h.mean(axis=-2)                     # mean-pool nodes
        z = jnp.concatenate([pooled, glob], axis=-1)
        z = jnp.tanh(_apply(p["t1"], z))
        return jnp.tanh(_apply(p["t2"], z))
    z = jnp.tanh(_apply(p["t0"], obs))
    z = jnp.tanh(_apply(p["t1"], z))
    return jnp.tanh(_apply(p["t2"], z))


def init_actor(key, encoder="gcn"):
    k1, k2 = jax.random.split(key)
    return dict(enc=init_encoder(k1, encoder),
                head=_linear(k2, HIDDEN, 2 * rl_env.ACT_DIM, scale=0.1))


def init_critic(key, encoder="gcn"):
    k1, k2, k3 = jax.random.split(key, 3)
    return dict(enc=init_encoder(k1, encoder),
                h1=_linear(k2, HIDDEN + rl_env.ACT_DIM, HIDDEN),
                out=_linear(k3, HIDDEN, 1, scale=0.1))


def actor_dist(p, obs, A, encoder="gcn"):
    z = apply_encoder(p["enc"], obs, A, encoder)
    out = _apply(p["head"], z)
    mu, log_std = jnp.split(out, 2, axis=-1)
    log_std = jnp.clip(log_std, LOG_STD_MIN, LOG_STD_MAX)
    return mu, log_std


def sample_action(p, obs, A, key, encoder="gcn"):
    """Reparameterized tanh-Gaussian sample and its guarded log-density.

    log pi(a|s) = log N(u; mu, sigma) - sum_i log(1 - tanh(u_i)^2 + 1e-6)
    (Haarnoja 2018, App. C); the 1e-6 floor and log-std clipping prevent NaNs
    at tanh saturation.
    """
    mu, log_std = actor_dist(p, obs, A, encoder)
    std = jnp.exp(log_std)
    u = mu + std * jax.random.normal(key, mu.shape)
    a = jnp.tanh(u)
    logp = (-0.5 * ((u - mu) / std) ** 2 - log_std
            - 0.5 * jnp.log(2.0 * jnp.pi)).sum(-1)
    logp = logp - jnp.log(1.0 - a ** 2 + 1e-6).sum(-1)
    return a, logp


def critic_value(p, obs, act, A, encoder="gcn"):
    z = apply_encoder(p["enc"], obs, A, encoder)
    h = jnp.tanh(_apply(p["h1"], jnp.concatenate([z, act], axis=-1)))
    return _apply(p["out"], h).squeeze(-1)


# ----------------------------------------------------------------------
# Prioritized experience replay (Schaul et al. 2016, proportional variant)
# ----------------------------------------------------------------------

class PERBuffer:
    """Numpy replay buffer with proportional prioritization.

    P(i) = p_i^alpha / sum_k p_k^alpha, importance weights
    w_i = (N * P(i))^-beta normalized by max_i w_i.  New transitions enter
    with the current maximum priority so every experience is replayed at
    least once with high probability.  ``use_per=False`` gives uniform
    sampling and unit weights.
    """

    def __init__(self, obs_dim, act_dim, capacity=CAPACITY,
                 alpha=PER_ALPHA, use_per=True, seed=0):
        self.capacity = capacity
        self.alpha = alpha
        self.use_per = use_per
        self.rng = np.random.default_rng(seed)
        self.obs = np.zeros((capacity, obs_dim))
        self.act = np.zeros((capacity, act_dim))
        self.rew = np.zeros(capacity)
        self.nobs = np.zeros((capacity, obs_dim))
        self.done = np.zeros(capacity)
        self.prio = np.zeros(capacity)
        self.ptr = 0
        self.size = 0
        self.max_prio = 1.0

    def add(self, obs, act, rew, nobs, done):
        i = self.ptr
        self.obs[i], self.act[i], self.rew[i] = obs, act, rew
        self.nobs[i], self.done[i] = nobs, float(done)
        self.prio[i] = self.max_prio
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch, beta):
        n = self.size
        if self.use_per:
            p = self.prio[:n] ** self.alpha
            p /= p.sum()
            idx = self.rng.choice(n, size=batch, p=p)
            w = (n * p[idx]) ** (-beta)
            w /= w.max()
        else:
            idx = self.rng.integers(0, n, size=batch)
            w = np.ones(batch)
        return idx, dict(obs=self.obs[idx], act=self.act[idx],
                         rew=self.rew[idx], nobs=self.nobs[idx],
                         done=self.done[idx], w=w)

    def update_priorities(self, idx, td_abs):
        p = np.asarray(td_abs) + PER_EPS
        self.prio[idx] = p
        self.max_prio = max(self.max_prio, float(p.max()))


# ----------------------------------------------------------------------
# SAC agent
# ----------------------------------------------------------------------

class SACAgent:
    """Soft actor-critic (Haarnoja 2018) with twin critics, target networks,
    and automatic entropy temperature; encoder is a GCN or flat MLP."""

    def __init__(self, seed=0, encoder="gcn", use_per=True,
                 batch=128, lr=LR, gamma=GAMMA, tau=TAU):
        self.encoder = encoder
        self.batch = batch
        key = jax.random.PRNGKey(seed)
        ka, k1, k2, self._key = jax.random.split(key, 4)
        self.A = rl_env.adjacency()
        self.actor = init_actor(ka, encoder)
        self.critics = dict(q1=init_critic(k1, encoder),
                            q2=init_critic(k2, encoder))
        self.targets = jax.tree_util.tree_map(lambda x: x, self.critics)
        self.log_alpha = jnp.asarray(0.0)

        self.opt_a = optax.adam(lr)
        self.opt_c = optax.adam(lr)
        self.opt_alpha = optax.adam(lr)
        self.st_a = self.opt_a.init(self.actor)
        self.st_c = self.opt_c.init(self.critics)
        self.st_alpha = self.opt_alpha.init(self.log_alpha)

        self.buffer = PERBuffer(rl_env.OBS_DIM, rl_env.ACT_DIM,
                                use_per=use_per, seed=seed + 1)
        self._update = self._make_update(gamma, tau)
        self._act = jax.jit(partial(sample_action, A=self.A,
                                    encoder=self.encoder))
        self._act_mean = jax.jit(
            lambda p, o: jnp.tanh(actor_dist(p, o, self.A, self.encoder)[0]))

    # -- acting -----------------------------------------------------------
    def act(self, obs, deterministic=False):
        obs = jnp.asarray(obs)
        if deterministic:
            return np.asarray(self._act_mean(self.actor, obs))
        self._key, k = jax.random.split(self._key)
        a, _ = self._act(self.actor, obs, key=k)
        return np.asarray(a)

    # -- jitted SAC update --------------------------------------------------
    def _make_update(self, gamma, tau):
        A, enc = self.A, self.encoder

        def critic_loss(critics, actor, targets, log_alpha, b, key):
            na, nlogp = sample_action(actor, b["nobs"], A, key, enc)
            tq1 = critic_value(targets["q1"], b["nobs"], na, A, enc)
            tq2 = critic_value(targets["q2"], b["nobs"], na, A, enc)
            tq = jnp.minimum(tq1, tq2) - jnp.exp(log_alpha) * nlogp
            y = b["rew"] + gamma * (1.0 - b["done"]) * tq
            y = jax.lax.stop_gradient(y)
            q1 = critic_value(critics["q1"], b["obs"], b["act"], A, enc)
            q2 = critic_value(critics["q2"], b["obs"], b["act"], A, enc)
            td1, td2 = q1 - y, q2 - y
            loss = 0.5 * jnp.mean(b["w"] * (td1 ** 2 + td2 ** 2))
            td_abs = 0.5 * (jnp.abs(td1) + jnp.abs(td2))
            return loss, td_abs

        def actor_loss(actor, critics, log_alpha, b, key):
            a, logp = sample_action(actor, b["obs"], A, key, enc)
            q1 = critic_value(critics["q1"], b["obs"], a, A, enc)
            q2 = critic_value(critics["q2"], b["obs"], a, A, enc)
            loss = jnp.mean(jnp.exp(log_alpha) * logp - jnp.minimum(q1, q2))
            return loss, logp

        def alpha_loss(log_alpha, logp):
            return -jnp.mean(log_alpha
                             * jax.lax.stop_gradient(logp + TARGET_ENTROPY))

        @jax.jit
        def update(actor, critics, targets, log_alpha,
                   st_a, st_c, st_alpha, b, key):
            kc, ka = jax.random.split(key)
            (closs, td_abs), gc = jax.value_and_grad(critic_loss, has_aux=True)(
                critics, actor, targets, log_alpha, b, kc)
            up, st_c = self.opt_c.update(gc, st_c)
            critics = optax.apply_updates(critics, up)

            (aloss, logp), ga = jax.value_and_grad(actor_loss, has_aux=True)(
                actor, critics, log_alpha, b, ka)
            up, st_a = self.opt_a.update(ga, st_a)
            actor = optax.apply_updates(actor, up)

            gal = jax.grad(alpha_loss)(log_alpha, logp)
            up, st_alpha = self.opt_alpha.update(gal, st_alpha)
            log_alpha = optax.apply_updates(log_alpha, up)

            targets = jax.tree_util.tree_map(
                lambda t, c: (1.0 - tau) * t + tau * c, targets, critics)
            metrics = dict(critic_loss=closs, actor_loss=aloss,
                           alpha=jnp.exp(log_alpha),
                           entropy=-jnp.mean(logp))
            return actor, critics, targets, log_alpha, st_a, st_c, st_alpha, \
                td_abs, metrics

        return update

    def update(self, beta):
        """One SAC gradient step on a (prioritized) minibatch; returns
        the update metrics dict.  Priorities are refreshed with |TD|+eps."""
        idx, b = self.buffer.sample(self.batch, beta)
        b = {k: jnp.asarray(v) for k, v in b.items()}
        self._key, k = jax.random.split(self._key)
        (self.actor, self.critics, self.targets, self.log_alpha,
         self.st_a, self.st_c, self.st_alpha, td_abs, metrics) = \
            self._update(self.actor, self.critics, self.targets,
                         self.log_alpha, self.st_a, self.st_c,
                         self.st_alpha, b, k)
        self.buffer.update_priorities(idx, np.asarray(td_abs))
        return {k: float(v) for k, v in metrics.items()}


def beta_schedule(step, total):
    """Linear PER beta annealing 0.4 -> 1.0 over the training budget."""
    f = min(max(step / max(total, 1), 0.0), 1.0)
    return PER_BETA0 + (PER_BETA1 - PER_BETA0) * f
