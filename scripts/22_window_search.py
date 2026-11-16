"""Universal-window search with a thermometer-coded band capacitor.
Band b (0..B) adds b*C_u (pF) per stage.  For each design, at every temperature of a
fine grid, the lockable band of capacitor setting b is [f(Vc_on + m_on), f(vhi)].
Score = min over T of the best (over b) log-margin of the target F0 inside the band.
Bias law: realizable divider vb = 6 (a2 g + ao h)/(a2 g + ao h + h).
usage: python3 window_search3.py tl_vcro.json m_on vhi F0_MHz B"""
import sys; import os; sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp, json
from ganpll import pinn, tl_anchor as ta
pst = pinn.load_state('results/pinn_state.json')
tp, ex = ta.load(sys.argv[1])
m_on, vhi, F_LO, F_HI, B = float(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4]) * 1e6, float(sys.argv[5]) * 1e6, int(sys.argv[6])
F0 = F_LO
TF = jnp.arange(218.0, 423.01, 5.0)

from ganpll import pdk_local
def g2deg(T):
    return pdk_local.r2deg_factor(T)
def hom(T):
    return pdk_local.rom_factor(T)
def vb_div(a2, ao, T):
    bot = a2 * g2deg(T) + ao * hom(T); return 6.0 * bot / (bot + hom(T))

def margins(d, s, cu, a2, ao):
    """(nT, B+1) log-margins of F0 inside each band."""
    T = TF[:, None]; b = jnp.arange(B + 1)[None, :]
    vb = vb_div(a2, ao, T); cb = b * cu
    T, vb, cb = jnp.broadcast_arrays(T, vb, cb)
    vo = ta.vc_on(tp, T, vb, d, s, cb)
    flo = ta.f_model(tp, pst, T, jnp.minimum(vo + m_on, vhi - 0.2), vb, d, s, cb)
    fhi = ta.f_model(tp, pst, T, jnp.full_like(T, vhi), vb, d, s, cb)
    return jnp.minimum(jnp.log(fhi / F_HI), jnp.log(F_LO / flo)), flo, fhi, vo

A2 = jnp.geomspace(0.02, 50.0, 25); AO = jnp.geomspace(1e-3, 20.0, 13)
@jax.jit
def score(d, s, cu):
    def one(a2, ao):
        m, _, _, _ = margins(d, s, cu, a2, ao)
        return jnp.min(jnp.max(m, axis=1))
    return jax.vmap(lambda a: jax.vmap(lambda b: one(a, b))(AO))(A2)

best = None
for d in np.linspace(1.0, 4.5, 8):
    for s in np.geomspace(1.0, 6.0, 6):
        for cu in np.geomspace(0.1, 3.0, 12):
            M = np.asarray(score(d, s, cu)); i, j = np.unravel_index(np.argmax(M), M.shape)
            if best is None or M[i, j] > best[0]:
                best = (float(M[i, j]), float(d), float(s), float(cu), float(A2[i]), float(AO[j]))
sc, d, s, cu, a2, ao = best
print('m_on=%.2f vhi=%.1f F=[%.1f,%.1f] MHz B=%d: min log-margin %.3f (x%.3f) drive=%.2f s_var=%.2f C_u=%.2f pF a2=%.3f ao=%.4f'
      % (m_on, vhi, F_LO / 1e6, F_HI / 1e6, B, sc, np.exp(sc), d, s, cu, a2, ao))
m, flo, fhi, vo = margins(d, s, cu, a2, ao)
m = np.asarray(m); vb = np.asarray(vb_div(a2, ao, TF))
for k, T in enumerate(np.asarray(TF)):
    ok = [b for b in range(B + 1) if m[k, b] > 0]
    print('T=%d vb=%.2f valid bands %s  ' % (T, vb[k], ok) + ' '.join('b%d:[%.1f,%.1f] on %.2f' % (b, flo[k, b] / 1e6, fhi[k, b] / 1e6, vo[k, b]) for b in range(B + 1)))
json.dump(dict(score=sc, drive=d, s_var=s, C_u=cu, a2=a2, ao=ao, m_on=m_on, vhi=vhi, F_LO=F_LO, F_HI=F_HI, B=B),
          open('ws3_best_%g_%g_%g_%d.json' % (m_on, F_LO / 1e6, F_HI / 1e6, B), 'w'))
