#!/usr/bin/env python3
"""24_cv_map.py -- map the PINN capacitance shape onto the open-flow foundry-model capacitance
per micron of gate width: C = C_A + C_B * cap_unit(V, T), least squares over -1..6.5 V and the
ten temperatures of results/devtables.json (generated locally from the PDK by tl/devtables.py).
Writes results/cv_map.json (fit constants are used by ganpll/pll2.py)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax.numpy as jnp
from ganpll import pinn
pst = pinn.load_state('results/pinn_state.json')
d = json.load(open('results/devtables.json'))
V = np.asarray(d['V'], float); X, Y = [], []
for T in d['T']:
    c = np.asarray(d['C_fF_per_um'][str(T)], float)
    u = np.asarray(pinn.cap_unit(pst, jnp.asarray(V), jnp.full_like(jnp.asarray(V), float(T)), 0))
    X.append(u); Y.append(c)
X = np.concatenate(X); Y = np.concatenate(Y)
A = np.stack([np.ones_like(X), X], 1); coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
res = A @ coef - Y
rel = np.sqrt(np.mean((res / Y) ** 2)) * 100
print('C_A = %.4f fF/um, C_B = %.4f fF/um, relative RMS error %.2f %%' % (coef[0], coef[1], rel))
json.dump(dict(C_A=float(coef[0]), C_B=float(coef[1]), rel_rmse_pct=float(rel), n=int(len(Y))), open('results/cv_map.json', 'w'), indent=1)
