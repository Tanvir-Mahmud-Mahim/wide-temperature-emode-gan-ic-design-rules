"""Open-source EDA cross-verification of the varactor surrogate.

Flow: the imec ganic200soi PDK ships the MIT Virtual-Source GaN HV (MVSG)
Verilog-A model.  We compile it with OpenVAF(-reloaded) to an OSDI shared
object and simulate the e-mode LV HEMT varactor (gate vs. shorted
source/drain/body) in ngspice across 218-423 K.  The small-signal gate
capacitance C(V) = Im{I_g}/(2*pi*f) is extracted with 1 MHz AC analyses at
swept bias, and compared with the PINN surrogate trained on Cadence Spectre
characterization.  This validates that the surrogate physics is reproducible
end-to-end in a fully open simulation stack (OpenVAF + ngspice).

Everything runs from the PDK's own model card (ganic200soi_ehemtlv section),
parsed automatically from the Spectre .scs file.
"""
import os
import sys, os, re, json, subprocess
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

SCS = os.environ.get('GANPLL_PDK', '')   # foundry HEMT model file (not distributed)
EDA = 'eda'
OSDI = os.path.abspath(os.path.join(EDA, 'mvsg.osdi'))
TEMPS_K = [218.0, 273.0, 300.0, 398.0, 423.0]
VBIAS = np.round(np.arange(0.0, 6.01, 0.25), 2)
FREQ = 1e6
# instance geometry: one finger of the large varactor test structure
W_INST, L_INST, NGF = 300e-6, 1.3e-6, 1.0
NF_TOTAL = 350  # total fingers of the large test structure

# ------------------------------------------------------------------
# 1. parse the ehemtlv model-parameter block from the PDK .scs
# ------------------------------------------------------------------
txt = open(SCS).read()
sec = re.search(r'subckt ganic200soi_ehemtlv.*?ends', txt, re.S).group(0)
params = {}
for m in re.finditer(r'^\s*\+\s*(\w+)\s*=\s*([-\d.eE+]+)', sec, re.M):
    name, val = m.group(1).lower(), m.group(2)
    params[name] = val  # later duplicates (e.g. lgd model param) overwrite
# instance-level parameters are passed on the instance line, not the card
for k in ('w', 'l', 'nf'):
    params.pop(k, None)
print(f'parsed {len(params)} MVSG model parameters from PDK card '
      f"(vto={params.get('vto')}, cg={params.get('cg')}, vtzeta={params.get('vtzeta')})")

card = ('.model mvsglv mvsg(' + ' '.join(f'{k}={v}' for k, v in params.items())
        + f' w={W_INST} l={L_INST} ngf={NGF})')

# ------------------------------------------------------------------
# 2. per-temperature ngspice runs (OSDI-loaded MVSG)
# ------------------------------------------------------------------
results = {}
for T in TEMPS_K:
    tc = T - 273.15
    # OSDI model is pre-loaded via ~/.spiceinit (pre_osdi runs before parse)
    lines = [
        f'* MVSG GaN HEMT varactor C-V testbench, T={T}K',
        f'.options temp={tc:.2f}',
        card,
        '* varactor: gate = node a ; source/drain/body shorted = node b (gnd)',
        'nvar 0 a 0 0 mvsglv',
        'va a 0 dc 0 ac 1',
        '.control',
    ]
    for v in VBIAS:
        lines += [f'alter va dc={v}',
                  f'ac lin 1 {FREQ:.0f} {FREQ:.0f}',
                  'let cval = imag(i(va))/(2*pi*{:.0f})'.format(FREQ),
                  'print cval >> cv_out.txt']
    lines += ['quit', '.endc', '.end']
    nl = os.path.join(EDA, f'varactor_{int(T)}K.sp')
    open(nl, 'w').write('\n'.join(lines))
    out = os.path.join(EDA, 'cv_out.txt')
    if os.path.exists(out):
        os.remove(out)
    r = subprocess.run(['/usr/local/bin/ngspice', '-b', os.path.basename(nl)], cwd=EDA,
                       capture_output=True, text=True, timeout=600)
    caps = []
    if os.path.exists(out):
        for line in open(out):
            m = re.search(r'cval\s*=\s*\(?([-\d.eE+]+)', line)
            if m:
                caps.append(float(m.group(1)))
    if len(caps) != len(VBIAS):
        # fallback: parse stdout
        caps = [float(x) for x in re.findall(r'cval\s*=\s*\(?([-\d.eE+]+)', r.stdout)]
    if len(caps) != len(VBIAS):
        print(f'T={T}: FAILED ({len(caps)} points). stderr tail:', r.stderr[-300:])
        continue
    # gate current sign: C = -Im(I(va))/(2 pi f) with source convention
    c = -np.array(caps)
    results[T] = c
    print(f'T={T:.0f}K: C range [{c.min()*1e15:.2f}, {c.max()*1e15:.2f}] fF/finger')

# ------------------------------------------------------------------
# 3. compare with the PINN surrogate (normalized shape + Vth trend)
# ------------------------------------------------------------------
import jax.numpy as jnp
from ganpll import pinn
pst = pinn.load_state('results/pinn_state.json')

def norm(c):
    return (c - c.min()) / (c.max() - c.min())

summary = dict(freq_Hz=FREQ, vbias=list(map(float, VBIAS)), temps_K=TEMPS_K,
               w=W_INST, l=L_INST, nf_total=NF_TOTAL, curves={}, metrics={})
rmse_all, vt_ngspice, vt_pinn = [], {}, {}
for T, c in results.items():
    cp = np.asarray(pinn.cap_fF(pst, jnp.asarray(VBIAS), jnp.full(len(VBIAS), T),
                                jnp.ones(len(VBIAS))))  # large device, fF
    nrm_ng, nrm_pi = norm(c), norm(cp)
    rmse = float(np.sqrt(np.mean((nrm_ng - nrm_pi) ** 2)))
    rmse_all.append(rmse)
    # transition voltage: bias at 50% capacitance swing
    vt_ng = float(np.interp(0.5, nrm_ng, VBIAS))
    vt_pi = float(np.interp(0.5, nrm_pi, VBIAS))
    vt_ngspice[T], vt_pinn[T] = vt_ng, vt_pi
    summary['curves'][str(int(T))] = dict(c_ngspice_fF=(c * 1e15).tolist(),
                                          c_pinn_fF=cp.tolist())
    summary['metrics'][str(int(T))] = dict(shape_rmse=rmse, vt_ngspice=vt_ng, vt_pinn=vt_pi,
                                           cratio_ngspice=float(c.max()/c.min()))
    print(f'T={T:.0f}K: normalized-shape RMSE {rmse*100:.2f}%, '
          f'V_t50: ngspice {vt_ng:.2f} V vs PINN {vt_pi:.2f} V')

if len(results) >= 2:
    Ts = sorted(results.keys())
    slope_ng = np.polyfit(Ts, [vt_ngspice[t] for t in Ts], 1)[0] * 1e3
    slope_pi = np.polyfit(Ts, [vt_pinn[t] for t in Ts], 1)[0] * 1e3
    summary['dvt_dT_mVK'] = dict(ngspice=float(slope_ng), pinn=float(slope_pi),
                                 pdk_card_vtzeta=float(params.get('vtzeta', 0)) * 1e3)
    summary['mean_shape_rmse_pct'] = float(np.mean(rmse_all) * 100)
    print(f'transition-voltage slope: ngspice/MVSG {slope_ng:+.2f} mV/K, '
          f'PINN {slope_pi:+.2f} mV/K, PDK card vtzeta {float(params["vtzeta"])*1e3:+.1f} mV/K')
    print(f'mean normalized-shape RMSE: {np.mean(rmse_all)*100:.2f}%')

json.dump(summary, open('results/eda_verification.json', 'w'), indent=1)
print('saved results/eda_verification.json')
