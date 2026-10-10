"""Transistor-level netlist helpers for the GaN PLL (ngspice-46 + OSDI MVSG).

Faithful to the PDK: the LV e-mode HEMT subcircuit ties body to source, and the 2DEG and
ohmic-metal resistors carry the PDK temperature coefficients.  The PDK itself (model files,
OSDI object) is not part of this repository; set GANPLL_PDK to the HEMT model file and
GANPLL_PDK_DIR to the PDK root (tl/pdk_extract.py derives the constants used here).
"""
import os, re, subprocess, tempfile
import numpy as np
_last_stdout = []

PDK = os.environ.get('GANPLL_PDK', '')
NGSPICE = os.environ.get('GANPLL_NGSPICE', 'ngspice')
SCRATCH = os.environ.get('GANPLL_SCRATCH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch'))
os.makedirs(SCRATCH, exist_ok=True)
L_G = 1.3e-6

def _card_params(sub='ganic200soi_ehemtlv'):
    txt = open(PDK).read()
    sec = re.search(r'subckt %s.*?ends' % sub, txt, re.S).group(0)
    p = {}
    for m in re.finditer(r'^\s*\+\s*(\w+)\s*=\s*([^/\s]+)', sec, re.M):
        k, v = m.group(1).lower(), m.group(2)
        if k in ('w', 'l', 'nf', 'lgd'):
            continue
        try:
            float(v)
        except ValueError:
            continue          # expressions (rth = f(w)) handled per instance
        p[k] = v
    return p

BASE = _card_params() if PDK else {}

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ganpll import pdk_local

def rth_of(w):
    """Width-dependent thermal resistance of the PDK HEMT subcircuit."""
    c = pdk_local.constants()['rth']
    return c['a'] * w ** c['b']

def card(name, w, **over):
    p = dict(BASE)
    p['rth'] = '%.6g' % rth_of(w)
    for k, v in over.items():
        p[k] = '%.6g' % v if isinstance(v, (int, float)) else v
    body = ' '.join('%s=%s' % (k, v) for k, v in p.items())
    return '.model %s mvsg(%s w=%.6g l=%.6g ngf=1)' % (name, body, w, L_G)

def r2deg(r25, tc):
    """PDK 2DEG resistor value at temperature tc (deg C); voltage coefficients not modelled."""
    return r25 * pdk_local.r2deg_factor(tc + 273.15)

def rom(r25, tc):
    """PDK ohmic-metal resistor value at temperature tc (deg C)."""
    return r25 * pdk_local.rom_factor(tc + 273.15)

class Net:
    def __init__(self, title, tc):
        self.lines = ['* ' + title, '.options temp=%g' % tc]
        self.tc = tc
        self.models = {}
        self.n = 0
    def model(self, w, **over):
        key = (round(w * 1e9), tuple(sorted(over.items())))
        if key not in self.models:
            name = 'm%d' % len(self.models)
            self.models[key] = name
            self.lines.insert(2, card(name, w, **over))
        return self.models[key]
    def hemt(self, d, g, s, w, **over):
        self.n += 1
        self.lines.append('n%d %s %s %s %s %s' % (self.n, d, g, s, s, self.model(w, **over)))
    def varactor(self, a, b, w, **over):
        """HEMT varactor: gate = A, drain+source shorted = B."""
        self.n += 1
        self.lines.append('n%d %s %s %s %s %s' % (self.n, b, a, b, b, self.model(w, **over)))
    def r(self, a, b, val):
        self.n += 1
        self.lines.append('r%d %s %s %.6g' % (self.n, a, b, val))
    def r2deg(self, a, b, r25):
        self.r(a, b, r2deg(r25, self.tc))
    def c(self, a, b, val):
        self.n += 1
        self.lines.append('c%d %s %s %.6g' % (self.n, a, b, val))
    def add(self, s):
        self.lines.append(s)
    def text(self):
        return '\n'.join(self.lines) + '\n'

def run(net_text, wav_vars, tstop, tstep, tmax=None, ic=None, timeout=20000, extra_ctl=(), keep=None, save_extra=()):
    """Run a transient; returns dict var->array plus 't'."""
    tmax = tmax or tstep
    d = tempfile.mkdtemp(prefix='ng_', dir=SCRATCH)
    wav = os.path.join(d, 'wav')
    lines = [net_text.rstrip('\n')]
    if ic:
        lines.append('.ic ' + ' '.join('v(%s)=%g' % (k, v) for k, v in ic.items()))
    lines.append('.save ' + ' '.join('v(%s)' % v for v in wav_vars) + (' ' + ' '.join(save_extra) if save_extra else ''))
    lines += ['.tran %g %g 0 %g' % (tstep, tstop, tmax), '.control', 'set filetype=binary',
              'run', *extra_ctl, 'wrdata %s %s' % (wav, ' '.join('v(%s)' % v for v in wav_vars)),
              'quit', '.endc', '.end']
    sp = os.path.join(d, 'c.sp')
    open(sp, 'w').write('\n'.join(lines) + '\n')
    r = subprocess.run([NGSPICE, '-b', sp], capture_output=True, text=True, timeout=timeout)
    global _last_stdout
    _last_stdout = [l for l in (r.stdout + r.stderr).splitlines() if 'Reference value' not in l]
    if not os.path.exists(wav):
        raise RuntimeError(r.stdout[-1500:] + r.stderr[-1500:])
    a = np.loadtxt(wav)
    out = {'t': a[:, 0], '_meas': {}}
    for line in r.stdout.splitlines():
        mm = re.match(r'^\s*(\w+)\s*=\s*([-+0-9.eE]+)', line)
        if mm and mm.group(1).lower() not in ('t', 'time'):
            try:
                out['_meas'][mm.group(1).lower()] = float(mm.group(2))
            except ValueError:
                pass
    for i, v in enumerate(wav_vars):
        out[v] = a[:, 2 * i + 1]
    if keep:
        os.makedirs(os.path.dirname(keep), exist_ok=True)
        open(keep, 'w').write(open(sp).read())
    return out

def freq(t, v, t0):
    m = t >= t0
    tt, vv = t[m], v[m]
    if len(vv) < 10:
        return 0.0, 0.0
    lo, hi = vv.min(), vv.max()
    mid = 0.5 * (lo + hi)
    s = vv > mid
    idx = np.where(s[1:] & ~s[:-1])[0]
    if len(idx) < 3 or hi - lo < 0.5:
        return 0.0, hi - lo
    tc = tt[idx] + (mid - vv[idx]) / (vv[idx + 1] - vv[idx]) * (tt[idx + 1] - tt[idx])
    return (len(tc) - 1) / (tc[-1] - tc[0]), hi - lo
