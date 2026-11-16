"""Translate the ngspice closed-loop netlist written by pllnet/netlib into a VACASK netlist,
run it with VACASK (an independent open-source simulator that loads the same MVSG model through
OSDI, compiled here by OpenVAF-Reloaded), and return the waveforms in the format of netlib.run.

Only the elements used in integer mode are translated: MVSG instances (n...), resistors,
capacitors and independent voltage sources (dc / pwl / pulse). Fractional mode uses an XSPICE
digital state machine, which VACASK does not have.
"""
import os, re, subprocess, tempfile
import numpy as np

import shutil
VACASK = os.environ.get('VACASK') or shutil.which('vacask') or 'vacask'   # VACASK 0.3.4 binary
VLIB = os.environ.get('VACASK_MOD', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(VACASK))), 'lib', 'vacask', 'mod'))
# the foundry MVSG Verilog-A model compiled with OpenVAF-Reloaded (openvaf-r); PDK file, not distributed
MVSG_OSDI_R = os.environ.get('MVSG_OSDI_R', '')
SCRATCH = os.environ.get('GANPLL_SCRATCH', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scratch'))
os.makedirs(SCRATCH, exist_ok=True)


def _nums(s):
    return [x for x in re.split(r'[\s,]+', s.strip()) if x]


def translate(ng_text, ic=None, tstop=30e-6, tmax=50e-12, save=(), rshunt=1e9, reltol=2e-3, abstol=1e-9, vntol=1e-5):
    out = ['PLL cross-check netlist (translated from ngspice)', 'ground 0',
           'load "%s"' % MVSG_OSDI_R, 'load "%s"' % os.path.join(VLIB, 'resistor.osdi'),
           'load "%s"' % os.path.join(VLIB, 'capacitor.osdi'),
           'model resistor resistor', 'model capacitor capacitor', 'model vsource vsource']
    nodes = set(); temp = 27.0
    for raw in ng_text.splitlines():
        line = raw.strip()
        if not line or line.startswith('*'):
            continue
        low = line.lower()
        if low.startswith('.options'):
            m = re.search(r'temp=([-+0-9.eE]+)', low)
            if m: temp = float(m.group(1))
            continue
        if low.startswith('.model'):
            m = re.match(r'\.model\s+(\S+)\s+(\w+)\s*\((.*)\)\s*$', line, re.I)
            name, mod, par = m.group(1), m.group(2), m.group(3)
            out.append('model %s %s %s' % (name, mod, ' '.join(_nums(par))))
            continue
        if low.startswith('.'):
            raise ValueError('unsupported control line: ' + line[:60])
        tok = line.split()
        k = tok[0][0].lower()
        if k == 'n':                                   # OSDI instance: n<k> d g s b model
            out.append('%s (%s) %s' % (tok[0], ' '.join(tok[1:5]), tok[5])); nodes.update(tok[1:5])
        elif k == 'r':
            out.append('%s (%s %s) resistor r=%s' % (tok[0], tok[1], tok[2], tok[3])); nodes.update(tok[1:3])
        elif k == 'c':
            out.append('%s (%s %s) capacitor c=%s' % (tok[0], tok[1], tok[2], tok[3])); nodes.update(tok[1:3])
        elif k == 'v':
            rest = ' '.join(tok[3:]); rl = rest.lower(); nodes.update(tok[1:3])
            if rl.startswith('dc'):
                out.append('%s (%s %s) vsource type="dc" dc=%s' % (tok[0], tok[1], tok[2], tok[4]))
            elif rl.startswith('pwl'):
                v = _nums(rest[rest.index('(') + 1:rest.rindex(')')])
                out.append('%s (%s %s) vsource type="pwl" wave=[%s]' % (tok[0], tok[1], tok[2], ','.join(v)))
            elif rl.startswith('pulse'):
                v0, v1, td, tr, tf, pw, per = _nums(rest[rest.index('(') + 1:rest.rindex(')')])
                out.append('%s (%s %s) vsource type="pulse" val0=%s val1=%s delay=%s rise=%s fall=%s width=%s period=%s'
                           % (tok[0], tok[1], tok[2], v0, v1, td, tr, tf, pw, per))
            else:
                raise ValueError('unsupported source: ' + line[:60])
        else:
            raise ValueError('unsupported element: ' + line[:60])
    nodes.discard('0')
    for i, nd in enumerate(sorted(nodes)):          # ngspice option rshunt: resistor from every node to ground
        out.append('rsh%d (%s 0) resistor r=%g' % (i, nd, rshunt))
    icl = ', '.join('"%s", %g' % (n_, v_) for n_, v_ in (ic or {}).items())
    out += ['control',
            '  options temp=%g tnom=27' % temp,
            '  options reltol=%g abstol=%g vntol=%g' % (reltol, abstol, vntol),
            '  options tran_method="gear2" rawfile="binary"',
            '  save ' + ' '.join('v(%s)' % s for s in save if not s.startswith('i(')) + ' ' + ' '.join(s for s in save if s.startswith('i(')),
            '  analysis tran1 tran stop=%g step=%g maxstep=%g icmode="op"%s' % (tstop, tmax, tmax, (' ic={%s}' % icl) if icl else ''),
            'endc']
    return '\n'.join(out) + '\n'


def run(ng_text, wav_vars, tstop, tmax=50e-12, ic=None, timeout=40000, save_extra=()):
    d = tempfile.mkdtemp(prefix='vc_', dir=SCRATCH)
    sim = os.path.join(d, 'pll.sim')
    open(sim, 'w').write(translate(ng_text, ic=ic, tstop=tstop, tmax=tmax, save=list(wav_vars) + list(save_extra)))
    r = subprocess.run([VACASK, 'pll.sim'], cwd=d, capture_output=True, text=True, timeout=timeout)
    raw = os.path.join(d, 'tran1.raw')
    if not os.path.exists(raw):
        raise RuntimeError((r.stdout + r.stderr)[-3000:])
    import sys
    sys.path.insert(0, os.environ.get('VACASK_PY', os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(VACASK))), 'lib', 'vacask', 'python')))
    from rawfile import rawread
    p = rawread(raw).get()
    w = {'t': np.asarray(p['time']), '_meas': {}, '_dir': d}
    for v in wav_vars:
        w[v] = np.asarray(p[v])
    for s in save_extra:
        key = s[2:-1] if s.startswith('i(') else s
        for cand in (s, key + ':flow(br)', key):
            if cand in p.names if hasattr(p, 'names') else False:
                w[s] = np.asarray(p[cand]); break
    return w
