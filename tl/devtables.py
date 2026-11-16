"""Per-micron HEMT-varactor C(V,T) and gate leakage Ig(V,T) from the PDK MVSG model (ngspice)."""
import os, netlib, numpy as np, subprocess, tempfile, json
TS = [218, 233, 253, 273, 300, 323, 348, 373, 398, 423]
VS = np.round(np.arange(-1.0, 6.51, 0.25), 3)
W = 100e-6
def cv(tc):
    out = []
    for v in VS:
        net = netlib.Net('cv', tc)
        net.add('vg a 0 dc %g ac 1' % v)
        net.varactor('a', '0', W)
        d = tempfile.mkdtemp(dir=netlib.SCRATCH)
        open(d + '/c.sp', 'w').write(net.text() + '.ac lin 1 1meg 1meg\n.control\nrun\nlet c = -imag(i(vg))/(2*3.14159265358979*1e6)\nprint c\nquit\n.endc\n.end\n')
        r = subprocess.run([netlib.NGSPICE, '-b', d + '/c.sp'], capture_output=True, text=True)
        val = [l for l in r.stdout.splitlines() if l.strip().startswith('c =')]
        out.append(float(val[-1].split('=')[1]))
    return out
def ig(tc):
    net = netlib.Net('ig', tc)
    net.add('vg a 0 dc 0'); net.varactor('a', '0', W)
    d = tempfile.mkdtemp(dir=netlib.SCRATCH)
    open(d + '/c.sp', 'w').write(net.text() + '.dc vg -1 6.5 0.25\n.control\nrun\nwrdata %s/ig i(vg)\nquit\n.endc\n.end\n' % d)
    subprocess.run([netlib.NGSPICE, '-b', d + '/c.sp'], capture_output=True)
    a = np.loadtxt(d + '/ig'); return (-a[:, 1]).tolist()
res = dict(V=VS.tolist(), T=TS, W_um=100.0, C_fF_per_um={}, Ig_A_per_um={})
for T in TS:
    c = cv(T - 273.15)
    res['C_fF_per_um'][str(T)] = [x * 1e15 / 100.0 for x in c]
    res['Ig_A_per_um'][str(T)] = [x / 100.0 for x in ig(T - 273.15)]
    print(T, 'C/um at 0,3,6 V:', [round(res['C_fF_per_um'][str(T)][i], 3) for i in (4, 16, 28)],
          'Ig/um at 5V:', '%.2e' % res['Ig_A_per_um'][str(T)][24], flush=True)
json.dump(res, open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'devtables.json'), 'w'))   # PDK-derived: keep private
