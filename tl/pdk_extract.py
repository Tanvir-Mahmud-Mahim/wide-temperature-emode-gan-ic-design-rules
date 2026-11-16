#!/usr/bin/env python3
"""Extract the PDK-derived constants used by the models into results/pdk_derived.json.
usage: GANPLL_PDK_DIR=/path/to/ganic200soi-pdk python3 tl/pdk_extract.py
The output must not be published (PDK non-disclosure agreement)."""
import os, re, json, sys
pdk = os.environ.get('GANPLL_PDK_DIR')
if not pdk:
    sys.exit('set GANPLL_PDK_DIR to the root of the foundry PDK')
res = open(os.path.join(pdk, 'models', 'spectre', 'ganic200soi_resistors.scs')).read()
hemt = open(os.path.join(pdk, 'models', 'spectre', 'ganic200soi_hemts.scs')).read()
def sub(txt, name):
    return re.search(r'subckt %s.*?ends' % name, txt, re.S).group(0)
def par(sec, key):
    return float(re.search(r'\+\s*%s\s*=\s*([-+0-9.eE]+)' % key, sec).group(1))
out = {}
for name, key in (('ganic200soi_res2deg', 'res2deg'), ('ganic200soi_resom', 'resom')):
    s = sub(res, name); out[key] = dict(ptc1=par(s, 'ptc1'), ptc2=par(s, 'ptc2'))
m = re.search(r'rth\s*=\s*([-+0-9.eE]+)\*pow\(w,\s*([-+0-9.eE]+)\)', sub(hemt, 'ganic200soi_ehemtlv'))
out['rth'] = dict(a=float(m.group(1)), b=float(m.group(2)))
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
json.dump(out, open(os.path.join(root, 'results', 'pdk_derived.json'), 'w'), indent=1)
print('wrote results/pdk_derived.json (keep private)')
