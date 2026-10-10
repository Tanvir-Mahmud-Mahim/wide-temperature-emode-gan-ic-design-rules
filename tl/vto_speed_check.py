"""Open-loop speed check of the final ring at 423 K with every HEMT threshold shifted by +0.2 V
(fastest band, V_b from the closed-loop run), for precharge rails of 7.0 and 7.5 V.
usage: python3 vto_speed_check.py designs/final_f2.json results/vto_speed_check.json"""
import sys, os, json
import netlib, vcro, doe
D = json.load(open(sys.argv[1])); d = D['design']
vb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), 'f2_423_int.json')))['vb_tail']
bh, bv = netlib.Net.hemt, netlib.Net.varactor
dv = 0.2
netlib.Net.hemt = lambda self, a, g, s, w, **o: bh(self, a, g, s, w, **dict({'vto': float(netlib.BASE['vto']) + dv}, **o))
netlib.Net.varactor = lambda self, a, b, w, **o: bv(self, a, b, w, **dict({'vto': float(netlib.BASE['vto']) + dv}, **o))
out = []
for vpre, vcs in ((7.0, (4.4, 4.8, 5.4)), (7.5, (4.4, 4.8, 5.4))):
    for vc in vcs:
        p = doe.design_params(d['drive'], d['s_var'], vb, 0.0); p['vboot'] = vpre
        r = vcro.tb(p, 423 - 273.15, vc, tstop=2e-6)
        out.append(dict(T=423.0, dvto=dv, vpre=vpre, vc=vc, vb=vb, f=r['f'])); print(out[-1], flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
