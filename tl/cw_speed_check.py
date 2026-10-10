"""Open-loop speed check of the final ring at 423 K (fastest band) with the estimated wire load
doubled, against V_c, for precharge rails of 7.0 and 7.5 V. usage: python3 cw_speed_check.py designs/final_f2.json results/cw_speed_check.json"""
import sys, os, json
import vcro, doe
D = json.load(open(sys.argv[1])); d = D['design']
vb = json.load(open(os.path.join(os.path.dirname(os.path.abspath(sys.argv[2])), 'f2_423_int.json')))['vb_tail']
out = []
for cw, vpre in ((1.0, 7.0), (2.0, 7.0), (2.0, 7.5)):
    for vc in (4.6, 5.0, 5.4):
        p = doe.design_params(d['drive'], d['s_var'], vb, 0.0); p['Cw'] = doe.CW * cw; p['vboot'] = vpre
        r = vcro.tb(p, 423 - 273.15, vc, tstop=2e-6)
        out.append(dict(T=423.0, cw_scale=cw, vpre=vpre, vc=vc, f=r['f'], vgs_max=r['vgs_max'])); print(out[-1], flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
