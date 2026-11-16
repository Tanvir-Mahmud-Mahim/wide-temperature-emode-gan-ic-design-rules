"""Phase error of a closed-loop run after removing the modulator's known quantization phase.

For the m-th feedback edge the divider has counted sum_{j<m} N_j oscillator periods. In lock the
feedback edge is therefore displaced from the reference edge by the quantization phase
q_m = 360 deg * sum_{j<m} (N_j - Nbar) / Nbar (reference-period degrees). Subtracting q (with the
integer alignment of the sequence index that minimizes the spread) leaves the loop's own phase error.
In integer mode q = 0. Over the last 20 % of a run this gives:
  res_mean_deg : static phase error
  res_pp_deg   : peak-to-peak residual phase error (a cycle slip would add 360 deg)
  f_err_ppm    : average output-frequency error from the slope of the residual phase
usage: python3 tl/frac_residual.py results/tl/fin_233_frac_5.2.json [...]
"""
import sys, json
import numpy as np


def mash2(frac, n_int, length, bits=20):
    """MASH 1-1 division-ratio sequence (same as tl/pllnet.mash2)."""
    M = 1 << bits; k = int(round(frac * M)); a1 = a2 = 0; c2p = 0; out = []
    for _ in range(length):
        a1 += k; c1 = a1 >> bits; a1 &= M - 1
        a2 += a1; c2 = a2 >> bits; a2 &= M - 1
        out.append(n_int + c1 + c2 - c2p); c2p = c2
    return out, k / M


def phase_metrics(r, f_cmp=10e6, win=0.2):
    ps = r['phase_series']; t = np.array(ps['t']); ph = np.array(ps['deg'])
    t_end = r['tstop_us'] * 1e-6; m = t > (1 - win) * t_end; k = np.arange(len(t))[m]
    if r['mode'] == 'frac':
        nint = int(np.floor(r['n_div'] + 1e-6)); frac = r['n_div'] - nint
        seq, fr = mash2(frac, nint, len(t) + 500); nbar = nint + fr
        q = 360.0 * np.concatenate([[0.0], np.cumsum(np.array(seq, float) - nbar)]) / nbar
        best = None
        for s in range(0, 400):
            if (k + s).max() >= len(q):
                break
            res = (ph[m] - q[k + s] + 180.0) % 360.0 - 180.0
            if best is None or np.ptp(res) < best[0]:
                best = (np.ptp(res), s, res, np.ptp(q[k + s]))
        _, s, res, qpp = best
    else:
        res = (ph[m] + 180.0) % 360.0 - 180.0; s = 0; qpp = 0.0
    pf = np.polyfit(t[m], res, 1); slope = pf[0]          # deg/s
    detr = res - np.polyval(pf, t[m])                      # residual without slow settling drift
    return dict(res_mean_deg=float(np.mean(res)), res_pp_deg=float(np.ptp(res)), align=int(s),
                res_rms_detr_deg=float(np.std(detr)),
                q_pp_deg=float(qpp), f_err_ppm=float(slope / 360.0 / f_cmp * 1e6))


if __name__ == '__main__':
    for p in sys.argv[1:]:
        print(p, phase_metrics(json.load(open(p))))
