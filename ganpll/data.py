"""Loaders for (i) digitized Cadence Spectre figure data and (ii) compact
summaries of the full transient corner simulations (218-523 K).

Digitized CSVs come from the matplotlib SVG figures of the manuscript
(F2: varactor C-V-T, F11: VCRO tuning curves, F12: PSD / phase noise,
F14: block power).  Corner summaries are .npz files extracted from the raw
Cadence Spectre PSF->pickle waveform databases (~23 GB), holding decimated
key waveforms (Vc, CLK_out, Up, Down, Vosc ...) and all CLK_out edge
timestamps, from which frequency-versus-time and jitter are recomputed here.
"""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIG = os.path.join(ROOT, "data", "digitized")
SUM = os.path.join(ROOT, "data", "cadence_summaries")

PAPER_TEMPS_K = [218.0, 273.0, 300.0, 398.0, 423.0]


def _temp_from_label(fname):
    m = re.search(r"_c\d+_(\d+)_K", fname)
    return float(m.group(1)) if m else None


def load_varactor_cv(device="large"):
    """Digitized C-V-T curves from manuscript Fig. F2.

    device='small': axes_1, capacitance in fF (Wf=10um x 50 fingers)
    device='large': axes_2, capacitance in pF (Wf=300um x 350 fingers)
    Returns list of dicts {T, V, C_pF}.
    """
    axes = "axes_1" if device == "small" else "axes_2"
    scale = 1e-3 if device == "small" else 1.0  # to pF
    out = []
    for f in sorted(glob.glob(os.path.join(DIG, f"F2_{axes}_c*_K.csv"))):
        T = _temp_from_label(os.path.basename(f))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        idx = np.argsort(d[:, 0])
        out.append(dict(T=T, V=d[idx, 0], C_pF=d[idx, 1] * scale))
    return out


def load_vcro_tuning():
    """Digitized VCRO tuning curves f(Vctrl) per temperature (Fig. F11a).

    Returns list of dicts {T, V, f_MHz}."""
    out = []
    for f in sorted(glob.glob(os.path.join(DIG, "F11_axes_1_c*_K.csv"))):
        T = _temp_from_label(os.path.basename(f))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        idx = np.argsort(d[:, 0])
        out.append(dict(T=T, V=d[idx, 0], f_MHz=d[idx, 1]))
    return out


def load_phase_noise():
    """Digitized phase-noise curves (Fig. F12b): {T, f_Hz, L_dBcHz}."""
    out = []
    for f in sorted(glob.glob(os.path.join(DIG, "F12_axes_2_c*_K.csv"))):
        T = _temp_from_label(os.path.basename(f))
        d = np.loadtxt(f, delimiter=",", skiprows=1)
        idx = np.argsort(d[:, 0])
        out.append(dict(T=T, f_Hz=d[idx, 0], L_dBcHz=d[idx, 1]))
    return out


def load_block_power():
    """Digitized per-block power (Fig. F14): dict block -> (T_K, P_mW)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(DIG, "F14_axes_1_c*.csv"))):
        label = re.search(r"_c\d+_(.+)\.csv", os.path.basename(f)).group(1)
        d = np.loadtxt(f, delimiter=",", skiprows=1, ndmin=2)
        out[label] = (d[:, 0], d[:, 1])
    return out


# ----------------------------------------------------------------------
# Cadence transient corner summaries
# ----------------------------------------------------------------------

def _stable_plateau(edges, min_frac=0.15):
    """Locate the longest quasi-stationary frequency plateau.

    Splits the edge-period sequence into segments where the local period
    stays within +-3% of its running median; returns (f_Hz, rms_jitter_s,
    p2p_jitter_s, t_start, t_end) of the longest such segment.
    """
    per = np.diff(edges)
    if len(per) < 10:
        return None
    med = np.median(per)
    # smooth local period with short window
    k = max(3, len(per) // 200)
    kernel = np.ones(k) / k
    smooth = np.convolve(per, kernel, mode="same")
    best = None
    i = 0
    n = len(per)
    while i < n:
        ref = smooth[i]
        j = i
        while j < n and abs(smooth[j] - ref) < 0.03 * ref:
            ref = 0.98 * ref + 0.02 * smooth[j]
            j += 1
        if best is None or (j - i) > best[1] - best[0]:
            best = (i, j)
        i = j if j > i else i + 1
    i, j = best
    if (j - i) < min_frac * n:
        # fall back to last quarter
        i, j = int(0.75 * n), n
    seg = per[i:j]
    return dict(
        f_Hz=1.0 / np.mean(seg),
        rms_jitter_s=float(np.std(seg)),
        p2p_jitter_s=float(seg.max() - seg.min()),
        t_start=float(edges[i]),
        t_end=float(edges[j]),
        frac=float((j - i) / n),
    )


def load_corners():
    """All 14 Cadence transient corners with recomputed plateau metrics.

    Returns list of dicts sorted by temperature (K), each with:
      T_K, f_lock_MHz (plateau), rms_jitter_ns, p2p_jitter_ns,
      locked_to_ref (|f-50MHz|<1%), plateau_frac, Vc_mean, meta (raw),
      t_Vc / v_Vc decimated control-voltage waveform,
      f_vs_t (edge-derived instantaneous frequency trace).
    """
    out = []
    for f in sorted(glob.glob(os.path.join(SUM, "*.npz"))):
        z = np.load(f)
        meta = json.loads(str(z["meta"]))
        T_K = meta["temp_C"] + 273.0
        rec = dict(T_K=T_K, meta=meta, name=os.path.basename(f)[:-4])
        if "clk_edges_t" in z:
            e = np.asarray(z["clk_edges_t"], dtype=float)
            pl = _stable_plateau(e)
            if pl:
                rec["f_lock_MHz"] = pl["f_Hz"] * 1e-6
                rec["rms_jitter_ns"] = pl["rms_jitter_s"] * 1e9
                rec["p2p_jitter_ns"] = pl["p2p_jitter_s"] * 1e9
                rec["plateau_frac"] = pl["frac"]
                rec["plateau_window"] = (pl["t_start"], pl["t_end"])
                rec["locked_to_ref"] = abs(pl["f_Hz"] * 1e-6 - 50.0) < 0.5
            # instantaneous frequency trace (median filtered)
            per = np.diff(e)
            k = max(5, len(per) // 300)
            kern = np.ones(k) / k
            fs = 1.0 / np.convolve(per, kern, mode="same")
            rec["f_vs_t"] = (e[1:], fs)
        if "t_Vc" in z:
            rec["t_Vc"] = np.asarray(z["t_Vc"], dtype=float)
            rec["v_Vc"] = np.asarray(z["v_Vc"], dtype=float)
        rec["Vc_mean"] = meta.get("Vc_mean_lock")
        out.append(rec)
    out.sort(key=lambda r: r["T_K"])
    return out
