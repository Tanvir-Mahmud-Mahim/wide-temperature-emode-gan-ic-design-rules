"""Absolute varactor bias of a released closed-loop result.

The released result files give V_b only relative to a withheld reference
(key ``dvb_tail``/``dvb``): together with the temperature law, absolute values
would disclose PDK device data (non-disclosure agreement).  Files produced by
re-running ``tl/tlclosed.py`` with the PDK contain the absolute ``vb_tail``.
To reuse a released file, set GANPLL_VB_REF to the 300-K integer-mode V_b of
your own run of ``tl/tlclosed.py``."""
import os


def vb_abs(r, key='vb_tail'):
    if key in r:
        return r[key]
    ref = os.environ.get('GANPLL_VB_REF')
    if ref is None:
        raise RuntimeError('result gives V_b relative to a withheld reference; set GANPLL_VB_REF (see tl/vbref.py)')
    return r['d' + key] + float(ref)
