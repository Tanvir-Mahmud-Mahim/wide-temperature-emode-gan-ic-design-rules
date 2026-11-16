# Changelog

## v1.1.2 (2026-11-16)
Reference check for the first revision; no simulation result or number changed.
- `scripts/31_paper_numbers.py`, benchmark table (`tab_bench`): the values of the cited works are
  those stated in their abstracts; cells not stated there are marked "n.s." (previously "n.r.",
  not reported, which could not be confirmed from the full text for every work).

## v1.1.1 (2026-11-16)
Non-disclosure redactions and corrections for the first revision of the article; no simulation was
re-run and no number in the article changed except where noted.
- Varactor bias: result files give `dvb` / `dvb_tail` (bias relative to a withheld reference) instead
  of absolute `vb` / `vb_tail`; `tl/vbref.py` restores absolute values from `GANPLL_VB_REF` for the
  scripts that reuse a closed-loop operating point. The closed-loop table lists the bias relative to
  its 300-K value (mV), and the design table the range of that difference instead of the absolute
  range.
- Delay surrogate `results/tl_vcro.json`: onset voltage `vt0` and its drift `kt` removed (read from a
  private file by `ganpll/tl_anchor.load`).
- PINN: start values of the threshold, its drift and the device area scales moved out of the code
  (`data/private/pinn_prior.json`); the 523-K extrapolation test reports the capacitance ratio
  instead of absolute capacitances (`results/pinn_results*.json`, `scripts/01_fit_pinn.py`).
- C-V figures (`fig_cv_norm`, `fig_pinn` (a) and (d), `fig_cvcheck`): voltage axis relative to the
  middle of each curve's capacitance range, within a window common to all temperatures.
- Design table: comparator thresholds printed to 0.1 K (235.5, 250.5, 290.5, 350.5 K).
- Comparison tables: the GaN ring-oscillator entry also gives the best 298-K stage delay.
- Comments and docstrings corrected: the oscillator wire load (150 fF, 2.9 ohm per stage) is an
  estimate, not a post-layout extraction (no layout was drawn); the closed loop uses the two-cell
  divide-by-2/3 divider and resistor-load comparator inverters; PINN correction bound and the
  device flag of the physics losses.

## v1.1.0 (2026-11-16)
Additions for the first revision of the article; no simulation was re-run and no earlier number changed.
- `scripts/31_paper_numbers.py` now also writes three supplementary tables computed from the released
  results: `tab_noise` (loop crossover, phase margin, phase noise and jitter of the final design at each
  temperature, with the transistor-level deterministic fractional jitter), `tab_power` (PLL-core power,
  7-V precharge rail, energy per output cycle, stand-alone 4-bit HEMT-logic modulator) and `tab_bandsel`
  (band selected between the comparator thresholds and the closed-loop runs in each interval), and the
  macros `NZpnIntR`, `NZpnFracR`, `NZpm`, `NZfc`, `TLecyc`, `TLecycMid` and `MASHtotMid`.
- Caption of the closed-loop table: the `a +- b` notation (mean static phase error and half the
  peak-to-peak wander) is defined, and the modulator is identified as a cycle-exact digital model.
- README: the modulator is described as above.

## v1.0.1 (2026-10-10)
Presentation changes only; no simulation result or number in the article changed.
- `scripts/04_make_figures.py`: FN3 (supplementary iteration-1 optimization figure) redrawn so that no
  panel title or axis label is clipped; value labels placed beside the bars.
- `scripts/30_paper_figures.py`: precharge figure (`fig_precharge`) with the `V_pre = V_DD` label
  moved clear of the data and a distinct colour per temperature.
- `scripts/31_paper_numbers.py`: divider table without the constant gate-size column and with uniform
  wording; benchmark table cells for the SiC PLL and the GaN ring oscillator restricted to values
  stated in those papers.
- Regenerated `figures/FN3.pdf`, `figures/FN3.svg`, `figures/fig_precharge.pdf`.

## v1.0.0 (2026-10-10)
First public release.
