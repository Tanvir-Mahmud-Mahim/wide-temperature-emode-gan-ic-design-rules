# Device-physics design rules for wide-temperature e-mode GaN ICs (code and results)

Code and circuit-level results for the article *"Device-physics design rules for wide-temperature
enhancement-mode gallium nitride integrated circuits, revealed by transistor-anchored machine
learning"* (T. M. Mahim, M. M. Rahman, N. Chowdhury).

The probe circuit is a monolithic e-mode GaN-on-SOI fractional-N PLL in which every block is built
from e-mode HEMTs, sized by learned models and verified by transistor-level simulation with the
foundry device model at ten temperatures from 218 to 423 K. All results are simulations; no chip
was fabricated.

## Non-disclosure agreement

The foundry process design kit (imec 200-V GaN-on-SOI) is covered by a non-disclosure agreement.
This repository contains **no** PDK file and **no** data that transcribe its device parameters or
device-level characteristics: no model cards, resistor temperature coefficients, thermal-resistance
law, device tables, absolute varactor C-V data, trained device surrogates (PINN states, prior
parameters), stand-alone pump currents, comparator calibration ratios, bias-divider resistor ratios
or netlists. Figures show device capacitance normalized to its maximum. Holders of the kit can
regenerate the private files with the scripts listed below.

## What is here

| Folder | Content |
|---|---|
| `ganpll/` | PINN varactor surrogate, deep ensembles, iteration-1 differentiable loop model (`pll.py`), adjoint optimizer, RL certificate (`gcn_sac.py`, `rl_env.py`), WGAN robustness score, transistor-anchored oscillator surrogate (`tl_anchor.py`), iteration-2 loop model (`pll2.py`), loader of the private PDK-derived constants (`pdk_local.py`) |
| `scripts/` | numbered pipeline: 01 PINN, 02 calibration, 03 adjoint co-design, 05 RL certificate, 07 WGAN score, 18 iteration-2 co-design, 20 surrogate fit, 21 PINN uncertainty, 22 feasibility search, 23 noise evaluation, 24 C-V map, 30 figures, 31 every number and table, 32 normalized C-V figure |
| `tl/` | transistor-level netlisters and testbenches: bootstrapped DCFL logic, detector, charge pump, filter, oscillator with band capacitors, comparators, two-cell divider, closed loop (`tlclosed.py`, option `--por` for the power-on acquisition aid), fractional phase analysis (`frac_residual.py`), precharge sweep, band check and speed checks; the VACASK cross-check (`ng2vacask.py`, `vacask_closed.py`), the ring-oscillator transient-noise test (`ring_noise.py`) and the stand-alone transistor-level MASH 1-1 modulator (`mash_tl.py`) |
| `results/` | model-level results; `results/tl/` holds the circuit-level transistor-level results (DOE, open loop, closed loop in ngspice and VACASK, robustness, modulator, noise test) |
| `data/` | circuit-level Spectre results of the hand-made design (transient summaries, tuning, phase-noise and power curves recovered from the vector graphics of the Spectre plots by `digitize/digitize.py`) |

## Reproducing

Circuit-level numbers, tables and figures (from the released results):

```bash
pip install -r requirements.txt
python scripts/31_paper_numbers.py     # numbers and tables (PDK-dependent items are skipped)
python scripts/30_paper_figures.py     # figures
```

With the PDK (kept private, never commit the outputs):

```bash
export GANPLL_PDK_DIR=/path/to/ganic200soi-pdk
export GANPLL_PDK=$GANPLL_PDK_DIR/models/spectre/ganic200soi_hemts.scs
python tl/pdk_extract.py        # results/pdk_derived.json
python tl/devtables.py          # results/devtables.json
python scripts/24_cv_map.py     # results/cv_map.json
cd tl && python tlclosed.py designs/final_f2.json 300 int --tstop 30e-6 --tag f2
python tlclosed.py designs/final_f2.json 300 frac --frac 0.2 --tstop 30e-6 --tag f2
python frac_residual.py ../results/tl/f2_300_frac_5.2.json
python mash_tl.py 300 --ncyc 40                 # HEMT-logic MASH 1-1 modulator, stand-alone
```

The released design files omit the bias-divider and comparator resistor values; the optimizer
(`scripts/18_tl_codesign.py`) regenerates them with the PDK. ngspice-46 must be built with OSDI
and XSPICE, and the MVSG Verilog-A model of the PDK compiled with OpenVAF.

VACASK cross-check (VACASK 0.3.4, <https://codeberg.org/arpadbuermen/VACASK>):

```bash
openvaf-r MVSG.va -o mvsg.osdi                  # the PDK's Verilog-A file, compiled locally
export VACASK=/path/to/vacask MVSG_OSDI_R=/path/to/mvsg.osdi
cd tl && python vacask_closed.py designs/final_f2.json 300 --tstop 30e-6 --tag vac
```

## License

Code: Apache-2.0 (see `LICENSE`). The foundry PDK and anything derived from it at device level are
not included.
