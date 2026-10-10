"""Access to constants derived from the foundry PDK.

The PDK is covered by a non-disclosure agreement.  Nothing derived from it is part of the
public repository: `tl/pdk_extract.py` reads the PDK (path in the environment variable
GANPLL_PDK_DIR) and writes `results/pdk_derived.json`, `results/devtables.json` and
`results/cv_map.json` locally; these files are listed in .gitignore.
"""
import json, os

_HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE = {}

def _load(name):
    if name not in _CACHE:
        path = os.path.join(_HERE, 'results', name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f'{path} is missing. It is derived from the foundry PDK and is not distributed; '
                'run tl/pdk_extract.py with GANPLL_PDK_DIR pointing to the PDK to create it.')
        _CACHE[name] = json.load(open(path))
    return _CACHE[name]

def constants():
    """Resistor temperature coefficients and the HEMT thermal-resistance law."""
    return _load('pdk_derived.json')

def r2deg_factor(T_K):
    c = constants()['res2deg']; d = T_K - 298.15
    return 1.0 + c['ptc1'] * d + c['ptc2'] * d * d

def rom_factor(T_K):
    c = constants()['resom']; d = T_K - 298.15
    return 1.0 + c['ptc1'] * d + c['ptc2'] * d * d

def cv_map():
    return _load('cv_map.json')

def devtables():
    return _load('devtables.json')
