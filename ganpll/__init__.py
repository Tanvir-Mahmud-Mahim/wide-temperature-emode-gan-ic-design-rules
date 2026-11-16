"""ganpll: Physics-informed ML surrogate modeling and adjoint co-design
for a monolithic e-mode GaN-on-SOI PLL with GaN HEMT varactors.

Modules
-------
physics  : analytic charge-control varactor C(V,T) prior and temperature laws
data     : loaders for digitized Cadence Spectre curves and transient corner summaries
pinn     : physics-informed neural network surrogate C(V,T,geometry) in JAX
pll      : differentiable behavioral PLL model (phase-domain, 3rd-order loop filter)
adjoint  : reverse-mode (discrete adjoint) gradient optimization of PLL design parameters
plots    : IEEE-style publication figures
"""
__version__ = "1.0.0"
