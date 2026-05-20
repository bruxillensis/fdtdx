"""Mechanical / electrostatic co-simulation primitives for EOM inverse design.

This subpackage provides the JAX-differentiable building blocks needed to
fold MEMS-style electrostatic actuation and the resulting mechanical
displacement into an fdtdx optimization loop.  The forward path is

    applied voltage → φ via Poisson (PoissonSolver) → Maxwell-stress force
    → mechanical displacement (ElasticityEigenmodes — modes of the actual
    geometry, recomputed every step) → first-order ε perturbation
    (apply_displacement_to_permittivity) → FDTD run on the deformed geometry.

All steps are JAX-pure so a single ``jax.grad`` differentiates the whole
chain against device parameters and any extra scalar leaves (bias voltage
etc.) carried in ``params``.

For squeeze-film damping, large-displacement / pull-in regimes, or
unstructured-mesh FEM mechanics, write a new subclass of
:class:`MechanicalModel`; nothing else in the pipeline needs to change.

See ``examples/optimize_phase_tuner.py`` for an end-to-end worked
example: a GDS-imported dual-slot MEMS phase tuner co-optimized for
phase shift, insertion loss and footprint with per-epoch pull-in
voltage tracking.
"""

from fdtdx.optimization.mechanical.base import MechanicalModel
from fdtdx.optimization.mechanical.deformation import (
    apply_displacement_to_permittivity,
    apply_finite_displacement_to_permittivity,
)
from fdtdx.optimization.mechanical.elasticity import (
    ElasticityEigenmodes,
    build_dof_mask,
    mass_operator_3d,
    stiffness_operator_3d,
)
from fdtdx.optimization.mechanical.electrostatic import PoissonSolver
from fdtdx.optimization.mechanical.modal import doubly_clamped_beam_modes
from fdtdx.optimization.mechanical.pullin import (
    advect_density,
    pull_in_voltage,
    pull_in_voltage_from_force_fn,
)

__all__ = [
    "ElasticityEigenmodes",
    "MechanicalModel",
    "PoissonSolver",
    "advect_density",
    "apply_displacement_to_permittivity",
    "apply_finite_displacement_to_permittivity",
    "build_dof_mask",
    "doubly_clamped_beam_modes",
    "mass_operator_3d",
    "pull_in_voltage",
    "pull_in_voltage_from_force_fn",
    "stiffness_operator_3d",
]
