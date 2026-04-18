"""Concrete ``Constraint`` subclasses for the fdtdx optimization suite.

All public classes are re-exported from :mod:`fdtdx.optimization` (and
:mod:`fdtdx`); importing them from the parent package is the preferred path.
"""

from fdtdx.optimization.constraints.connectivity import VirtualTemperatureConnectivity
from fdtdx.optimization.constraints.function import FunctionConstraint
from fdtdx.optimization.constraints.lithography import LithographyModel, OPCConstraint
from fdtdx.optimization.constraints.manufacturing import (
    MinInclusion,
    MinLineSpace,
    NoFloatingMaterial,
)
from fdtdx.optimization.constraints.physics import (
    LinearSteadyStatePDEConstraint,
    PhysicsConstraint,
)

__all__ = [
    "FunctionConstraint",
    "LinearSteadyStatePDEConstraint",
    "LithographyModel",
    "MinInclusion",
    "MinLineSpace",
    "NoFloatingMaterial",
    "OPCConstraint",
    "PhysicsConstraint",
    "VirtualTemperatureConnectivity",
]
