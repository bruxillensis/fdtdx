"""Inverse-design optimization suite for fdtdx.

Public API entry points for building and running a scheduled, multi-term
gradient-descent optimization over fdtdx devices.

Layout of this package:

- :mod:`~fdtdx.optimization.optimization` - the ``Optimization`` driver.
- :mod:`~fdtdx.optimization.terms` - abstract ``LossTerm`` / ``Objective`` /
  ``Constraint`` bases.
- :mod:`~fdtdx.optimization.schedules` - epoch-gated weight schedules.
- :mod:`~fdtdx.optimization.constraints` - concrete ``Constraint`` classes
  (function wrapper, physics / ODE bases, connectivity, lithography,
  manufacturing DRC).
- :mod:`~fdtdx.optimization.objectives` - concrete ``Objective`` classes
  (function wrapper; room for built-in objectives).
- :mod:`~fdtdx.optimization.utils` - morphology primitives, checkpoint I/O,
  CLI argparse builder.

Typical use::

    import fdtdx

    parser = fdtdx.build_arg_parser()
    args = parser.parse_args()

    opt = fdtdx.Optimization(
        objects=objects,
        arrays=arrays,
        params=params,
        config=config,
        simulate_fn=simulate_fn,
        objectives=(obj_flux, obj_overlap),
        constraints=(connectivity_penalty,),
        optimizer=optax.adam(1e-2),
        total_epochs=500,
        logger=logger,
    )
    opt.run(
        key=key,
        seed_from=args.seed_from,
        seed_iter=args.seed_iter,
        resume_from=args.resume_from,
    )
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
from fdtdx.optimization.objectives.function import FunctionObjective
from fdtdx.optimization.optimization import Optimization
from fdtdx.optimization.schedules import (
    ConstantSchedule,
    CosineSchedule,
    ExponentialSchedule,
    LinearSchedule,
    OnOffSchedule,
    WeightSchedule,
)
from fdtdx.optimization.terms import (
    Constraint,
    LossTerm,
    Objective,
)
from fdtdx.optimization.utils.checkpoint import (
    load_checkpoint,
    load_seed_params,
    save_checkpoint,
)
from fdtdx.optimization.utils.cli import build_arg_parser
from fdtdx.optimization.utils.morphology import (
    box_filter_2d,
    gaussian_filter_2d,
    meters_to_odd_kernel,
    smooth_dilation,
    smooth_erosion,
)

__all__ = [
    "ConstantSchedule",
    "Constraint",
    "CosineSchedule",
    "ExponentialSchedule",
    "FunctionConstraint",
    "FunctionObjective",
    "LinearSchedule",
    "LinearSteadyStatePDEConstraint",
    "LithographyModel",
    "LossTerm",
    "MinInclusion",
    "MinLineSpace",
    "NoFloatingMaterial",
    "OPCConstraint",
    "Objective",
    "OnOffSchedule",
    "Optimization",
    "PhysicsConstraint",
    "VirtualTemperatureConnectivity",
    "WeightSchedule",
    "box_filter_2d",
    "build_arg_parser",
    "gaussian_filter_2d",
    "load_checkpoint",
    "load_seed_params",
    "meters_to_odd_kernel",
    "save_checkpoint",
    "smooth_dilation",
    "smooth_erosion",
]
