"""Shared utilities for the fdtdx optimization suite.

Submodules:

- :mod:`~fdtdx.optimization.utils.morphology` - differentiable morphological
  primitives (erosion, dilation, filters) used by fabrication constraints.
- :mod:`~fdtdx.optimization.utils.checkpoint` - save/load full optimization
  state and warm-start helpers.
- :mod:`~fdtdx.optimization.utils.cli` - argparse builder for the standard
  fdtdx optimization command-line flags.
"""

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
