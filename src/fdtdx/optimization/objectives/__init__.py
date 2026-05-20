"""Concrete ``Objective`` subclasses for the fdtdx optimization suite.

All public classes are re-exported from :mod:`fdtdx.optimization` (and
:mod:`fdtdx`); importing them from the parent package is the preferred path.
"""

from fdtdx.optimization.objectives.function import FunctionObjective

__all__ = [
    "FunctionObjective",
]
