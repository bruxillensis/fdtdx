"""Callable-wrapped :class:`Constraint`.

See :class:`~fdtdx.optimization.objectives.function.FunctionObjective` for the
callable signature - :class:`FunctionConstraint` is the mirror for penalty
terms contributing with sign ``+1``.
"""

from typing import Callable

import jax

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.optimization.terms import Constraint

__all__ = [
    "FunctionConstraint",
]

MetricFn = Callable[..., tuple[jax.Array, dict[str, jax.Array]]]


@autoinit
class FunctionConstraint(Constraint):
    """A Constraint that delegates `compute` to a user-supplied function.

    See :class:`~fdtdx.optimization.objectives.function.FunctionObjective` for
    the callable signature.
    """

    fn: MetricFn = frozen_field()

    def compute(self, *, params, objects, arrays, config, epoch):
        return self.fn(
            params=params,
            objects=objects,
            arrays=arrays,
            config=config,
            epoch=epoch,
        )
