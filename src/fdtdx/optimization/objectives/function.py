"""Callable-wrapped :class:`Objective`.

Lets users register a simple detector-based metric without subclassing -
see :class:`FunctionObjective` for the callable signature.
"""

from typing import Callable

import jax

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.optimization.terms import Objective

__all__ = [
    "FunctionObjective",
]

# User-supplied metric functions take the full compute context by keyword and
# return (scalar, info_dict).
MetricFn = Callable[..., tuple[jax.Array, dict[str, jax.Array]]]


@autoinit
class FunctionObjective(Objective):
    """An Objective that delegates `compute` to a user-supplied function.

    Example:

        def flux_eff(*, params, objects, arrays, config, epoch):
            down = arrays.detector_states["down"]["poynting_flux"].sum()
            in_ = arrays.detector_states["in"]["poynting_flux"].sum()
            val = down / jnp.maximum(in_, 1e-30)
            return val, {}

        obj = FunctionObjective(
            name="flux_down",
            schedule=ConstantSchedule(value=0.5),
            fn=flux_eff,
        )
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
