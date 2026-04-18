"""Loss-term abstractions for the fdtdx optimization suite.

A `LossTerm` is a scheduled, named contribution to the optimization loss. Two
thin subclasses label intent:

- `Objective` - a term you want to MAXIMISE (contributes `-weight * value`).
- `Constraint` - a term you want to MINIMISE (contributes `+weight * value`).

Users either subclass these and override `compute()`, or use the convenience
wrappers :class:`~fdtdx.optimization.objectives.function.FunctionObjective`
and :class:`~fdtdx.optimization.constraints.function.FunctionConstraint`,
which accept a plain Python callable - no boilerplate needed for simple
detector-based objectives.

All terms are JIT-compatible pytrees (via `TreeClass` + `@autoinit`). The
`Optimization` orchestrator iterates `objectives` and `constraints` inside a
jitted `loss_fn`, summing `sign * schedule(epoch) * raw` where `sign` is
`-1` for objectives and `+1` for constraints.
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.optimization.schedules import ConstantSchedule, WeightSchedule


@autoinit
class LossTerm(TreeClass, ABC):
    """Base class for a scheduled, named loss contribution.

    Subclasses override `compute` to produce a scalar `raw` metric. The weight
    is supplied by `schedule(epoch)` and combined via `__call__`.
    """

    name: str = frozen_field()
    schedule: WeightSchedule = field(default=ConstantSchedule())

    @abstractmethod
    def compute(
        self,
        *,
        params,
        objects,
        arrays,
        config,
        epoch,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Compute the raw scalar metric and an info dict for this term.

        The return scalar is combined with the schedule and sign in `__call__`.
        The info dict is propagated into the Optimization's per-epoch log.
        Key names should be term-specific (e.g. include `self.name`) to avoid
        collisions when multiple terms log similarly named statistics.
        """

    def __call__(
        self,
        *,
        sign: float,
        params,
        objects,
        arrays,
        config,
        epoch,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        raw, info = self.compute(
            params=params,
            objects=objects,
            arrays=arrays,
            config=config,
            epoch=epoch,
        )
        weight = self.schedule(epoch)
        contribution = jnp.asarray(sign, dtype=jnp.float32) * weight * raw
        merged = {
            **info,
            f"{self.name}_raw": raw,
            f"{self.name}_weight": weight,
            f"{self.name}_contrib": contribution,
        }
        return contribution, merged


@autoinit
class Objective(LossTerm, ABC):
    """LossTerm contributing `-schedule(epoch) * value` to the loss.

    Use when the underlying metric is something you want to MAXIMISE
    (transmission efficiency, mode overlap, etc.). The Optimization
    orchestrator passes `sign=-1.0` when calling objectives.
    """


@autoinit
class Constraint(LossTerm, ABC):
    """LossTerm contributing `+schedule(epoch) * value` to the loss.

    Use when the underlying metric is a PENALTY you want to MINIMISE
    (fabrication violation count, disconnected-material temperature, etc.).
    The Optimization orchestrator passes `sign=+1.0` when calling constraints.
    """
