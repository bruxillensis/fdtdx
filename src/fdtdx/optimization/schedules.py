"""Epoch-gated weight schedules for optimization loss terms.

Each :class:`WeightSchedule` is a callable pytree that maps a scalar epoch
(Python ``int`` or ``jax.Array``) to a ``jax.Array`` scalar weight. Weights are
zero outside the ``[epoch_start, epoch_end]`` window; inside the window each
concrete subclass defines the interpolation rule (constant, linear,
exponential, cosine, on/off).

All schedules are JIT-safe: they avoid Python branching on traced values by
composing with :func:`jax.numpy.where`. They can therefore be used inside a
``jax.jit`` compiled loss function where ``epoch`` is a traced value.

Typical use::

    from fdtdx.optimization.schedules import LinearSchedule

    schedule = LinearSchedule(
        start_value=0.0,
        end_value=1.0,
        epoch_start=100,
        epoch_end=400,
    )
    weight = schedule(epoch)  # jax.Array scalar
"""

from abc import ABC, abstractmethod

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field


@autoinit
class WeightSchedule(TreeClass, ABC):
    """Epoch-gated weight schedule for an optimization loss term.

    Returns ``0.0`` when ``epoch`` is outside ``[epoch_start, epoch_end]``.
    When ``epoch_end`` is ``None`` the active window is unbounded above and
    the schedule will never turn off past ``epoch_start``.

    Attributes:
        epoch_start: Inclusive lower bound of the active window.
        epoch_end: Inclusive upper bound of the active window, or ``None`` for
            an unbounded window.
    """

    epoch_start: int = frozen_field(default=0)
    epoch_end: int | None = frozen_field(default=None)

    def __call__(self, epoch) -> jax.Array:
        """Evaluate the schedule at the given epoch.

        Args:
            epoch: Python ``int`` or ``jax.Array`` scalar.

        Returns:
            A ``jax.Array`` scalar weight (``float32``).
        """
        epoch_f = jnp.asarray(epoch, dtype=jnp.float32)
        start = jnp.asarray(self.epoch_start, dtype=jnp.float32)
        is_bounded = self.epoch_end is not None
        end = jnp.asarray(
            self.epoch_end if is_bounded else jnp.inf,
            dtype=jnp.float32,
        )
        # Normalized time t in [0, 1] across the active window. When
        # `epoch_end is None` the window is unbounded: the ramp saturates at
        # `end_value` once `epoch >= epoch_start` (rather than infinitely
        # slow-ramping forever). Clamp the denominator to avoid NaN when
        # start == end (zero-length window).
        if is_bounded:
            denom = jnp.maximum(end - start, 1.0)
            t = jnp.clip((epoch_f - start) / denom, 0.0, 1.0)
        else:
            t = jnp.where(epoch_f >= start, 1.0, 0.0)
        active = (epoch_f >= start) & (epoch_f <= end)
        value = self._value_at(t)
        return jnp.where(active, value, jnp.asarray(0.0, dtype=jnp.float32))

    @abstractmethod
    def _value_at(self, t: jax.Array) -> jax.Array:
        """Given normalized time ``t`` in ``[0, 1]``, return the raw weight value."""


@autoinit
class ConstantSchedule(WeightSchedule):
    """Holds a constant ``value`` inside the active window, ``0`` outside."""

    value: float = frozen_field(default=1.0)

    def _value_at(self, t):
        del t
        return jnp.asarray(self.value, dtype=jnp.float32)


@autoinit
class LinearSchedule(WeightSchedule):
    """Linearly interpolates ``start_value`` -> ``end_value`` across the window."""

    start_value: float = frozen_field(default=0.0)
    end_value: float = frozen_field(default=1.0)

    def _value_at(self, t):
        s = jnp.asarray(self.start_value, dtype=jnp.float32)
        e = jnp.asarray(self.end_value, dtype=jnp.float32)
        return s + (e - s) * t


@autoinit
class ExponentialSchedule(WeightSchedule):
    """Exponentially interpolates ``start_value`` -> ``end_value``.

    Computed as ``start * (end / start) ** t`` via log-linear interpolation.
    Both ``start_value`` and ``end_value`` must be strictly positive. For
    sign-flipping ramps use :class:`LinearSchedule`.
    """

    start_value: float = frozen_field(default=1e-3)
    end_value: float = frozen_field(default=1.0)

    def _value_at(self, t):
        s = jnp.asarray(self.start_value, dtype=jnp.float32)
        e = jnp.asarray(self.end_value, dtype=jnp.float32)
        # Log-linear interpolation is numerically stabler than (end/start)**t
        # when the ratio spans many orders of magnitude.
        log_val = jnp.log(s) + (jnp.log(e) - jnp.log(s)) * t
        return jnp.exp(log_val)


@autoinit
class CosineSchedule(WeightSchedule):
    """Cosine-eased interpolation from ``start_value`` to ``end_value``.

    The easing function ``(1 - cos(pi * t)) / 2`` maps ``t in [0, 1]`` to
    ``[0, 1]`` with zero derivative at both endpoints, giving a smooth ramp.
    """

    start_value: float = frozen_field(default=0.0)
    end_value: float = frozen_field(default=1.0)

    def _value_at(self, t):
        s = jnp.asarray(self.start_value, dtype=jnp.float32)
        e = jnp.asarray(self.end_value, dtype=jnp.float32)
        eased = 0.5 * (1.0 - jnp.cos(jnp.pi * t))
        return s + (e - s) * eased


@autoinit
class OnOffSchedule(WeightSchedule):
    """Binary on/off schedule.

    Emits ``value`` inside the active window and ``0`` outside. Semantically
    equivalent to :class:`ConstantSchedule` but named for documentation
    clarity when the intent is a hard switch rather than a held level.
    """

    value: float = frozen_field(default=1.0)

    def _value_at(self, t):
        del t
        return jnp.asarray(self.value, dtype=jnp.float32)
