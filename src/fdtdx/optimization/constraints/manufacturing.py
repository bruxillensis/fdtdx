"""Manufacturing (DRC) constraints for inverse-design devices.

Three design-rule-checker style penalties commonly required by fabrication:

- :class:`MinLineSpace` - no solid features thinner than ``min_line_width``
  and no gaps narrower than ``min_space``, per-XY-plane.

- :class:`MinInclusion` - an *inner* device layer must lie within an
  *outer* device layer, eroded by at least ``min_margin``.  Typical use:
  a doped / silicided region must be entirely inside a wider metal pad.

- :class:`NoFloatingMaterial` - in a stack of two or more aligned devices
  (bottom -> top) representing successive etch depths, an upper layer
  cannot carry solid material above a void in the layer immediately below
  it.  This formalizes the inline ``fab_penalty`` loop from
  ``examples/optimize_emitter.py``.

All three derive from :class:`~fdtdx.optimization.terms.Constraint` and
read each involved device's post-projection density via the standard
``device(params[name])`` pipeline.  Gradients flow to the device
parameters through the morphology primitives in
:mod:`fdtdx.optimization.utils.morphology` and the vectorized ReLU comparisons.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.optimization.constraints.physics import _get_device, _squeeze_trailing_singletons
from fdtdx.optimization.terms import Constraint
from fdtdx.optimization.utils.morphology import (
    meters_to_odd_kernel,
    smooth_dilation,
    smooth_erosion,
)

__all__ = [
    "MinLineSpace",
    "MinInclusion",
    "NoFloatingMaterial",
]


def _rho_for(
    objects: Any,
    params: Any,
    device_name: str,
) -> tuple[jax.Array, float]:
    """Evaluate a device's post-projection density and return the XY pitch."""
    device = _get_device(objects, device_name)
    rho = device(params[device_name])
    rho = _squeeze_trailing_singletons(rho)
    rho = jnp.clip(rho, 0.0, 1.0).astype(jnp.float32)
    pitch = float(device.single_voxel_real_shape[0])
    return rho, pitch


@autoinit
class MinLineSpace(Constraint):
    """Penalize solid features thinner than ``min_line_width_m`` or gaps
    narrower than ``min_space_m``.

    Uses the Sigmund-Wang feature-size projection::

        eroded  = smooth_erosion(rho,  k_line,  eta=eta_erode)
        dilated = smooth_dilation(rho, k_space, eta=eta_dilate)
        thin_line = rho        * (1 - eroded)     # solid but neighbourhood isn't
        thin_gap  = (1 - rho)  * dilated          # void but neighbourhood has solid

    The penalty is::

        mean(thin_line^2) + mean(thin_gap^2)

    Window sizes are translated from meters into odd voxel counts using
    the device's XY voxel pitch.  The operation is applied on the last
    two axes, so a multi-Z-slice device (e.g. ``(Nx, Ny, Nz)``) is
    broadcast over Z.
    """

    device_name: str = frozen_field()
    min_line_width_m: float = frozen_field()
    min_space_m: float = frozen_field()
    beta: float = frozen_field(default=8.0)
    eta_erode: float = frozen_field(default=0.75)
    eta_dilate: float = frozen_field(default=0.25)

    def compute(
        self,
        *,
        params: Any,
        objects: Any,
        **_unused: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        rho, pitch = _rho_for(objects, params, self.device_name)
        k_line = meters_to_odd_kernel(self.min_line_width_m, pitch)
        k_space = meters_to_odd_kernel(self.min_space_m, pitch)
        eroded = smooth_erosion(rho, k_line, beta=self.beta, eta=self.eta_erode)
        dilated = smooth_dilation(rho, k_space, beta=self.beta, eta=self.eta_dilate)
        thin_line = rho * (1.0 - eroded)
        thin_gap = (1.0 - rho) * dilated
        line_penalty = jnp.mean(thin_line**2)
        gap_penalty = jnp.mean(thin_gap**2)
        penalty = line_penalty + gap_penalty
        info = {
            f"{self.name}_thin_line": line_penalty,
            f"{self.name}_thin_gap": gap_penalty,
        }
        return penalty, info


@autoinit
class MinInclusion(Constraint):
    """Penalize where ``inner_device`` has material outside the eroded
    ``outer_device`` boundary by at least ``min_margin_m``.

    Concretely::

        violation = rho_inner * (1 - smooth_erosion(rho_outer, k_margin))
        penalty   = mean(violation^2)

    The inner and outer devices must share the same XY design-voxel grid
    (same ``matrix_voxel_grid_shape[:2]`` and ``single_voxel_real_shape[0]``).
    The Z dimension is broadcast; a single-slab inner over a multi-slab
    outer is the common case.
    """

    inner_device_name: str = frozen_field()
    outer_device_name: str = frozen_field()
    min_margin_m: float = frozen_field()
    beta: float = frozen_field(default=8.0)
    eta: float = frozen_field(default=0.75)

    def compute(
        self,
        *,
        params: Any,
        objects: Any,
        **_unused: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        rho_inner, _ = _rho_for(objects, params, self.inner_device_name)
        rho_outer, pitch_outer = _rho_for(objects, params, self.outer_device_name)
        if rho_inner.shape[-2:] != rho_outer.shape[-2:]:
            raise ValueError(
                "MinInclusion: inner and outer devices must share XY grid; "
                f"got inner {rho_inner.shape[-2:]} vs outer {rho_outer.shape[-2:]}"
            )
        k = meters_to_odd_kernel(self.min_margin_m, pitch_outer)
        outer_eroded = smooth_erosion(rho_outer, k, beta=self.beta, eta=self.eta)
        violation = rho_inner * (1.0 - outer_eroded)
        penalty = jnp.mean(violation**2)
        info = {
            f"{self.name}_max_violation": jnp.max(violation),
            f"{self.name}_mean_violation": penalty,
        }
        return penalty, info


@autoinit
class NoFloatingMaterial(Constraint):
    """Penalize solid material stacked above a void in the next lower layer.

    For a vertical stack of devices named ``device_stack_names`` (bottom to
    top), the penalty is

    .. math::

        \\sum_{i=0}^{N-2} \\operatorname{mean}\\bigl(
            \\operatorname{ReLU}(\\rho_{i+1} - \\rho_i)^2
        \\bigr),

    which is zero iff each upper layer has no more material than the layer
    directly beneath it - i.e. the stack is monotonically non-increasing
    from bottom to top, so no solid sits unsupported over a void.

    All devices in the stack must share the same XY design-voxel grid
    (same ``matrix_voxel_grid_shape[:2]``).  Each device is expected to
    have a single Z voxel per layer (typical for multi-etch-depth designs)
    but any matching Z shape is accepted.

    Formalizes the inline ``fab_penalty`` pattern in
    ``examples/optimize_emitter.py``.
    """

    device_stack_names: tuple[str, ...] = frozen_field()

    def compute(
        self,
        *,
        params: Any,
        objects: Any,
        **_unused: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        if len(self.device_stack_names) < 2:
            raise ValueError(
                f"NoFloatingMaterial requires at least two devices in the stack; got {self.device_stack_names!r}"
            )
        rhos: list[jax.Array] = []
        for name in self.device_stack_names:
            rho, _ = _rho_for(objects, params, name)
            rhos.append(rho)

        # Validate that all layers share shape so the element-wise diff is well-defined.
        ref = rhos[0].shape
        for r, name in zip(rhos, self.device_stack_names):
            if r.shape != ref:
                raise ValueError(
                    "NoFloatingMaterial: every device in the stack must share shape; "
                    f"{name!r} has shape {r.shape}, expected {ref}"
                )

        total = jnp.asarray(0.0, dtype=jnp.float32)
        max_excess = jnp.asarray(0.0, dtype=jnp.float32)
        for below, above in zip(rhos[:-1], rhos[1:]):
            excess = jax.nn.relu(above - below)
            total = total + jnp.mean(excess**2)
            max_excess = jnp.maximum(max_excess, jnp.max(excess))

        info = {
            f"{self.name}_max_excess": max_excess,
            f"{self.name}_penalty": total,
        }
        return total, info
