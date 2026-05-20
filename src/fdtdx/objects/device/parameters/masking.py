"""Fixed-material mask parameter transform.
Inverse-design smoothing + projection (e.g. ``GaussianSmoothing2D`` →
``SubpixelSmoothedProjection``) blurs material boundaries.  For an
electro-opto-mechanical device some regions must stay solid material for
the *entire* optimization — the input/output waveguides that launch/collect
the mode, and the spring/anchor supports that carry the mechanical load.
Without pinning, the smoothing bleeds air into those supports (and the
optimizer has no incentive to keep them solid early on), which is both
optically lossy and mechanically wrong.
``FixedMaterialMask`` clamps the masked cells to a fixed parameter value
(default ``1.0`` = the high-index / solid material) regardless of the
upstream pipeline.  Append it **last** in ``Device.param_transforms`` so it
overrides the projected density.  Gradients are exactly zero inside the
mask (those cells are not design freedom) and pass straight through
elsewhere — no straight-through estimator needed since the op is a plain
``where``.
"""

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.objects.device.parameters.transform import SameShapeTypeParameterTransform

__all__ = ["FixedMaterialMask"]


@autoinit
class FixedMaterialMask(SameShapeTypeParameterTransform):
    """Force ``mask`` cells to ``value`` for the whole optimization.
    Parameters
    ----------
    mask:
        Boolean / 0-1 array broadcastable to each device parameter array
        (the device's matrix-voxel grid; a 2-D ``(Nx, Ny)`` mask broadcasts
        over a singleton Z, matching the other 2-D transforms).  ``True``
        cells are pinned.
    value:
        The pinned parameter value.  ``1.0`` (default) pins to the solid /
        high-index material; ``0.0`` pins to void / low-index.
    """

    mask: jax.Array = frozen_field()
    value: float = frozen_field(default=1.0)

    def __call__(
        self,
        params: dict[str, jax.Array],
        **kwargs,
    ) -> dict[str, jax.Array]:
        del kwargs
        m = self.mask.astype(bool)
        v = jnp.asarray(self.value)
        result = {}
        for k, arr in params.items():
            bm = jnp.broadcast_to(m.reshape(m.shape + (1,) * (arr.ndim - m.ndim)), arr.shape)
            result[k] = jnp.where(bm, v.astype(arr.dtype), arr)
        return result