"""Differentiable morphological primitives for fabrication constraints.

All operations treat the last two axes as spatial (``..., Ny, Nx``) and
broadcast trivially over any leading axes, so they work on 2D patterns, 3D
layer stacks, or batched collections alike.

The erosion / dilation primitives use the classic Sigmund-Wang **filter and
project** scheme: apply a smooth linear filter, then push the filtered value
through a sigmoid centred at :math:`\\eta`.  For :math:`\\eta > 0.5` (default
``0.75``) the operation acts as an **erosion** - only regions where the
filter response is near 1 survive, so thin solid features are suppressed.
For :math:`\\eta < 0.5` (default ``0.25``) it acts as a **dilation** - any
neighbourhood with appreciable solid content lights up.

This matches the convention used elsewhere in fdtdx (``GaussianSmoothing2D`` +
``TanhProjection`` / ``SubpixelSmoothedProjection``) and has bounded, smooth
gradients, which is important for gradient-based topology optimization.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

__all__ = [
    "box_filter_2d",
    "gaussian_filter_2d",
    "smooth_erosion",
    "smooth_dilation",
    "meters_to_odd_kernel",
]


def _spatial_window(ndim: int, kernel_size: int) -> tuple[int, ...]:
    """Window shape for reduce_window that spans only the last two axes."""
    if ndim < 2:
        raise ValueError(f"Expected at least 2 spatial axes, got ndim={ndim}")
    return (1,) * (ndim - 2) + (kernel_size, kernel_size)


def box_filter_2d(rho: jax.Array, kernel_size: int) -> jax.Array:
    """Edge-aware uniform (box) filter on the last two axes.

    For each spatial location, returns the mean of ``rho`` inside a
    ``kernel_size x kernel_size`` window centred on that pixel.  Pixels near
    the boundary are divided by the *actual* number of contributing cells
    (smaller than ``kernel_size**2``), so the filter response at edges is not
    biased toward zero.

    Parameters
    ----------
    rho:
        Density-like array with at least two axes.  The last two are treated
        as spatial; all leading axes are broadcast as batch.
    kernel_size:
        Odd positive integer.  Window is square.
    """
    if kernel_size < 1:
        raise ValueError(f"kernel_size must be >= 1, got {kernel_size}")
    if kernel_size == 1:
        return rho
    if rho.ndim < 2:
        raise ValueError(f"box_filter_2d requires ndim >= 2, got {rho.shape}")
    window = _spatial_window(rho.ndim, kernel_size)
    strides = (1,) * rho.ndim
    summed = jax.lax.reduce_window(rho, 0.0, jax.lax.add, window, strides, "SAME")
    ones = jnp.ones_like(rho)
    count = jax.lax.reduce_window(ones, 0.0, jax.lax.add, window, strides, "SAME")
    return summed / count


def _gaussian_1d_kernel(sigma: float, truncate: float) -> jax.Array:
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = jnp.arange(-radius, radius + 1, dtype=jnp.float32)
    k = jnp.exp(-(x**2) / (2.0 * max(sigma, 1e-12) ** 2))
    return k / k.sum()


def gaussian_filter_2d(
    rho: jax.Array,
    sigma: float,
    *,
    truncate: float = 3.0,
) -> jax.Array:
    """Separable Gaussian filter on the last two axes.

    Uses zero-padding at the boundary.  For strongly normalised
    (density-style) inputs the edge bias is small; if you need an edge-aware
    filter, use :func:`box_filter_2d` instead.
    """
    if rho.ndim < 2:
        raise ValueError(f"gaussian_filter_2d requires ndim >= 2, got {rho.shape}")
    if sigma <= 0.0:
        return rho

    k = _gaussian_1d_kernel(sigma, truncate)
    radius = (k.size - 1) // 2

    leading = rho.shape[:-2]
    ny, nx = rho.shape[-2:]
    x = rho.reshape((-1, 1, ny, nx)).astype(jnp.float32)

    # Convolve along X.
    kx = k.reshape((1, 1, 1, -1))
    x = jax.lax.conv_general_dilated(
        x,
        kx,
        window_strides=(1, 1),
        padding=((0, 0), (radius, radius)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    # Convolve along Y.
    ky = k.reshape((1, 1, -1, 1))
    x = jax.lax.conv_general_dilated(
        x,
        ky,
        window_strides=(1, 1),
        padding=((radius, radius), (0, 0)),
        dimension_numbers=("NCHW", "OIHW", "NCHW"),
    )
    return x.reshape(leading + (ny, nx))


def smooth_erosion(
    rho: jax.Array,
    kernel_size: int,
    *,
    beta: float = 8.0,
    eta: float = 0.75,
) -> jax.Array:
    """Differentiable morphological erosion via filter + sigmoid projection.

    A pixel survives the erosion only if its :math:`k \\times k`
    neighbourhood is mostly solid.  Formally::

        rho_eroded = sigmoid(beta * (box_filter_2d(rho, k) - eta))

    With ``eta`` close to 1, the output is near-1 only in solid regions whose
    filter response is near 1 (i.e. whose *entire* neighbourhood is solid).
    Thin features are therefore eroded away.

    Parameters
    ----------
    rho:
        Density in [0, 1] on a 2D or batched spatial grid.
    kernel_size:
        Odd positive integer.  The erosion "radius" is ``kernel_size // 2``
        voxels.
    beta:
        Sigmoid sharpness.  Larger ``beta`` -> sharper transition, smaller
        gradients away from ``eta``.
    eta:
        Sigmoid centre.  For erosion, should be > 0.5 (default 0.75).
    """
    filt = box_filter_2d(rho, kernel_size)
    return jax.nn.sigmoid(beta * (filt - eta))


def smooth_dilation(
    rho: jax.Array,
    kernel_size: int,
    *,
    beta: float = 8.0,
    eta: float = 0.25,
) -> jax.Array:
    """Differentiable morphological dilation via filter + sigmoid projection.

    A pixel lights up whenever its :math:`k \\times k` neighbourhood
    contains appreciable material::

        rho_dilated = sigmoid(beta * (box_filter_2d(rho, k) - eta))

    With ``eta`` close to 0, the output is near-1 even when only a fraction
    of the neighbourhood is solid - the "solid" region effectively grows by
    ``kernel_size // 2`` voxels.
    """
    filt = box_filter_2d(rho, kernel_size)
    return jax.nn.sigmoid(beta * (filt - eta))


def meters_to_odd_kernel(length_m: float, voxel_pitch_m: float) -> int:
    """Convert a physical length to an odd, >=3 kernel size in voxels.

    Used by manufacturing constraints to translate ``min_line_width_m`` or
    ``min_space_m`` into a concrete box-filter / morphology window size on
    the device's XY design-voxel grid.
    """
    if voxel_pitch_m <= 0.0:
        raise ValueError(f"voxel_pitch_m must be > 0, got {voxel_pitch_m}")
    if length_m <= 0.0:
        raise ValueError(f"length_m must be > 0, got {length_m}")
    k = int(round(length_m / voxel_pitch_m))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1
    return k
