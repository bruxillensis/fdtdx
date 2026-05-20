"""Unit tests for :mod:`fdtdx.optimization.utils.morphology`."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.optimization.utils.morphology import (
    box_filter_2d,
    gaussian_filter_2d,
    meters_to_odd_kernel,
    smooth_dilation,
    smooth_erosion,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# box_filter_2d
# ---------------------------------------------------------------------------


def _numpy_box_filter(a: np.ndarray, k: int) -> np.ndarray:
    """Edge-aware box filter: mean over the k x k neighbourhood, counting
    only in-bounds cells (matches the count-normalized jax implementation)."""
    r = k // 2
    ny, nx = a.shape[-2:]
    out = np.zeros_like(a, dtype=np.float32)
    ones = np.ones_like(a, dtype=np.float32)
    count = np.zeros_like(a, dtype=np.float32)
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            y_src = slice(max(0, -dy), ny - max(0, dy))
            x_src = slice(max(0, -dx), nx - max(0, dx))
            y_dst = slice(max(0, dy), ny - max(0, -dy))
            x_dst = slice(max(0, dx), nx - max(0, -dx))
            out[..., y_dst, x_dst] += a[..., y_src, x_src]
            count[..., y_dst, x_dst] += ones[..., y_src, x_src]
    return out / count


def test_box_filter_matches_numpy_reference() -> None:
    rng = np.random.default_rng(0)
    rho = rng.uniform(0.0, 1.0, size=(12, 15)).astype(np.float32)
    for k in (1, 3, 5, 7):
        jax_out = np.asarray(box_filter_2d(jnp.asarray(rho), k))
        np_out = _numpy_box_filter(rho, k)
        assert np.allclose(jax_out, np_out, atol=1e-5), f"mismatch at k={k}"


def test_box_filter_preserves_uniform() -> None:
    rho = jnp.full((10, 10), 0.7, dtype=jnp.float32)
    out = box_filter_2d(rho, kernel_size=5)
    assert jnp.allclose(out, 0.7, atol=1e-6)


def test_box_filter_broadcasts_leading_axes() -> None:
    rng = np.random.default_rng(1)
    rho = jnp.asarray(rng.uniform(size=(3, 2, 8, 8)).astype(np.float32))
    out = box_filter_2d(rho, kernel_size=3)
    assert out.shape == rho.shape
    per_slice = box_filter_2d(rho[0, 0], kernel_size=3)
    assert jnp.allclose(out[0, 0], per_slice, atol=1e-6)


def test_box_filter_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        box_filter_2d(jnp.zeros((4, 4)), kernel_size=0)
    with pytest.raises(ValueError):
        box_filter_2d(jnp.zeros((5,)), kernel_size=3)


# ---------------------------------------------------------------------------
# gaussian_filter_2d
# ---------------------------------------------------------------------------


def test_gaussian_filter_zero_sigma_is_identity() -> None:
    rho = jnp.asarray(np.random.default_rng(2).uniform(size=(8, 8)).astype(np.float32))
    out = gaussian_filter_2d(rho, sigma=0.0)
    assert jnp.allclose(out, rho, atol=1e-6)


def test_gaussian_filter_preserves_mass_on_delta() -> None:
    """A centered delta impulse should spread but (approximately) conserve
    mass away from the edges.  Test on a grid large enough that the kernel
    support fits inside."""
    a = jnp.zeros((21, 21), dtype=jnp.float32).at[10, 10].set(1.0)
    out = gaussian_filter_2d(a, sigma=1.0)
    assert out[10, 10] > 0.0
    assert out[10, 10] < 1.0
    # Mass is approximately conserved (edges may leak a tiny amount).
    assert float(out.sum()) == pytest.approx(1.0, rel=5e-3)


# ---------------------------------------------------------------------------
# Morphology: erosion shrinks, dilation grows
# ---------------------------------------------------------------------------


def _disk(n: int, radius: float) -> jax.Array:
    yy, xx = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    cy = cx = (n - 1) / 2.0
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    return jnp.asarray((r <= radius).astype(np.float32))


def test_smooth_erosion_saturates_on_interior() -> None:
    rho = jnp.ones((9, 9), dtype=jnp.float32)
    out = smooth_erosion(rho, kernel_size=3, beta=8.0, eta=0.75)
    # Interior cells see a fully-solid neighbourhood, so sigmoid of
    # beta * (1 - eta) = 8 * 0.25 = 2 => ~0.88.
    assert float(out[4, 4]) > 0.85


def test_smooth_erosion_shrinks_disk_monotonically() -> None:
    rho = _disk(n=31, radius=6.0)
    small = smooth_erosion(rho, kernel_size=3, beta=20.0, eta=0.75)
    big = smooth_erosion(rho, kernel_size=5, beta=20.0, eta=0.75)
    # More erosion with a wider kernel => less total mass survives.
    assert float(small.sum()) > float(big.sum())
    # Both should remove mass compared to the original disk.
    assert float(small.sum()) < float(rho.sum())


def test_smooth_dilation_grows_disk() -> None:
    rho = _disk(n=31, radius=3.0)
    small = smooth_dilation(rho, kernel_size=3, beta=20.0, eta=0.25)
    big = smooth_dilation(rho, kernel_size=5, beta=20.0, eta=0.25)
    assert float(big.sum()) > float(small.sum())
    assert float(small.sum()) > float(rho.sum())


def test_morphology_gradients_finite() -> None:
    rho = _disk(n=17, radius=4.0)

    def loss_erode(rho: jax.Array) -> jax.Array:
        return jnp.mean(smooth_erosion(rho, kernel_size=3))

    def loss_dilate(rho: jax.Array) -> jax.Array:
        return jnp.mean(smooth_dilation(rho, kernel_size=3))

    grads_e = jax.grad(loss_erode)(rho)
    grads_d = jax.grad(loss_dilate)(rho)
    assert bool(jnp.all(jnp.isfinite(grads_e)))
    assert bool(jnp.all(jnp.isfinite(grads_d)))
    assert bool(jnp.any(grads_e != 0.0))
    assert bool(jnp.any(grads_d != 0.0))


# ---------------------------------------------------------------------------
# meters_to_odd_kernel
# ---------------------------------------------------------------------------


def test_meters_to_odd_kernel() -> None:
    # Length = 5 * pitch => 5 (already odd).
    assert meters_to_odd_kernel(5e-9, 1e-9) == 5
    # Length = 6 * pitch => 6 rounds up to 7.
    assert meters_to_odd_kernel(6e-9, 1e-9) == 7
    # Very small length clamps to 3 (minimum usable kernel).
    assert meters_to_odd_kernel(0.5e-9, 1e-9) == 3
    # 4.4 rounds to 4 -> bumped to 5.
    assert meters_to_odd_kernel(4.4e-9, 1e-9) == 5


def test_meters_to_odd_kernel_rejects_invalid() -> None:
    with pytest.raises(ValueError):
        meters_to_odd_kernel(1e-9, 0.0)
    with pytest.raises(ValueError):
        meters_to_odd_kernel(0.0, 1e-9)
