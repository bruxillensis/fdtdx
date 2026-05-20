"""Unit tests for the static pull-in (limit-point) continuation.

Validated against the closed-form ideal parallel-plate result: with elastic
spring ``k``, gap ``g``, plate area ``A`` and electrostatic force per V²
``f(s) = ε₀ A / (2 (g − s)²)``, the static balance ``k s = V² f(s)`` has its
limit point at ``s* = g/3`` with

    V_pull_in = √( 8 k g³ / (27 ε₀ A) ).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.constants import eps0
from fdtdx.optimization.mechanical.pullin import (
    advect_density,
    pull_in_voltage,
    pull_in_voltage_from_force_fn,
)

pytestmark = pytest.mark.unit


def _analytic(k: float, g: float, A: float) -> tuple[float, float]:
    v_pi = float(np.sqrt(8.0 * k * g**3 / (27.0 * eps0 * A)))
    return v_pi, g / 3.0


def test_pull_in_matches_parallel_plate_closed_form() -> None:
    k, g, A = 5.0, 1.0e-6, 4.0e-12  # N/m, m, m²
    s = jnp.linspace(g / 800.0, 0.9 * g, 800, dtype=jnp.float32)
    f = eps0 * A / (2.0 * (g - s) ** 2)
    v_pi, v_branch = pull_in_voltage(s, jnp.asarray(k), f)

    v_pi_an, s_star = _analytic(k, g, A)
    assert abs(float(v_pi) - v_pi_an) / v_pi_an < 5e-3  # sampling-limited
    # limit point at s ≈ g/3
    s_at_max = float(s[int(jnp.argmax(v_branch))])
    assert abs(s_at_max - s_star) / s_star < 2e-2
    # branch rises then falls (a genuine interior maximum).
    assert float(v_branch[0]) < float(v_pi)
    assert float(v_branch[-1]) < float(v_pi)


def test_pull_in_sqrt_k_scaling_and_gradient() -> None:
    """V_pi ∝ √k ⇒ dV_pi/dk = V_pi/(2k); the envelope-theorem max yields it."""
    g, A = 8.0e-7, 2.0e-12
    s = jnp.linspace(g / 600.0, 0.9 * g, 600, dtype=jnp.float32)

    def v_pi_of_k(k):
        f = eps0 * A / (2.0 * (g - s) ** 2)
        vp, _ = pull_in_voltage(s, k, f)
        return vp

    k0 = jnp.asarray(7.0, dtype=jnp.float32)
    v0 = v_pi_of_k(k0)
    grad = jax.grad(v_pi_of_k)(k0)
    assert jnp.isfinite(grad) and float(grad) > 0.0
    # analytic envelope derivative
    assert abs(float(grad) - float(v0) / (2.0 * float(k0))) / float(v0 / (2 * k0)) < 1e-2


def test_pull_in_from_force_fn_wrapper() -> None:
    k, g, A = 3.0, 1.2e-6, 5.0e-12

    def f_of_s(s):
        return eps0 * A / (2.0 * (g - s) ** 2)

    v_pi, branch = pull_in_voltage_from_force_fn(
        jnp.asarray(k), f_of_s, max_amplitude_m=0.9 * g, n_samples=400
    )
    v_pi_an, _ = _analytic(k, g, A)
    assert abs(float(v_pi) - v_pi_an) / v_pi_an < 1e-2
    assert branch.shape == (400,)


def test_advect_density_shift_and_grad() -> None:
    """Linear Eulerian shift rigidly translates an interior block and stays
    differentiable.  (An interior block with air on both sides, not touching
    the grid edge, so the ``mode="nearest"`` clamp never fills it — that is
    the real EOM use: a moving conductor surrounded by gap.)"""
    field = jnp.zeros((60,), dtype=jnp.float32).at[20:30].set(1.0)

    def shifted_centroid(shift):
        disp = shift * jnp.ones((1, 60), dtype=jnp.float32)
        out = advect_density(field, disp, order=1)
        idx = jnp.arange(60, dtype=jnp.float32)
        return jnp.sum(out * idx) / jnp.sum(out)

    c0 = shifted_centroid(jnp.asarray(0.0))
    c5 = shifted_centroid(jnp.asarray(5.0))
    assert abs(float(c5 - c0) - 5.0) < 0.05  # true rigid translation
    g = jax.grad(shifted_centroid)(jnp.asarray(3.0))
    assert jnp.isfinite(g) and float(g) > 0.0
