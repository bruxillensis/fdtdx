"""Unit tests for the virtual-temperature connectivity solver primitives.

These tests exercise the matrix-free heat-equation solver in
``fdtdx.optimization.constraints.connectivity`` directly, without requiring an
``ObjectContainer`` / ``ParameterContainer``.  The ``VirtualTemperatureConnectivity``
class itself is tested at the integration level once the other optimization
suite modules land.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from fdtdx.optimization.constraints.connectivity import _solve_heat_eq

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Homogeneous 1D-equivalent problem: linear temperature profile
# ---------------------------------------------------------------------------


def test_homogeneous_1d() -> None:
    """Uniform kappa on a (1, N, 1)-flavoured 1D strip should yield a monotonic
    profile that drops linearly from source to drain.
    """
    n = 16
    kappa = jnp.ones((n,), dtype=jnp.float32)

    source_mask = jnp.zeros((n,), dtype=jnp.float32)
    source_mask = source_mask.at[0].set(1.0)

    drain_mask = jnp.zeros((n,), dtype=bool)
    drain_mask = drain_mask.at[-1].set(True)

    T = _solve_heat_eq(
        kappa=kappa,
        source_mask=source_mask,
        drain_mask=drain_mask,
        cg_iterations=500,
        cg_tol=1e-8,
    )

    # Dirichlet enforced exactly at the drain.
    assert float(T[-1]) == pytest.approx(0.0, abs=1e-6)

    # Strict monotone decrease from source to drain.
    assert float(T[0]) > float(T[n // 2]) > float(T[-1])

    # Strictly positive temperature everywhere except the drain cell.
    assert bool(jnp.all(T[:-1] > 0.0))


# ---------------------------------------------------------------------------
# 2. Disconnected design → large penalty relative to a connected control
# ---------------------------------------------------------------------------


def test_disconnected_high_penalty() -> None:
    """A low-kappa gap between source and drain should raise the source-cell
    mean temperature by more than an order of magnitude compared to a fully
    connected conductor.
    """
    ny, nx = 16, 18
    kappa_max = 1.0
    kappa_min = 1e-3

    # Columns split into three thirds.
    third = nx // 3

    # Connected control: kappa_max everywhere.
    kappa_connected = jnp.full((ny, nx), kappa_max, dtype=jnp.float32)

    # Disconnected: kappa_min on the middle third (the "gap").
    kappa_disconnected = kappa_connected.at[:, third : 2 * third].set(kappa_min)

    source_mask = jnp.zeros((ny, nx), dtype=jnp.float32)
    source_mask = source_mask.at[:, 0].set(1.0)

    drain_mask = jnp.zeros((ny, nx), dtype=bool)
    drain_mask = drain_mask.at[:, -1].set(True)

    def _source_mean_T(kappa: jax.Array) -> jax.Array:
        T = _solve_heat_eq(
            kappa=kappa,
            source_mask=source_mask,
            drain_mask=drain_mask,
            cg_iterations=1000,
            cg_tol=1e-8,
        )
        return jnp.sum(T * source_mask) / jnp.maximum(source_mask.sum(), 1.0)

    penalty_connected = float(_source_mean_T(kappa_connected))
    penalty_disconnected = float(_source_mean_T(kappa_disconnected))

    # The disconnected design should have dramatically higher source-cell
    # temperature.  With a 10^3 conductivity ratio the expected ratio is of
    # order 10^3; we require at least 10x to stay comfortably robust.
    assert penalty_connected > 0.0
    assert penalty_disconnected > 10.0 * penalty_connected


# ---------------------------------------------------------------------------
# 3. Gradients propagate through the CG solve
# ---------------------------------------------------------------------------


def test_gradient_flows() -> None:
    """``jax.grad`` through the heat-equation solve should return finite,
    non-zero gradients w.r.t. the density field.
    """
    ny, nx = 8, 8
    source_mask = jnp.zeros((ny, nx), dtype=jnp.float32).at[:, 0].set(1.0)
    drain_mask = jnp.zeros((ny, nx), dtype=bool).at[:, -1].set(True)

    rho = jnp.full((ny, nx), 0.5, dtype=jnp.float32)

    def loss(rho: jax.Array) -> jax.Array:
        kappa = 1e-3 + (1.0 - 1e-3) * jnp.power(rho, 3.0)
        T = _solve_heat_eq(
            kappa=kappa,
            source_mask=source_mask,
            drain_mask=drain_mask,
            cg_iterations=400,
            cg_tol=1e-8,
        )
        return jnp.sum(T * source_mask)

    grads = jax.grad(loss)(rho)

    assert grads.shape == rho.shape
    assert bool(jnp.all(jnp.isfinite(grads)))
    assert bool(jnp.any(grads != 0.0))


# ---------------------------------------------------------------------------
# 4. JIT-compilation of the solver
# ---------------------------------------------------------------------------


def test_jit_compiles() -> None:
    """``jax.jit`` should compile and execute ``_solve_heat_eq`` end-to-end."""
    ny, nx = 8, 8
    kappa = jnp.full((ny, nx), 1.0, dtype=jnp.float32)
    source_mask = jnp.zeros((ny, nx), dtype=jnp.float32).at[:, 0].set(1.0)
    drain_mask = jnp.zeros((ny, nx), dtype=bool).at[:, -1].set(True)

    # ``cg_iterations`` and ``cg_tol`` are Python-level static args; close over
    # them rather than tracing them.
    @jax.jit
    def solve(kappa: jax.Array, source_mask: jax.Array, drain_mask: jax.Array) -> jax.Array:
        return _solve_heat_eq(
            kappa=kappa,
            source_mask=source_mask,
            drain_mask=drain_mask,
            cg_iterations=100,
            cg_tol=1e-6,
        )

    T = solve(kappa, source_mask, drain_mask)

    assert isinstance(T, jax.Array)
    assert T.shape == (ny, nx)
    assert bool(jnp.all(jnp.isfinite(T)))
