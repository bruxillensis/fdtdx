"""Unit tests for :mod:`fdtdx.optimization.constraints.physics`.

Exercises :class:`LinearSteadyStatePDEConstraint` via a minimal concrete
subclass that solves the 1D Poisson equation ``-u''(x) = f`` with homogeneous
Dirichlet boundary conditions on a uniform grid.  The analytical solution is
known in closed form, so we can verify both CG convergence and implicit-
differentiation gradients without any FDTD machinery.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import pytest

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.optimization.constraints.physics import (
    LinearSteadyStatePDEConstraint,
    PhysicsConstraint,
    _get_device,
    _squeeze_to_match,
    _squeeze_trailing_singletons,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stubs that satisfy the `objects.devices` + `device(params[name])` contract
# ---------------------------------------------------------------------------


class _StubDevice:
    def __init__(self, name: str, voxel_pitch_m: float = 1.0):
        self.name = name
        self.single_voxel_real_shape = (voxel_pitch_m, voxel_pitch_m, voxel_pitch_m)

    def __call__(self, rho: jax.Array) -> jax.Array:
        return rho


class _StubObjects:
    def __init__(self, devices: list[_StubDevice]):
        self.devices = devices


# ---------------------------------------------------------------------------
# Concrete 1D Poisson constraint for testing the base class
# ---------------------------------------------------------------------------


@autoinit
class _PoissonPenalty(LinearSteadyStatePDEConstraint):
    """Solve a symmetric positive-definite 1D Poisson system.

    The operator is ``A u = 2u - u_prev - u_next`` on a periodic index
    space, plus a small Tikhonov regularizer ``alpha * u`` so ``A`` is
    strictly positive definite (the plain discrete Laplacian has the
    constant-mode null space).  Periodic stencils are automatically
    symmetric, so CG and its implicit-diff VJP are both stable.

    ``rhs`` returns ``rho`` directly.  ``penalty`` is the mean of ``u`` —
    a linear functional of the solution, giving non-trivial gradients.
    """

    alpha: float = frozen_field(default=0.1)

    def operator(self, u: jax.Array, rho: jax.Array) -> jax.Array:
        del rho
        left = jnp.roll(u, shift=1, axis=0)
        right = jnp.roll(u, shift=-1, axis=0)
        lap = 2.0 * u - left - right
        return lap + self.alpha * u

    def rhs(self, rho: jax.Array) -> jax.Array:
        return rho.astype(jnp.float32)

    def penalty(
        self,
        u: jax.Array,
        rho: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del rho
        val = jnp.mean(u)
        return val, {f"{self.name}_u_max": u.max()}

    def preconditioner_diag(self, rho: jax.Array) -> jax.Array:
        return (2.0 + self.alpha) * jnp.ones_like(rho, dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_squeeze_trailing_singletons() -> None:
    x = jnp.zeros((4, 5, 1, 1))
    assert _squeeze_trailing_singletons(x).shape == (4, 5)
    x = jnp.zeros((4, 5, 3))
    assert _squeeze_trailing_singletons(x).shape == (4, 5, 3)


def test_squeeze_to_match_drops_singletons() -> None:
    x = jnp.zeros((1, 8, 8, 1))
    assert _squeeze_to_match(x, target_ndim=2).shape == (8, 8)


def test_squeeze_to_match_raises_on_nonsingleton() -> None:
    x = jnp.zeros((3, 4, 5))
    with pytest.raises(ValueError, match="Cannot reduce"):
        _squeeze_to_match(x, target_ndim=2)


def test_get_device_found_and_missing() -> None:
    dev_a = _StubDevice("a")
    dev_b = _StubDevice("b")
    objects = _StubObjects([dev_a, dev_b])
    assert _get_device(objects, "a") is dev_a
    assert _get_device(objects, "b") is dev_b
    with pytest.raises(ValueError, match="No device named"):
        _get_device(objects, "c")


# ---------------------------------------------------------------------------
# PhysicsConstraint is abstract
# ---------------------------------------------------------------------------


def test_physics_constraint_is_abstract() -> None:
    with pytest.raises(TypeError):
        PhysicsConstraint(name="x", device_name="dev")  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# LinearSteadyStatePDEConstraint: CG convergence and implicit gradients
# ---------------------------------------------------------------------------


def _make_poisson(
    n: int = 33,
    alpha: float = 0.1,
    rho_value: float = 0.5,
) -> tuple[_PoissonPenalty, Any, jax.Array]:
    objects = _StubObjects([_StubDevice("dev")])
    # Keep rho strictly inside (0, 1) so the ``jnp.clip(rho, 0, 1)`` call
    # inside PhysicsConstraint.compute has identity gradient (clip at an
    # exact boundary has subgradient 0.5 in JAX, which would halve the
    # analytical closed-form gradient).
    rho = jnp.full((n,), rho_value, dtype=jnp.float32)
    constraint = _PoissonPenalty(
        name="poisson",
        device_name="dev",
        alpha=alpha,
        cg_iterations=500,
        cg_tol=1e-9,
    )
    return constraint, objects, rho


def test_poisson_penalty_matches_analytical_solution() -> None:
    """For a uniform forcing ``rho = c``, the periodic Laplacian + alpha*I
    system has the constant solution ``u = c / alpha`` (the Laplacian part
    vanishes on a constant)."""
    n = 33
    alpha = 0.1
    rho_value = 0.5
    constraint, objects, rho = _make_poisson(n=n, alpha=alpha, rho_value=rho_value)
    val, info = constraint.compute(params={"dev": rho}, objects=objects)
    expected = rho_value / alpha
    assert float(val) == pytest.approx(expected, rel=1e-4)
    assert f"{constraint.name}_u_max" in info


def test_poisson_gradient_flows() -> None:
    """jax.grad through the CG solve returns the analytical closed-form
    gradient.  For the periodic Laplacian + ``alpha * I`` operator, the
    gradient ``d(mean u) / d rho`` is exactly ``1/alpha * 1/n`` per cell."""
    n = 17
    alpha = 0.1
    constraint, objects, rho = _make_poisson(n=n, alpha=alpha, rho_value=0.5)

    def loss(rho: jax.Array) -> jax.Array:
        val, _ = constraint.compute(params={"dev": rho}, objects=objects)
        return val

    grads = jax.grad(loss)(rho)
    assert grads.shape == rho.shape
    assert bool(jnp.all(jnp.isfinite(grads)))
    expected = 1.0 / (alpha * n)
    assert bool(jnp.allclose(grads, expected, rtol=1e-3))


def test_no_preconditioner_still_converges() -> None:
    """`preconditioner_diag` returning None should fall back to identity M."""

    @autoinit
    class _PoissonNoPrecond(_PoissonPenalty):
        def preconditioner_diag(self, rho: jax.Array) -> None:
            del rho
            return None

    objects = _StubObjects([_StubDevice("dev")])
    rho = jnp.full((17,), 0.5, dtype=jnp.float32)
    constraint = _PoissonNoPrecond(
        name="poisson_np",
        device_name="dev",
        alpha=0.1,
        cg_iterations=2000,
        cg_tol=1e-8,
    )
    val, _ = constraint.compute(params={"dev": rho}, objects=objects)
    assert float(val) == pytest.approx(5.0, rel=1e-3)
