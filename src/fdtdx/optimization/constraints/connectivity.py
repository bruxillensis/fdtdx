"""Virtual-temperature connectivity constraint for inverse design.

Implements the *virtual-temperature* (a.k.a. thermal-diffusion) method from
topology-optimization literature as a differentiable connectivity penalty.

The design density :math:`\\rho \\in [0, 1]` on the device voxel grid is mapped
to a thermal conductivity :math:`\\kappa(\\rho)` via a SIMP-style power law::

    kappa(rho) = kappa_min + (kappa_max - kappa_min) * rho**p

A steady-state heat equation is then solved on the voxel grid,

.. math::

    -\\nabla \\cdot (\\kappa(\\rho) \\nabla T) = f,
    \\qquad T = 0 \\text{ on the drain mask,}

with a unit volumetric source :math:`f` on the ``source_mask``.  If the design
forms a continuous high-conductivity path between source and drain, heat flows
and the temperature at the source stays low; if the path is broken, the
temperature at the source blows up.  The mean source-cell temperature is thus
a smooth, differentiable proxy for *connectivity* and can be added to the
total loss as a penalty term.

The linear system is solved with :func:`jax.scipy.sparse.linalg.cg`, which
supplies implicit-function gradients w.r.t. ``rho`` out of the box.  A Jacobi
(diagonal) preconditioner keeps CG stable even at high material contrast.

Both 2D and 3D problems are supported; the operator dimension is auto-detected
from ``source_mask.ndim`` (5-point stencil in 2D, 7-point in 3D).

This class is a concrete subclass of
:class:`~fdtdx.optimization.constraints.physics.LinearSteadyStatePDEConstraint`
- the abstract plumbing (device lookup, CG driver, implicit-diff VJP) lives
there; this file supplies only the heat-specific operator, RHS, diagonal, and
penalty.
"""

from typing import Any

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.optimization.constraints.physics import (
    LinearSteadyStatePDEConstraint,
    _get_device,
    _squeeze_to_match,
)

__all__ = [
    "VirtualTemperatureConnectivity",
]


# ---------------------------------------------------------------------------
# Matrix-free operators (module-level for direct unit-testing)
# ---------------------------------------------------------------------------


def _boundary_mask(shape: tuple[int, ...], axis: int, side: str, dtype: Any) -> jax.Array:
    """Return an array of ones with zeros on one boundary slab along ``axis``.

    Used to zero out the flux across the (nonexistent) face that would lie
    outside the physical domain once ``jnp.roll`` has wrapped the array.
    """
    mask = jnp.ones(shape, dtype=dtype)
    if side == "left":
        idx = tuple(slice(0, 1) if i == axis else slice(None) for i in range(len(shape)))
    elif side == "right":
        idx = tuple(slice(-1, None) if i == axis else slice(None) for i in range(len(shape)))
    else:  # pragma: no cover - defensive
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return mask.at[idx].set(0.0)


def _apply_heat_operator_clean(
    T: jax.Array,
    kappa: jax.Array,
    drain_mask: jax.Array,
) -> jax.Array:
    """Symmetric matrix-free action ``y = A T`` for the heat operator.

    For interior cells this computes

    .. math::

        y_i = \\sum_{\\text{faces}} \\kappa_{\\text{face}} (T_i - T_{\\text{neighbour}})

    where ``kappa_face`` is the harmonic mean of the cell conductivities on
    either side of the face (the standard choice for variable-coefficient
    diffusion).  On drain cells the row is overwritten with the identity
    (``y_i = T_i``) so the Dirichlet condition ``T = 0`` is encoded as an
    identity block.  No-flux (Neumann) boundary conditions are applied on the
    outer domain boundary by zeroing the face conductivity there.

    The operator is symmetric positive definite, so CG is applicable.
    """
    y = jnp.zeros_like(T)
    for axis in range(T.ndim):
        T_left = jnp.roll(T, shift=1, axis=axis)
        T_right = jnp.roll(T, shift=-1, axis=axis)
        k_left = jnp.roll(kappa, shift=1, axis=axis)
        k_right = jnp.roll(kappa, shift=-1, axis=axis)

        k_face_left = 2.0 * kappa * k_left / jnp.maximum(kappa + k_left, 1e-30)
        k_face_right = 2.0 * kappa * k_right / jnp.maximum(kappa + k_right, 1e-30)

        mask_l = _boundary_mask(kappa.shape, axis=axis, side="left", dtype=kappa.dtype)
        mask_r = _boundary_mask(kappa.shape, axis=axis, side="right", dtype=kappa.dtype)
        k_face_left = k_face_left * mask_l
        k_face_right = k_face_right * mask_r

        y = y + k_face_left * (T - T_left) + k_face_right * (T - T_right)

    # Dirichlet rows: y = T on drain cells (identity block).
    y = jnp.where(drain_mask, T, y)
    return y


def _operator_diagonal(kappa: jax.Array, drain_mask: jax.Array) -> jax.Array:
    """Return the diagonal entries of the heat operator for Jacobi preconditioning."""
    diag = jnp.zeros_like(kappa)
    for axis in range(kappa.ndim):
        k_left = jnp.roll(kappa, shift=1, axis=axis)
        k_right = jnp.roll(kappa, shift=-1, axis=axis)

        k_face_left = 2.0 * kappa * k_left / jnp.maximum(kappa + k_left, 1e-30)
        k_face_right = 2.0 * kappa * k_right / jnp.maximum(kappa + k_right, 1e-30)

        mask_l = _boundary_mask(kappa.shape, axis=axis, side="left", dtype=kappa.dtype)
        mask_r = _boundary_mask(kappa.shape, axis=axis, side="right", dtype=kappa.dtype)
        k_face_left = k_face_left * mask_l
        k_face_right = k_face_right * mask_r

        diag = diag + k_face_left + k_face_right

    # Drain rows are identity (diagonal = 1).
    diag = jnp.where(drain_mask, jnp.asarray(1.0, dtype=diag.dtype), diag)
    # Guard against zero-diagonal (e.g. an isolated cell walled in by kappa=0).
    diag = jnp.maximum(diag, 1e-12)
    return diag


# ---------------------------------------------------------------------------
# Solver (kept module-level for direct unit-testing; LinearSteadyStatePDE-
# Constraint.build_penalty drives the same CG under the constraint path)
# ---------------------------------------------------------------------------


def _solve_heat_eq(
    kappa: jax.Array,
    source_mask: jax.Array,
    drain_mask: jax.Array,
    cg_iterations: int = 200,
    cg_tol: float = 1e-6,
) -> jax.Array:
    """Solve ``-div(kappa grad T) = source_mask`` with ``T = 0`` on ``drain_mask``.

    Uses :func:`jax.scipy.sparse.linalg.cg` with a Jacobi preconditioner.  The
    returned temperature field is differentiable with respect to ``kappa``
    through JAX's implicit-differentiation VJP for CG.
    """
    kappa = kappa.astype(jnp.float32)
    drain_mask = drain_mask.astype(bool)
    source_mask = source_mask.astype(kappa.dtype)

    diag = _operator_diagonal(kappa, drain_mask)

    def A(T: jax.Array) -> jax.Array:
        return _apply_heat_operator_clean(T, kappa, drain_mask)

    def M_inv(T: jax.Array) -> jax.Array:
        return T / diag

    # Dirichlet rows of the RHS must be zero to match the identity block.
    b = jnp.where(drain_mask, jnp.asarray(0.0, dtype=kappa.dtype), source_mask)

    T0 = jnp.zeros_like(b)
    T, _info = jax.scipy.sparse.linalg.cg(
        A,
        b,
        x0=T0,
        M=M_inv,
        tol=cg_tol,
        maxiter=cg_iterations,
    )
    # Enforce Dirichlet exactly in the returned field.
    T = jnp.where(drain_mask, jnp.asarray(0.0, dtype=T.dtype), T)
    return T


# ---------------------------------------------------------------------------
# Constraint class
# ---------------------------------------------------------------------------


@autoinit
class VirtualTemperatureConnectivity(LinearSteadyStatePDEConstraint):
    """Virtual-temperature connectivity constraint for inverse design.

    Solves a steady-state heat equation on the device voxel grid,

    .. math::

        -\\nabla \\cdot (\\kappa(\\rho) \\nabla T) = f,
        \\qquad T = 0 \\text{ on the drain mask,}

    where :math:`\\rho` is the device density (material indicator in
    :math:`[0, 1]` after the parameter-transform pipeline), :math:`f` is a
    unit heat source on ``source_mask`` cells, and :math:`\\kappa` uses the
    SIMP interpolation
    :math:`\\kappa(\\rho) = \\kappa_\\min + (\\kappa_\\max - \\kappa_\\min) \\rho^p`.

    The penalty is the mean temperature on source cells: high when heat cannot
    flow to the drain (disconnected design), low when :math:`\\rho` forms a
    continuous path.  Gradients flow through
    :func:`jax.scipy.sparse.linalg.cg` via implicit differentiation.
    """

    source_mask: jax.Array = frozen_field()
    drain_mask: jax.Array = frozen_field()
    kappa_min: float = frozen_field(default=1e-3)
    kappa_max: float = frozen_field(default=1.0)
    p: float = frozen_field(default=3.0)

    def _kappa(self, rho: jax.Array) -> jax.Array:
        return self.kappa_min + (self.kappa_max - self.kappa_min) * jnp.power(rho, self.p)

    # --- PhysicsConstraint hook -----------------------------------------

    def compute(
        self,
        *,
        params: Any,
        objects: Any,
        **_unused: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Return ``(penalty, info_dict)`` for the current ``params``.

        Overrides :meth:`PhysicsConstraint.compute` to squeeze the density
        to match ``source_mask.ndim`` (enabling 2D/3D auto-detection) before
        handing off to :meth:`LinearSteadyStatePDEConstraint.build_penalty`.
        """
        device = _get_device(objects, self.device_name)
        rho = device(params[self.device_name])
        rho = _squeeze_to_match(rho, self.source_mask.ndim)
        rho = jnp.clip(rho, 0.0, 1.0).astype(jnp.float32)
        return self.build_penalty(rho)

    # --- LinearSteadyStatePDEConstraint hooks ---------------------------

    def operator(self, u: jax.Array, rho: jax.Array) -> jax.Array:
        return _apply_heat_operator_clean(u, self._kappa(rho), self.drain_mask.astype(bool))

    def rhs(self, rho: jax.Array) -> jax.Array:
        del rho
        drain = self.drain_mask.astype(bool)
        return jnp.where(drain, jnp.asarray(0.0, dtype=jnp.float32), self.source_mask.astype(jnp.float32))

    def preconditioner_diag(self, rho: jax.Array) -> jax.Array:
        return _operator_diagonal(self._kappa(rho), self.drain_mask.astype(bool))

    def penalty(
        self,
        u: jax.Array,
        rho: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        del rho
        drain = self.drain_mask.astype(bool)
        u = jnp.where(drain, jnp.asarray(0.0, dtype=u.dtype), u)
        src = self.source_mask.astype(u.dtype)
        src_count = jnp.maximum(src.sum(), jnp.asarray(1.0, dtype=u.dtype))
        p = jnp.sum(u * src) / src_count
        info = {
            f"{self.name}_temp_max": u.max(),
            f"{self.name}_temp_mean_source": p,
        }
        return p, info
