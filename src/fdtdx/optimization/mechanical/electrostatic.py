"""Differentiable quasi-static electrostatic solver on the design-voxel grid.

Solves the variable-coefficient Poisson equation

.. math::

    -\\nabla \\cdot (\\varepsilon(\\rho) \\nabla \\varphi) = 0,
    \\quad \\varphi = V \\text{ on electrode,} \\quad
    \\varphi = 0 \\text{ on ground,}

with Dirichlet boundaries enforced via the identity-row substitution used
elsewhere in :mod:`fdtdx.optimization.constraints.connectivity` — the same
harmonic-mean stencil applies, since the operator structure is identical
between the heat and electrostatic problems.  Solved with
:func:`jax.scipy.sparse.linalg.cg` so gradients with respect to ``rho`` flow
through the linear system via implicit differentiation for free.

The body-force on a dielectric / conductor interface is computed via the
Maxwell stress tensor; in the small-displacement regime the relevant term is

.. math::

    F(x) = \\tfrac{1}{2} \\varepsilon_0 \\, \\varepsilon(\\rho) \\, |\\nabla \\varphi|^2
           \\, \\nabla \\chi(\\rho),

where ``χ(ρ)`` is the indicator of the movable solid (gradient concentrated
at its boundary).  This gives a force per unit volume in N/m³, which a
mechanical model can integrate against its mode shapes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from fdtdx.constants import eps0
from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.optimization.constraints.connectivity import (
    _apply_heat_operator_clean,
    _operator_diagonal,
)

__all__ = [
    "PoissonSolver",
]


def _centered_diff(arr: jax.Array, axis: int) -> jax.Array:
    """Centered finite difference along ``axis``.

    Used in place of :func:`jnp.gradient` so the return type is unambiguously
    ``jax.Array`` (``jnp.gradient`` is typed as ``Array | list[Array]`` and
    confuses static type checkers).  At the array boundary this wraps via
    :func:`jnp.roll`, which is acceptable here because the Poisson solver
    only consumes the gradient in the device interior — boundary cells are
    pinned by Dirichlet conditions and the gradient there is unused.
    """
    return 0.5 * (jnp.roll(arr, -1, axis=axis) - jnp.roll(arr, 1, axis=axis))


@autoinit
class PoissonSolver(TreeClass):
    """Quasi-static electrostatic CG solver, exposed as a callable (not a Constraint).

    The masks are precomputed once for a given device geometry and frozen as
    pytree leaves.  ``solve(rho, voltage)`` returns the potential field;
    ``force_field(phi, rho)`` returns the Maxwell-stress body force per unit
    volume.  Both methods are JIT-friendly and differentiable.

    The default Si-DC permittivity (``11.7``) is a sensible interior value
    for an SOI MEMS beam; ``eps_min = 1.0`` represents the surrounding air
    gap.  Adjust for other material systems via ``eps_min`` / ``eps_max``.
    """

    electrode_mask: jax.Array = frozen_field()  # bool, design-grid shape
    ground_mask: jax.Array = frozen_field()     # bool, design-grid shape
    eps_min: float = frozen_field(default=1.0)
    eps_max: float = frozen_field(default=11.7)
    cg_iterations: int = frozen_field(default=400)
    cg_tol: float = frozen_field(default=1e-6)
    # Physical voxel spacing (m).  When ``None`` the spatial derivatives in
    # ``force_field`` use unit (index) spacing — fine for relative/qualitative
    # tests, but the resulting force is NOT in physical N/m³.  Set this to the
    # real voxel size so ∇φ is V/m and ∇χ is 1/m, giving a true N/m³ body
    # force that a mechanical model can integrate against mode shapes.
    voxel_size_m: tuple[float, float, float] | None = frozen_field(default=None)

    # ------------------------------------------------------------------
    # Forward solve
    # ------------------------------------------------------------------

    def _eps(self, rho: jax.Array) -> jax.Array:
        return self.eps_min + (self.eps_max - self.eps_min) * rho

    def solve(self, rho: jax.Array, voltage: jax.Array) -> jax.Array:
        """Return ``φ(x)`` on the design grid with the boundary conditions baked in."""
        rho = rho.astype(jnp.float32)
        eps = self._eps(rho)
        electrode = self.electrode_mask.astype(bool)
        ground = self.ground_mask.astype(bool)
        dirichlet = electrode | ground

        diag = _operator_diagonal(eps, dirichlet)

        def A(u: jax.Array) -> jax.Array:
            return _apply_heat_operator_clean(u, eps, dirichlet)

        def M(u: jax.Array) -> jax.Array:
            return u / diag

        # --- Dirichlet lifting --------------------------------------------
        # The identity-row operator is only symmetric-consistent for CG when
        # the Dirichlet values are ZERO (the coupling term -k·φ_D from a
        # bulk row into a non-zero Dirichlet neighbour has no symmetric
        # partner, so a naive RHS makes the system non-SPD and CG overshoots
        # — observed as φ_max ≫ V).  Solve instead for the correction
        # ``ψ = φ − φ_D`` with homogeneous Dirichlet, where φ_D carries the
        # known boundary values (V on electrode, 0 on ground).  Then
        # ``A ψ = −A φ_D`` on bulk rows, ``ψ = 0`` on Dirichlet rows, and the
        # restricted operator is SPD so CG converges cleanly.
        v = jnp.asarray(voltage, dtype=jnp.float32)
        phi_D = jnp.where(electrode, v, jnp.asarray(0.0, dtype=jnp.float32))
        lift = _apply_heat_operator_clean(phi_D, eps, dirichlet)
        b = jnp.where(dirichlet, jnp.asarray(0.0, dtype=jnp.float32), -lift)

        psi, _info = jax.scipy.sparse.linalg.cg(
            A,
            b,
            x0=jnp.zeros_like(b),
            M=M,
            tol=self.cg_tol,
            maxiter=self.cg_iterations,
        )
        psi = jnp.where(dirichlet, jnp.asarray(0.0, dtype=psi.dtype), psi)
        phi = psi + phi_D
        # Hard-enforce Dirichlet exactly (numerical insurance).
        phi = jnp.where(electrode, v, phi)
        phi = jnp.where(ground, jnp.asarray(0.0, dtype=phi.dtype), phi)
        return phi

    # ------------------------------------------------------------------
    # Force computation
    # ------------------------------------------------------------------

    def force_field(self, phi: jax.Array, rho: jax.Array) -> jax.Array:
        """Body-force field on the movable solid (N/m³), shape ``(3, *grid)``.

        Computes ``F = ½ ε₀ ε(ρ) |∇φ|² ∇χ_solid``, where ``χ_solid = ρ`` (the
        soft indicator of the movable region; its gradient concentrates the
        force at the air-Si interface, which is where the Maxwell stress
        physically acts).  Returned in volume-density units; a mechanical
        solver should integrate ``∫ F · Φ dV`` to obtain modal forces.
        """
        rho = rho.astype(jnp.float32)
        eps = self._eps(rho)
        ndim = phi.ndim
        # Physical spacing per axis so ∇φ is V/m and ∇χ is 1/m.  Falls back
        # to unit spacing when ``voxel_size_m`` is unset (qualitative use).
        if self.voxel_size_m is None:
            spacings = tuple(1.0 for _ in range(ndim))
        else:
            spacings = tuple(float(self.voxel_size_m[a]) for a in range(ndim))
        grads_phi = jnp.stack(
            [_centered_diff(phi, a) / spacings[a] for a in range(ndim)], axis=0
        )
        e_sq = jnp.sum(grads_phi * grads_phi, axis=0)
        grads_rho = jnp.stack(
            [_centered_diff(rho, a) / spacings[a] for a in range(ndim)], axis=0
        )

        # |E|² has units V²; multiplying by ε₀·ε gives J/m³ = N/m²·(1/m); the
        # spatial gradient of χ adds the missing 1/m so the total is N/m³.
        force = 0.5 * eps0 * eps[None] * e_sq[None] * grads_rho
        # Pad component axis to 3 if the geometry is 2D so downstream code can
        # always assume shape (3, *grid).
        if ndim == 2:
            zero_z = jnp.zeros_like(force[:1])
            force = jnp.concatenate([force, zero_z], axis=0)
        return force
