"""Unit tests for the differentiable Poisson solver.

The solver is tested against the analytical parallel-plate capacitor: two
electrode masks separated by a uniform dielectric should give a linear
potential profile, constant |E| field, and zero force density inside a
homogeneous region.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.optimization.mechanical.electrostatic import PoissonSolver

pytestmark = pytest.mark.unit


def _make_parallel_plate(nz: int, ny: int = 4, nx: int = 4) -> tuple[jnp.ndarray, jnp.ndarray]:
    electrode = jnp.zeros((nx, ny, nz), dtype=bool).at[:, :, 0].set(True)
    ground = jnp.zeros((nx, ny, nz), dtype=bool).at[:, :, -1].set(True)
    return electrode, ground


def test_parallel_plate_linear_potential() -> None:
    """Uniform ε between two plates → φ varies linearly between V and 0."""
    nz = 16
    electrode, ground = _make_parallel_plate(nz)
    solver = PoissonSolver(
        electrode_mask=electrode,
        ground_mask=ground,
        eps_min=1.0,
        eps_max=1.0,  # homogeneous medium
        cg_iterations=400,
        cg_tol=1e-7,
    )
    rho = jnp.zeros_like(electrode, dtype=jnp.float32)
    phi = solver.solve(rho, jnp.asarray(1.0))
    profile = np.asarray(phi[2, 2, :])

    expected = np.linspace(1.0, 0.0, nz, dtype=np.float32)
    # Allow modest tolerance: CG residual + harmonic-mean stencil discretisation.
    assert np.allclose(profile, expected, atol=2e-2)


def test_zero_voltage_gives_zero_field() -> None:
    """V = 0 → φ ≡ 0 anywhere."""
    nz = 8
    electrode, ground = _make_parallel_plate(nz)
    solver = PoissonSolver(electrode_mask=electrode, ground_mask=ground)
    rho = jnp.ones_like(electrode, dtype=jnp.float32)  # solid in between
    phi = solver.solve(rho, jnp.asarray(0.0))
    assert jnp.all(jnp.abs(phi) < 1e-5)


def test_force_field_zero_in_homogeneous_region() -> None:
    """Maxwell stress only acts at dielectric interfaces, so force ≈ 0 in
    the middle of a uniformly filled cavity."""
    nz = 16
    electrode, ground = _make_parallel_plate(nz)
    solver = PoissonSolver(electrode_mask=electrode, ground_mask=ground)
    rho = jnp.ones_like(electrode, dtype=jnp.float32)
    phi = solver.solve(rho, jnp.asarray(1.0))
    force = solver.force_field(phi, rho)
    # In the homogeneous bulk (away from the masks), |∇ρ| = 0 so force = 0.
    interior_force_norm = float(jnp.linalg.norm(force[:, :, :, 4:-4]))
    assert interior_force_norm < 1e-5


def test_force_field_nonzero_at_interface() -> None:
    """A step in ρ → non-zero force from Maxwell stress at the boundary.

    We only check that (i) the force is finite, (ii) the bulk force
    integrates to a non-trivial value, and (iii) cells far from any density
    gradient see zero z-force.  The precise peak location depends on the
    one-sided stencil ``jnp.gradient`` uses at boundaries and on the CG
    convergence threshold, so we don't pin it.
    """
    nz = 16
    electrode, ground = _make_parallel_plate(nz)
    solver = PoissonSolver(electrode_mask=electrode, ground_mask=ground)
    # Half-filled cavity: Si on lower half, air on upper.
    rho = jnp.zeros_like(electrode, dtype=jnp.float32)
    rho = rho.at[:, :, : nz // 2].set(1.0)
    phi = solver.solve(rho, jnp.asarray(1.0))
    force = solver.force_field(phi, rho)

    assert jnp.all(jnp.isfinite(force))
    # Force in z must be nonzero somewhere.  Use abs().max() rather than the
    # L2 norm so the very small physical magnitude (~ε₀·V²·∇ρ at unit voxel
    # spacing ≈ 1e-22) doesn't underflow when squared in float32.
    assert float(jnp.abs(force[2]).max()) > 0.0

    # Cells in the homogeneous interior of the Si region (away from
    # interface and electrode) have ∇ρ = 0, so z-force must be (nearly) zero
    # there — well below the interface peak.
    interior_si_max = float(jnp.abs(force[2, :, :, 2:5]).max())
    interface_peak = float(jnp.abs(force[2]).max())
    assert interior_si_max < 0.1 * interface_peak


def test_gradients_flow_through_solver() -> None:
    """A loss on the solved potential must produce finite gradients w.r.t. ρ."""
    nz = 12
    electrode, ground = _make_parallel_plate(nz)
    solver = PoissonSolver(electrode_mask=electrode, ground_mask=ground)
    rho = 0.5 * jnp.ones_like(electrode, dtype=jnp.float32)

    def loss(rho):
        phi = solver.solve(rho, jnp.asarray(2.0))
        return jnp.mean(phi**2)

    grad = jax.grad(loss)(rho)
    assert jnp.all(jnp.isfinite(grad))
    assert float(jnp.linalg.norm(grad)) > 0.0
