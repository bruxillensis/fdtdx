"""Unit tests for the matrix-free elasticity eigensolver."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.optimization.mechanical.elasticity import (
    ElasticityEigenmodes,
    _gradient_components,
    _strain_energy,
    build_dof_mask,
    mass_operator_3d,
    stiffness_operator_3d,
)
from fdtdx.optimization.mechanical.modal import doubly_clamped_beam_modes

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _uniform_si_params(grid_shape: tuple[int, int, int]) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Si Lamé parameters and density on a uniform fully-dense grid."""
    E = 170e9
    nu = 0.27
    density = 2330.0
    mu = jnp.full(grid_shape, E / (2.0 * (1.0 + nu)), dtype=jnp.float32)
    lam = jnp.full(grid_shape, E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu)), dtype=jnp.float32)
    rho_phys = jnp.full(grid_shape, density, dtype=jnp.float32)
    return mu, lam, rho_phys


def _all_free_mask(grid_shape: tuple[int, int, int]) -> jax.Array:
    return jnp.ones((3, *grid_shape), dtype=bool)


# ---------------------------------------------------------------------------
# 1. Operator symmetry
# ---------------------------------------------------------------------------


def test_stiffness_operator_is_symmetric() -> None:
    """⟨v, K u⟩ == ⟨u, K v⟩ for any u, v on the free subspace."""
    grid_shape = (8, 6, 4)
    voxel = (50e-9, 50e-9, 50e-9)
    mu, lam, _ = _uniform_si_params(grid_shape)
    mask = _all_free_mask(grid_shape)
    # Clamp opposite faces so the operator restricts to a non-trivial subspace.
    mask = mask.at[:, 0, :, :].set(False)
    mask = mask.at[:, -1, :, :].set(False)

    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    u = jax.random.normal(k1, (3, *grid_shape), dtype=jnp.float32)
    v = jax.random.normal(k2, (3, *grid_shape), dtype=jnp.float32)

    Ku = stiffness_operator_3d(u, mu, lam, voxel, mask)
    Kv = stiffness_operator_3d(v, mu, lam, voxel, mask)
    lhs = float(jnp.sum(v * Ku))
    rhs = float(jnp.sum(u * Kv))
    assert abs(lhs - rhs) / max(abs(lhs) + abs(rhs), 1.0) < 1e-4


def test_stiffness_operator_is_psd_in_free_subspace() -> None:
    """⟨u, K u⟩ >= 0 — strain energy is non-negative."""
    grid_shape = (8, 4, 4)
    voxel = (50e-9, 50e-9, 50e-9)
    mu, lam, _ = _uniform_si_params(grid_shape)
    mask = _all_free_mask(grid_shape)
    mask = mask.at[:, 0, :, :].set(False)
    mask = mask.at[:, -1, :, :].set(False)

    key = jax.random.PRNGKey(7)
    for sk in jax.random.split(key, 5):
        u = jax.random.normal(sk, (3, *grid_shape), dtype=jnp.float32)
        u = u * mask
        Ku = stiffness_operator_3d(u, mu, lam, voxel, mask)
        quad = float(jnp.sum(u * Ku))
        assert quad >= -1e-6  # tiny float32 noise allowed


# ---------------------------------------------------------------------------
# 2. Rigid-body kernel (without BCs)
# ---------------------------------------------------------------------------


def test_rigid_body_translation_is_in_kernel() -> None:
    """With ALL DOFs free, a uniform rigid-body translation has zero strain."""
    grid_shape = (6, 6, 6)
    voxel = (50e-9, 50e-9, 50e-9)
    mu, lam, _ = _uniform_si_params(grid_shape)
    mask = _all_free_mask(grid_shape)

    # Pure x-translation.
    u = jnp.zeros((3, *grid_shape), dtype=jnp.float32)
    u = u.at[0].set(1.0)
    Ku = stiffness_operator_3d(u, mu, lam, voxel, mask)
    # The central-difference gradient of a constant is identically zero,
    # except at array boundaries where the wrap brings in the same value
    # (also constant), so strain remains zero everywhere.
    assert float(jnp.abs(Ku).max()) < 1e-3


# ---------------------------------------------------------------------------
# 3. Mass operator basics
# ---------------------------------------------------------------------------


def test_mass_operator_zeros_constrained_dofs() -> None:
    """Constrained components of Mu must be zero."""
    grid_shape = (4, 4, 4)
    voxel = (50e-9, 50e-9, 50e-9)
    _, _, rho_phys = _uniform_si_params(grid_shape)
    mask = _all_free_mask(grid_shape)
    mask = mask.at[2, :, :, 0].set(False)  # pin u_z at the bottom face

    key = jax.random.PRNGKey(3)
    u = jax.random.normal(key, (3, *grid_shape), dtype=jnp.float32)
    Mu = mass_operator_3d(u, rho_phys, voxel, mask)
    assert float(jnp.abs(Mu[2, :, :, 0]).max()) == 0.0


def test_mass_operator_matches_diagonal() -> None:
    """In the all-free case, ⟨u, Mu⟩ == ∫ ρ |u|² dV."""
    grid_shape = (4, 4, 4)
    voxel = (50e-9, 50e-9, 50e-9)
    _, _, rho_phys = _uniform_si_params(grid_shape)
    mask = _all_free_mask(grid_shape)

    key = jax.random.PRNGKey(11)
    u = jax.random.normal(key, (3, *grid_shape), dtype=jnp.float32)
    Mu = mass_operator_3d(u, rho_phys, voxel, mask)
    voxel_volume = voxel[0] * voxel[1] * voxel[2]
    expected = float(jnp.sum(rho_phys * jnp.sum(u * u, axis=0)) * voxel_volume)
    got = float(jnp.sum(u * Mu))
    rel_err = abs(expected - got) / max(abs(expected), 1.0)
    assert rel_err < 1e-5


# ---------------------------------------------------------------------------
# 4. build_dof_mask helper
# ---------------------------------------------------------------------------


def test_build_dof_mask_clamped_regions() -> None:
    grid = (10, 6, 4)
    mask = build_dof_mask(
        grid,
        clamped_regions=[
            (slice(0, 1), slice(None), slice(None)),
            (slice(-1, None), slice(None), slice(None)),
        ],
    )
    assert mask.shape == (3, 10, 6, 4)
    # Endpoints fully clamped.
    assert not bool(mask[:, 0, :, :].any())
    assert not bool(mask[:, -1, :, :].any())
    # Interior free.
    assert bool(mask[:, 1:-1, :, :].all())


def test_build_dof_mask_symmetry_plane() -> None:
    """Symmetry plane pins one component, leaves the other two free."""
    grid = (6, 6, 6)
    # y-symmetry at j=3: pin u_y there.
    mask = build_dof_mask(
        grid,
        symmetry_planes=[(1, slice(None), slice(3, 4), slice(None))],
    )
    assert not bool(mask[1, :, 3, :].any())  # u_y pinned on the plane
    assert bool(mask[0, :, 3, :].all())  # u_x free
    assert bool(mask[2, :, 3, :].all())  # u_z free


# ---------------------------------------------------------------------------
# 5. ElasticityEigenmodes — basic functionality
# ---------------------------------------------------------------------------


def _doubly_clamped_geometry() -> dict:
    """Return a small but representative doubly-clamped Si beam geometry."""
    return dict(
        length_m=10e-6,
        width_m=500e-9,
        thickness_m=220e-9,
        young_modulus=170e9,
        density=2330.0,
        grid_shape=(32, 4, 3),
    )


def _doubly_clamped_mask(grid_shape: tuple[int, int, int]) -> jax.Array:
    """Both x-end faces clamped (all components)."""
    return build_dof_mask(
        grid_shape,
        clamped_regions=[
            (slice(0, 1), slice(None), slice(None)),
            (slice(-1, None), slice(None), slice(None)),
        ],
    )


def test_eigenmodes_returns_right_shapes() -> None:
    geom = _doubly_clamped_geometry()
    nx, ny, nz = geom["grid_shape"]
    voxel = (geom["length_m"] / nx, geom["width_m"] / ny, geom["thickness_m"] / nz)

    em = ElasticityEigenmodes(
        young_modulus=geom["young_modulus"],
        poisson_ratio=0.27,
        density=geom["density"],
        voxel_size_m=voxel,
        free_dof_mask=_doubly_clamped_mask(geom["grid_shape"]),
        n_modes=2,
        n_subspace_iter=3,
        cg_iterations=40,
        cg_tol=1e-4,
    )
    rho = jnp.ones(geom["grid_shape"], dtype=jnp.float32)
    eigvals, modes = em.compute_modes(rho)
    assert eigvals.shape == (2,)
    assert modes.shape == (2, 3, nx, ny, nz)
    assert jnp.all(eigvals > 0.0)
    # Sorted ascending.
    assert float(eigvals[0]) <= float(eigvals[1]) + 1e-12


def test_eigenmodes_modes_vanish_at_clamped_faces() -> None:
    geom = _doubly_clamped_geometry()
    nx, ny, nz = geom["grid_shape"]
    voxel = (geom["length_m"] / nx, geom["width_m"] / ny, geom["thickness_m"] / nz)

    em = ElasticityEigenmodes(
        young_modulus=geom["young_modulus"],
        poisson_ratio=0.27,
        density=geom["density"],
        voxel_size_m=voxel,
        free_dof_mask=_doubly_clamped_mask(geom["grid_shape"]),
        n_modes=1,
        n_subspace_iter=4,
        cg_iterations=60,
        cg_tol=1e-5,
    )
    rho = jnp.ones(geom["grid_shape"], dtype=jnp.float32)
    _, modes = em.compute_modes(rho)
    end_face_left = modes[0, :, 0, :, :]
    end_face_right = modes[0, :, -1, :, :]
    assert float(jnp.abs(end_face_left).max()) < 1e-6
    assert float(jnp.abs(end_face_right).max()) < 1e-6


# ---------------------------------------------------------------------------
# 6. Sanity check vs Euler-Bernoulli
# ---------------------------------------------------------------------------


def test_first_eigenvalue_matches_analytical_order_of_magnitude() -> None:
    """The Q1/SRI eigensolver must reproduce the Euler-Bernoulli fundamental
    of a slender doubly-clamped Si beam to within a small physical factor.

    The selective-reduced-integration trilinear hexahedral element does NOT
    lock (unlike a node-collocated central-difference stencil, which
    over-stiffens this L/T≈45 flexure by ~250×).  On a modestly refined
    grid (48 × 4 × 4 — 4 cells through the 220 nm thickness) the 3D
    continuum frequency lands a little *below* the Euler-Bernoulli value
    because the continuum captures shear/rotary effects that thin-beam
    theory neglects.  A factor-of-three band is therefore a genuine
    physical-agreement assertion, not an order-of-magnitude escape hatch.
    """
    geom = _doubly_clamped_geometry()
    # Refine over the stored coarse grid: enough length cells and ≥4 through
    # the thickness so the bending mode is resolved (per the project rule of
    # increasing resolution rather than loosening the tolerance).
    grid_shape = (48, 4, 4)
    nx, ny, nz = grid_shape
    voxel = (geom["length_m"] / nx, geom["width_m"] / ny, geom["thickness_m"] / nz)

    analytical = doubly_clamped_beam_modes(
        length_m=geom["length_m"],
        width_m=geom["width_m"],
        thickness_m=geom["thickness_m"],
        young_modulus=geom["young_modulus"],
        density=geom["density"],
        grid_shape=grid_shape,
        beam_axis=0,
        deflection_axis=2,
        n_modes=1,
    )

    em = ElasticityEigenmodes(
        young_modulus=geom["young_modulus"],
        poisson_ratio=0.27,
        density=geom["density"],
        voxel_size_m=voxel,
        free_dof_mask=_doubly_clamped_mask(grid_shape),
        n_modes=1,
        n_subspace=4,
        n_subspace_iter=10,
        cg_iterations=200,
        cg_tol=1e-7,
        initial_subspace=analytical["mode_shapes"],
    )
    rho = jnp.ones(grid_shape, dtype=jnp.float32)
    eigvals, _ = em.compute_modes(rho)
    omega_fd = float(jnp.sqrt(jnp.maximum(eigvals[0], 1e-30)))
    f_fd = omega_fd / (2.0 * np.pi)
    f_anal = float(analytical["natural_frequencies_hz"][0])

    assert f_fd > 0.0
    ratio = f_fd / f_anal
    assert 0.33 < ratio < 3.0, f"f_fd={f_fd:.2e} Hz, f_anal={f_anal:.2e} Hz, ratio={ratio:.2f}"


# ---------------------------------------------------------------------------
# 7. Warm-start vs default convergence
# ---------------------------------------------------------------------------


def test_warm_start_lowers_residual_faster() -> None:
    """Seeding with analytical modes should give a closer first eigenvalue
    in fewer iterations than the default basis."""
    geom = _doubly_clamped_geometry()
    nx, ny, nz = geom["grid_shape"]
    voxel = (geom["length_m"] / nx, geom["width_m"] / ny, geom["thickness_m"] / nz)
    mask = _doubly_clamped_mask(geom["grid_shape"])

    common = dict(
        young_modulus=geom["young_modulus"],
        poisson_ratio=0.27,
        density=geom["density"],
        voxel_size_m=voxel,
        free_dof_mask=mask,
        n_modes=1,
        n_subspace=2,
        n_subspace_iter=2,  # very few iters to expose warm-start benefit
        cg_iterations=30,
        cg_tol=1e-4,
    )
    analytical = doubly_clamped_beam_modes(
        length_m=geom["length_m"],
        width_m=geom["width_m"],
        thickness_m=geom["thickness_m"],
        young_modulus=geom["young_modulus"],
        density=geom["density"],
        grid_shape=geom["grid_shape"],
        beam_axis=0,
        deflection_axis=2,
        n_modes=1,
    )

    em_cold = ElasticityEigenmodes(**common)
    em_warm = ElasticityEigenmodes(**common, initial_subspace=analytical["mode_shapes"])

    rho = jnp.ones(geom["grid_shape"], dtype=jnp.float32)
    # After more iterations both should agree closely; here with very few
    # iterations the warm-start should already be closer to converged.
    eig_cold, _ = em_cold.compute_modes(rho)
    eig_warm, _ = em_warm.compute_modes(rho)
    em_ref = ElasticityEigenmodes(**{**common, "n_subspace_iter": 12})
    eig_ref, _ = em_ref.compute_modes(rho)

    err_cold = float(jnp.abs(eig_cold[0] - eig_ref[0]))
    err_warm = float(jnp.abs(eig_warm[0] - eig_ref[0]))
    # Warm start should be no worse than cold; usually substantially better.
    # Allow slack since both might happen to land close on a benign geometry.
    assert err_warm <= err_cold + 1e-2 * abs(float(eig_ref[0]))


# ---------------------------------------------------------------------------
# 8. Gradient flow
# ---------------------------------------------------------------------------


def test_gradient_of_first_eigenvalue_w_r_t_rho_is_finite() -> None:
    """jax.grad of the first eigenvalue w.r.t. rho returns finite values."""
    geom = _doubly_clamped_geometry()
    nx, ny, nz = geom["grid_shape"]
    voxel = (geom["length_m"] / nx, geom["width_m"] / ny, geom["thickness_m"] / nz)
    mask = _doubly_clamped_mask(geom["grid_shape"])

    em = ElasticityEigenmodes(
        young_modulus=geom["young_modulus"],
        poisson_ratio=0.27,
        density=geom["density"],
        voxel_size_m=voxel,
        free_dof_mask=mask,
        n_modes=1,
        n_subspace=2,
        n_subspace_iter=3,
        cg_iterations=30,
        cg_tol=1e-4,
    )

    def first_eigenvalue(rho):
        eigvals, _ = em.compute_modes(rho)
        return eigvals[0]

    # ρ above the Si threshold (0.5) so the operator has a non-empty Si
    # domain; 0.9 leaves SIMP-scaled Lamé / density with a non-trivial
    # ρ-derivative inside the Si region.
    rho0 = 0.9 * jnp.ones(geom["grid_shape"], dtype=jnp.float32)
    g = jax.grad(first_eigenvalue)(rho0)
    assert jnp.all(jnp.isfinite(g))
    assert float(jnp.linalg.norm(g)) > 0.0


# ---------------------------------------------------------------------------
# 9. equilibrium_displacement integration
# ---------------------------------------------------------------------------


def test_equilibrium_displacement_runs_and_is_finite() -> None:
    geom = _doubly_clamped_geometry()
    nx, ny, nz = geom["grid_shape"]
    voxel = (geom["length_m"] / nx, geom["width_m"] / ny, geom["thickness_m"] / nz)
    mask = _doubly_clamped_mask(geom["grid_shape"])

    em = ElasticityEigenmodes(
        young_modulus=geom["young_modulus"],
        poisson_ratio=0.27,
        density=geom["density"],
        voxel_size_m=voxel,
        free_dof_mask=mask,
        n_modes=1,
        n_subspace=2,
        n_subspace_iter=3,
        cg_iterations=40,
        cg_tol=1e-4,
    )
    rho = jnp.ones(geom["grid_shape"], dtype=jnp.float32)
    force = jnp.zeros((3, *geom["grid_shape"]), dtype=jnp.float32)
    # Uniform z-direction body force.
    force = force.at[2].set(1e6)
    u = em.equilibrium_displacement(rho, force)
    assert u.shape == (3, *geom["grid_shape"])
    assert jnp.all(jnp.isfinite(u))
    # Constrained DOFs zero.
    assert float(jnp.abs(u[:, 0, :, :]).max()) < 1e-6
    assert float(jnp.abs(u[:, -1, :, :]).max()) < 1e-6


# ---------------------------------------------------------------------------
# 10. Internal helper sanity
# ---------------------------------------------------------------------------


def test_gradient_components_zero_for_constant_field() -> None:
    """∂_j u_i = 0 for spatially constant u_i."""
    u = jnp.ones((3, 4, 4, 4), dtype=jnp.float32)
    voxel = (50e-9, 50e-9, 50e-9)
    du = _gradient_components(u, voxel)
    # All derivatives of a constant field are zero in the interior; with
    # wrap-around they're zero everywhere (constant wraps to constant).
    assert float(jnp.abs(du).max()) < 1e-6


def test_strain_energy_zero_for_rigid_translation() -> None:
    """Pure translation has zero strain energy."""
    grid_shape = (4, 4, 4)
    voxel = (50e-9, 50e-9, 50e-9)
    mu, lam, _ = _uniform_si_params(grid_shape)
    u = jnp.zeros((3, *grid_shape), dtype=jnp.float32)
    u = u.at[0].set(1.0)
    U = float(_strain_energy(u, mu, lam, voxel))
    assert abs(U) < 1e-3


# ---------------------------------------------------------------------------
# 11. Sparse air-dominant geometry (Si-only operator)
# ---------------------------------------------------------------------------


def _sparse_beam_rho(
    grid_shape: tuple[int, int, int],
    band_y: tuple[int, int],
    band_z: tuple[int, int],
) -> jax.Array:
    """ρ field: a thin Si beam in the y/z bands, air everywhere else.

    The x-axis runs the beam length; both x-end faces will be clamped.
    """
    rho = np.zeros(grid_shape, dtype=np.float32)
    y0, y1 = band_y
    z0, z1 = band_z
    rho[:, y0:y1, z0:z1] = 1.0
    return jnp.asarray(rho)


def test_air_dofs_are_pinned() -> None:
    """Eigenmodes vanish on every cell whose ρ is below the threshold."""
    grid_shape = (32, 12, 6)
    voxel = (10e-6 / grid_shape[0], 1e-6 / grid_shape[1], 500e-9 / grid_shape[2])
    band_y, band_z = (5, 7), (2, 4)
    rho = _sparse_beam_rho(grid_shape, band_y, band_z)
    mask = _doubly_clamped_mask(grid_shape)

    em = ElasticityEigenmodes(
        young_modulus=170e9,
        poisson_ratio=0.27,
        density=2330.0,
        voxel_size_m=voxel,
        free_dof_mask=mask,
        n_modes=1,
        n_subspace=4,
        n_subspace_iter=4,
        cg_iterations=60,
        cg_tol=1e-5,
    )
    _, modes = em.compute_modes(rho)
    si_mask_3d = np.asarray(rho > 0.5)
    air_block = np.asarray(modes[:, :, ~si_mask_3d])
    assert float(np.max(np.abs(air_block))) == 0.0


def test_thin_beam_in_airspace_matches_euler_bernoulli() -> None:
    """Sparse-grid eigensolve on a thin Si beam in a large airspace lands
    inside the factor-of-3 Euler-Bernoulli band — the element-level Si
    mask gives true traction-free Si surfaces despite the surrounding
    air voxels.
    """
    # Beam: 10 µm long, 250 nm wide, 220 nm thick.  Sparse: 8× width and 4×
    # thickness of headroom around the Si so air dominates the volume.
    L, w, t = 10e-6, 250e-9, 220e-9
    nx, ny, nz = 48, 16, 6
    voxel = (L / nx, (8.0 * w) / ny, (4.0 * t) / nz)
    y0 = ny // 2 - 1
    z0 = nz // 2 - 1
    band_y, band_z = (y0, y0 + 2), (z0, z0 + 2)
    rho = _sparse_beam_rho((nx, ny, nz), band_y, band_z)
    mask = _doubly_clamped_mask((nx, ny, nz))

    analytical = doubly_clamped_beam_modes(
        length_m=L,
        width_m=w,
        thickness_m=t,
        young_modulus=170e9,
        density=2330.0,
        grid_shape=(nx, ny, nz),
        beam_axis=0,
        deflection_axis=2,
        n_modes=1,
    )
    f_anal = float(analytical["natural_frequencies_hz"][0])

    em = ElasticityEigenmodes(
        young_modulus=170e9,
        poisson_ratio=0.27,
        density=2330.0,
        voxel_size_m=voxel,
        free_dof_mask=mask,
        n_modes=1,
        n_subspace=4,
        n_subspace_iter=10,
        cg_iterations=200,
        cg_tol=1e-7,
    )
    eigvals, _ = em.compute_modes(rho)
    f_fd = float(jnp.sqrt(jnp.maximum(eigvals[0], 1e-30))) / (2.0 * np.pi)
    ratio = f_fd / f_anal
    assert 0.33 < ratio < 3.0, f"f_fd={f_fd:.2e} Hz, f_anal={f_anal:.2e} Hz, ratio={ratio:.2f}"


def test_gradient_finite_on_sparse_beam() -> None:
    """jax.grad of the first eigenvalue stays finite on a sparse Si geometry —
    the Heaviside Si mask is non-differentiable but SIMP-on-Si routes
    gradients to ρ via the Lamé / density factors.
    """
    grid_shape = (24, 10, 5)
    voxel = (10e-6 / grid_shape[0], 1e-6 / grid_shape[1], 500e-9 / grid_shape[2])
    band_y, band_z = (4, 6), (2, 4)
    rho0 = _sparse_beam_rho(grid_shape, band_y, band_z)
    # Soften to a graded design so SIMP scaling has a non-trivial derivative
    # interior — pure 0/1 ρ would have zero ε-perturbation Jacobian inside Si.
    rho0 = jnp.where(rho0 > 0.5, jnp.asarray(0.9), jnp.asarray(0.0))
    mask = _doubly_clamped_mask(grid_shape)

    em = ElasticityEigenmodes(
        young_modulus=170e9,
        poisson_ratio=0.27,
        density=2330.0,
        voxel_size_m=voxel,
        free_dof_mask=mask,
        n_modes=1,
        n_subspace=2,
        n_subspace_iter=3,
        cg_iterations=40,
        cg_tol=1e-4,
    )

    def first_eigenvalue(rho):
        eigvals, _ = em.compute_modes(rho)
        return eigvals[0]

    g = jax.grad(first_eigenvalue)(rho0)
    assert jnp.all(jnp.isfinite(g))
    si_cells = np.asarray(rho0 > 0.5)
    assert float(jnp.linalg.norm(g[si_cells])) > 0.0
