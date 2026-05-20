"""Differentiable elasticity eigensolver on the design voxel grid.

Computes the lowest eigenmodes of the linear-isotropic elasticity
generalized eigenproblem

.. math::

    K(\\rho) u = \\omega^2 M(\\rho) u

directly from the current density ρ.  The operator is restricted to the
**Si subdomain** of ρ — only Q1 elements whose 8 corner cells satisfy
``ρ > rho_mask_threshold`` contribute strain energy and mass.  Boundary
"mixed" elements straddling Si/air drop out entirely, so the Si surfaces
are traction-free (the same mesh-only-the-solid behaviour COMSOL gets by
not meshing air).  Air-side nodes are pinned in the DOF mask so they
don't appear as zero-eigenvalue rigid-body modes.  Mode shapes and
natural frequencies flow gradient back to ρ end-to-end through
:func:`jax.scipy.sparse.linalg.cg` (inverse-iteration step) and
:func:`jax.numpy.linalg.eigh` (Rayleigh-Ritz reduction); the Boolean
ρ-threshold comparison is intentionally non-differentiable
(``stop_gradient``) and ρ-gradients reach the modes through the SIMP-
scaled Lamé / density on Si cells.

**Discretisation.**  Linear isotropic elasticity on a structured Cartesian
voxel grid with cell-centred displacement ``u`` of shape ``(3, Nx, Ny, Nz)``.
Strain ``ε = sym(∇u)`` and stress ``σ = λ tr(ε) I + 2μ ε``; the stiffness
operator is the gradient of the strain energy
``U = ½∫(λ (tr ε)² + 2μ ε:ε) dV`` w.r.t. ``u``, guaranteeing that ``K`` is
symmetric in the discrete sense.  Selective reduced integration on a
trilinear Q1 hex element (deviatoric energy at the 2×2×2 Gauss rule,
volumetric energy at the element centre) avoids shear / volumetric
locking — a node-collocated central-difference stencil over-stiffens
slender flexures by ~250×.

**SIMP interpolation.**  Lamé parameters scale as
``E_eff(ρ) = E (ε_min + (1−ε_min) ρ^p)``; mass linearly in ρ.  Combined
with the element-level Si mask, the SIMP factor only matters on partially-
filled cells inside the Si mask (boundary smoothing during the projection
ramp) — air cells contribute zero through the mask regardless.

**Eigensolver.**  Subspace iteration with M-orthonormalisation and
Rayleigh-Ritz projection.  Each iteration:

1. Inverse iteration ``B' = K^{-1} M B`` via CG, column by column.
2. Modified Gram-Schmidt M-orthonormalisation so ``B^T M B = I``.
3. Rayleigh-Ritz: form ``K_red = B^T K B`` (k × k, symmetric), solve a
   small dense eigenproblem with :func:`jax.numpy.linalg.eigh`, rotate the
   basis so its rows are eigenvectors in ascending order.

For modest problems (k ≤ 6, n_subspace_iter ~ 8) this converges to the
lowest modes in a few iterations.

**Requirements.**  Si features must be ≥ 2 cells per axis (else no fully-Si
elements exist along that axis and the operator is empty).  For 1-cell-
thick optical layers, replicate to ~4 cells in the mech grid.
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.optimization.mechanical.base import MechanicalModel

__all__ = [
    "ElasticityEigenmodes",
    "build_dof_mask",
    "mass_operator_3d",
    "stiffness_operator_3d",
]


# ---------------------------------------------------------------------------
# Boundary condition helper
# ---------------------------------------------------------------------------


def build_dof_mask(
    grid_shape: tuple[int, int, int],
    *,
    clamped_regions: Sequence[tuple[slice, slice, slice]] = (),
    symmetry_planes: Sequence[tuple[int, slice, slice, slice]] = (),
) -> jax.Array:
    """Build a free-DOF mask of shape ``(3, Nx, Ny, Nz)`` for the elasticity solver.

    Parameters
    ----------
    grid_shape:
        Spatial shape of the design voxel grid.
    clamped_regions:
        Iterable of ``(slice_x, slice_y, slice_z)`` triples.  Within each
        region **all three** displacement components are pinned to zero
        (Dirichlet).  Use for fully-clamped anchors, bond pads, etc.
    symmetry_planes:
        Iterable of ``(normal_axis, slice_x, slice_y, slice_z)`` where the
        component **along** ``normal_axis`` is pinned to zero (the normal
        component vanishes across the symmetry plane); the other two
        components stay free.  Use for in-plane mirror symmetries.

    Returns
    -------
    jax.Array
        Boolean array, ``True`` where the DOF is free, ``False`` where it
        is constrained.
    """
    mask = jnp.ones((3, *grid_shape), dtype=bool)
    for region in clamped_regions:
        sx, sy, sz = region
        mask = mask.at[:, sx, sy, sz].set(False)
    for plane in symmetry_planes:
        axis, sx, sy, sz = plane
        if axis not in (0, 1, 2):
            raise ValueError(f"symmetry-plane normal axis must be in (0, 1, 2), got {axis}")
        mask = mask.at[axis, sx, sy, sz].set(False)
    return mask


# ---------------------------------------------------------------------------
# SIMP interpolation
# ---------------------------------------------------------------------------


def _simp_stiffness_scale(rho: jax.Array, eps_min: float, p: float) -> jax.Array:
    """SIMP penalty: ``s(ρ) = ε_min + (1−ε_min) ρ^p``."""
    rho_clip = jnp.clip(rho, 0.0, 1.0)
    return eps_min + (1.0 - eps_min) * jnp.power(rho_clip, p)


def _linear_mass_scale(rho: jax.Array, eps_min: float) -> jax.Array:
    """Linear interpolation for mass: ``ε_min + (1 − ε_min) ρ``."""
    rho_clip = jnp.clip(rho, 0.0, 1.0)
    return eps_min + (1.0 - eps_min) * rho_clip


# ---------------------------------------------------------------------------
# Strain energy / stiffness operator
# ---------------------------------------------------------------------------


# Trilinear (Q1) 8-node hexahedral element with **selective reduced
# integration** (deviatoric energy at the full 2×2×2 Gauss rule,
# volumetric energy at the single element-centre point).  This is the
# standard structured-grid topology-optimization element and does not
# shear/volumetrically lock for slender bending members — a node-
# collocated central-difference stencil would over-stiffen a doubly-
# clamped Si flexure (aspect ratio ~45) by ~250×.  Corner ``a`` carries
# natural-coordinate signs ``(sx, sy, sz) ∈ {-1, +1}³``.
_CORNER_SIGNS = np.array(
    [[2 * ((a >> 0) & 1) - 1, 2 * ((a >> 1) & 1) - 1, 2 * ((a >> 2) & 1) - 1] for a in range(8)],
    dtype=np.float64,
)  # (8, 3)

_GP = 1.0 / np.sqrt(3.0)
_GAUSS_PTS = np.array(
    [[sx * _GP, sy * _GP, sz * _GP] for sx in (-1.0, 1.0) for sy in (-1.0, 1.0) for sz in (-1.0, 1.0)],
    dtype=np.float64,
)  # (8, 3)
_CENTER_PT = np.zeros((1, 3), dtype=np.float64)


def _shape_grads_phys(natural_pts: np.ndarray, voxel_size_m: tuple[float, float, float]) -> jax.Array:
    """∂N_a/∂x_j evaluated at each natural-coordinate point.

    Returns shape ``(P, 8, 3)`` for ``P`` quadrature points, 8 corners,
    3 spatial derivative directions.  The element is axis-aligned with
    edge lengths ``voxel_size_m`` so the Jacobian is diagonal
    (``∂χ_j/∂x_j = 2 / h_j``) and the same for every element.
    """
    inv_jac = np.array([2.0 / h for h in voxel_size_m], dtype=np.float64)  # (3,)
    out = np.zeros((natural_pts.shape[0], 8, 3), dtype=np.float64)
    for p, (xi, eta, ze) in enumerate(natural_pts):
        nat = np.array([xi, eta, ze], dtype=np.float64)
        for a in range(8):
            s = _CORNER_SIGNS[a]
            # dN_a/dχ_j = (1/8) s_j ∏_{m≠j} (1 + s_m χ_m)
            for j in range(3):
                others = 1.0
                for m in range(3):
                    if m != j:
                        others *= 1.0 + s[m] * nat[m]
                dN_dchi = 0.125 * s[j] * others
                out[p, a, j] = dN_dchi * inv_jac[j]
    return jnp.asarray(out, dtype=jnp.float32)


def _element_corner_gather(field: jax.Array) -> list[jax.Array]:
    """Gather the 8 corner sub-arrays of every element.

    ``field`` is node/voxel data ``(..., Nx, Ny, Nz)``.  Element ``e`` spans
    nodes ``(i+bx, j+by, k+bz)``; the returned list is ordered to match
    ``_CORNER_SIGNS`` and each entry has spatial shape
    ``(Nx-1, Ny-1, Nz-1)``.
    """
    nx, ny, nz = field.shape[-3:]
    corners: list[jax.Array] = []
    for a in range(8):
        bx, by, bz = (a >> 0) & 1, (a >> 1) & 1, (a >> 2) & 1
        corners.append(field[..., bx : nx - 1 + bx, by : ny - 1 + by, bz : nz - 1 + bz])
    return corners


def _gradient_components(u: jax.Array, voxel_size_m: tuple[float, float, float]) -> jax.Array:
    """Central-difference ``∂_j u_i``; returns ``(3, 3, Nx, Ny, Nz)``.

    Plain gradient stencil for tests / diagnostics.  Index convention
    ``du[j, i] = ∂_j u_i``.
    """
    spacings = voxel_size_m
    du_rows = []
    for j in range(3):
        spacing = spacings[j]
        row = jnp.stack(
            [(jnp.roll(u[i], -1, axis=j) - jnp.roll(u[i], 1, axis=j)) / (2.0 * spacing) for i in range(3)],
            axis=0,
        )
        du_rows.append(row)
    return jnp.stack(du_rows, axis=0)


def _strain_energy(
    u: jax.Array,
    lame_mu: jax.Array,
    lame_lambda: jax.Array,
    voxel_size_m: tuple[float, float, float],
    element_mask: jax.Array | None = None,
) -> jax.Array:
    """Total strain energy of the Q1 mesh with selective reduced integration.

    ``U = Σ_elem V_el [ ½ κ (tr ε)²|_centre + ⟨ μ ε_dev:ε_dev ⟩_8gp ]`` with
    bulk modulus ``κ = λ + ⅔μ``.  The volumetric term is under-integrated at
    the element centre (cures volumetric locking); the deviatoric term uses
    the full 2×2×2 Gauss rule.  ``K u = ∂U/∂u`` via :func:`jax.grad` stays
    exactly symmetric because ``U`` is a quadratic form in ``u``.

    ``element_mask``, when supplied, is an element-level multiplicative
    indicator ``(Nx-1, Ny-1, Nz-1)`` — masked-out elements contribute
    zero stiffness, used by :class:`ElasticityEigenmodes` to restrict
    the operator to the Si subdomain so the Si surface is traction-free.
    """
    nx, ny, nz = u.shape[-3:]
    if nx < 2 or ny < 2 or nz < 2:
        # Degenerate (single layer in some axis) — fall back to a per-voxel
        # central-difference form so 2D-ish test grids still produce a
        # finite, symmetric operator.  Slender 3D flexures must use ≥2 cells
        # per axis to benefit from the non-locking element.
        return _strain_energy_collocated(u, lame_mu, lame_lambda, voxel_size_m)

    v_el = float(voxel_size_m[0] * voxel_size_m[1] * voxel_size_m[2])

    # Element-averaged material (mean of the 8 surrounding voxels).
    mu_el = jnp.mean(jnp.stack(_element_corner_gather(lame_mu), axis=0), axis=0)
    lam_el = jnp.mean(jnp.stack(_element_corner_gather(lame_lambda), axis=0), axis=0)
    if element_mask is not None:
        em = element_mask.astype(mu_el.dtype)
        mu_el = mu_el * em
        lam_el = lam_el * em
    kappa_el = lam_el + (2.0 / 3.0) * mu_el  # (Ex, Ey, Ez)

    # Corner displacement, vectorised: U_c[i, a] over element grid.
    u_c = jnp.stack(
        [jnp.stack(_element_corner_gather(u[i]), axis=0) for i in range(3)], axis=0
    )  # (3 comp, 8 corner, Ex, Ey, Ez)

    def _grad_tensor(dNdx: jax.Array) -> jax.Array:
        # dNdx: (8 corner, 3 deriv).  du[j, i] = Σ_a dNdx[a, j] u_i^a.
        # → (3 deriv j, 3 comp i, Ex, Ey, Ez)
        return jnp.asarray(jnp.einsum("aj,iaxyz->jixyz", dNdx, u_c))

    # --- volumetric part: 1-point (centre) ---------------------------------
    dN_c = _shape_grads_phys(_CENTER_PT, voxel_size_m)[0]  # (8, 3)
    du_c = _grad_tensor(dN_c)
    tr_c = du_c[0, 0] + du_c[1, 1] + du_c[2, 2]
    u_vol = jnp.sum(0.5 * kappa_el * tr_c**2) * v_el

    # --- deviatoric part: full 2×2×2 Gauss (all 8 points at once) ----------
    dN_g = _shape_grads_phys(_GAUSS_PTS, voxel_size_m)  # (8gp, 8, 3)
    du_g = jnp.einsum("gaj,iaxyz->gjixyz", dN_g, u_c)  # (8gp,3,3,E…)
    eps_g = 0.5 * (du_g + jnp.swapaxes(du_g, 1, 2))
    tr_g = eps_g[:, 0, 0] + eps_g[:, 1, 1] + eps_g[:, 2, 2]  # (8gp, E…)
    eye = jnp.eye(3, dtype=eps_g.dtype)[None, :, :, None, None, None]
    eps_dev = eps_g - (tr_g[:, None, None] / 3.0) * eye
    dd = jnp.sum(eps_dev * eps_dev, axis=(1, 2))  # (8gp, E…)
    # Mean over the 8 Gauss points (each weight 1, |J| = V_el/8 ⇒ Σ = V_el·mean).
    u_dev = jnp.sum(mu_el * jnp.mean(dd, axis=0)) * v_el
    return u_vol + u_dev


def _strain_energy_collocated(
    u: jax.Array,
    lame_mu: jax.Array,
    lame_lambda: jax.Array,
    voxel_size_m: tuple[float, float, float],
) -> jax.Array:
    """Node-collocated central-difference strain energy (fallback only).

    Used when a grid axis has < 2 cells, where the Q1 element is undefined.
    Known to lock for slender bending — acceptable only for the thick / 2D
    degenerate test cases that exercise this branch.
    """
    dx, dy, dz = voxel_size_m
    spacings = (dx, dy, dz)
    du_rows = []
    for j in range(3):
        spacing = spacings[j]
        row = jnp.stack(
            [(jnp.roll(u[i], -1, axis=j) - jnp.roll(u[i], 1, axis=j)) / (2.0 * spacing) for i in range(3)],
            axis=0,
        )
        du_rows.append(row)
    du = jnp.stack(du_rows, axis=0)
    eps = 0.5 * (du + du.transpose((1, 0, 2, 3, 4)))
    trace_eps = eps[0, 0] + eps[1, 1] + eps[2, 2]
    eps_dot_eps = jnp.sum(eps * eps, axis=(0, 1))
    energy_density = 0.5 * lame_lambda * trace_eps**2 + lame_mu * eps_dot_eps
    voxel_volume = voxel_size_m[0] * voxel_size_m[1] * voxel_size_m[2]
    return jnp.sum(energy_density) * voxel_volume


def stiffness_operator_3d(
    u: jax.Array,
    lame_mu: jax.Array,
    lame_lambda: jax.Array,
    voxel_size_m: tuple[float, float, float],
    free_mask: jax.Array,
    element_mask: jax.Array | None = None,
) -> jax.Array:
    """Matrix-free linear isotropic elasticity stiffness ``K(ρ) u``.

    Parameters
    ----------
    u:
        Displacement field ``(3, Nx, Ny, Nz)``.
    lame_mu, lame_lambda:
        Per-cell Lamé parameters ``(Nx, Ny, Nz)``.
    voxel_size_m:
        Physical voxel spacing along each axis.
    free_mask:
        Bool ``(3, Nx, Ny, Nz)`` — ``True`` where the DOF is free.
    element_mask:
        Optional ``(Nx-1, Ny-1, Nz-1)`` Si element indicator forwarded to
        :func:`_strain_energy`.  ``None`` means every element contributes
        its full SIMP-scaled stiffness.

    Returns
    -------
    jax.Array
        ``K u`` of the same shape as ``u``, with constrained DOFs zeroed.
    """
    u_constrained = jnp.where(free_mask, u, jnp.asarray(0.0, dtype=u.dtype))
    Ku = jax.grad(_strain_energy)(u_constrained, lame_mu, lame_lambda, voxel_size_m, element_mask)
    return jnp.where(free_mask, Ku, jnp.asarray(0.0, dtype=Ku.dtype))


def mass_operator_3d(
    u: jax.Array,
    density_field: jax.Array,
    voxel_size_m: tuple[float, float, float],
    free_mask: jax.Array,
) -> jax.Array:
    """Lumped (diagonal) mass operator ``M(ρ) u = ρ_phys · u · V_voxel``.

    The diagonal form means the M-orthogonalisation in the eigensolver
    reduces to per-cell weighting; no inner linear solve is needed.
    """
    u_constrained = jnp.where(free_mask, u, jnp.asarray(0.0, dtype=u.dtype))
    voxel_volume = voxel_size_m[0] * voxel_size_m[1] * voxel_size_m[2]
    Mu = density_field[None] * u_constrained * voxel_volume
    return jnp.where(free_mask, Mu, jnp.asarray(0.0, dtype=Mu.dtype))


# ---------------------------------------------------------------------------
# Subspace iteration
# ---------------------------------------------------------------------------


def _m_orthonormalize(
    B: jax.Array,
    M_op,
) -> jax.Array:
    """Modified Gram-Schmidt with respect to the M-inner product.

    Parameters
    ----------
    B:
        Shape ``(k, 3, Nx, Ny, Nz)`` — k candidate vectors.
    M_op:
        Callable ``v → M v``.

    Returns
    -------
    jax.Array
        Same shape as ``B``, with ``B[i]^T M B[j] = δ_ij`` (within float32
        roundoff).
    """
    k = B.shape[0]
    Q: list[jax.Array] = []
    for i in range(k):
        q = B[i]
        # Subtract M-projections onto previously orthonormalised vectors.
        for j in range(i):
            Mq = M_op(q)
            coeff = jnp.sum(Q[j] * Mq)
            q = q - coeff * Q[j]
        # Normalise to unit M-norm.
        norm = jnp.sqrt(jnp.maximum(jnp.sum(q * M_op(q)), 1e-30))
        Q.append(q / norm)
    return jnp.stack(Q, axis=0)


def subspace_iteration(
    K_op,
    M_op,
    B_init: jax.Array,
    n_iter: int,
    cg_iterations: int,
    cg_tol: float,
) -> tuple[jax.Array, jax.Array]:
    """Find the lowest ``k`` modes of ``K u = ω² M u`` via subspace iteration.

    Parameters
    ----------
    K_op, M_op:
        Matrix-free callables ``v → K v``, ``v → M v``.
    B_init:
        Initial subspace ``(k, 3, Nx, Ny, Nz)``.  Better if M-conformant
        (e.g. constrained DOFs already zero), but the orthonormalisation
        step will tolerate sloppy inputs.
    n_iter:
        Number of outer subspace iterations.
    cg_iterations, cg_tol:
        Inner CG solver parameters for the ``K^{-1}`` step.

    Returns
    -------
    eigvals:
        Shape ``(k,)``, ascending, in units of ω² (rad/s)².
    eigvecs:
        Shape ``(k, 3, Nx, Ny, Nz)``, ``M``-orthonormal.  ``eigvecs[0]`` is
        the lowest-frequency mode.
    """
    k = B_init.shape[0]

    def _cg_solve(b: jax.Array) -> jax.Array:
        x, _ = jax.scipy.sparse.linalg.cg(
            K_op,
            b,
            x0=jnp.zeros_like(b),
            tol=cg_tol,
            maxiter=cg_iterations,
        )
        return x

    B = B_init
    eigvals = jnp.zeros((k,), dtype=B.dtype)
    for _ in range(n_iter):
        # 1. Inverse iteration B' = K^{-1} M B  (column-wise).
        # Loop is unrolled by tracing; k is small (≤ ~6).
        MB_cols = [M_op(B[i]) for i in range(k)]
        B_inv_cols = [_cg_solve(MB_cols[i]) for i in range(k)]
        B_inv = jnp.stack(B_inv_cols, axis=0)

        # 2. M-orthonormalise.
        B = _m_orthonormalize(B_inv, M_op)

        # 3. Rayleigh-Ritz: K_red = B^T K B, symmetric. M_red ≈ I after step 2.
        KB_cols = [K_op(B[i]) for i in range(k)]
        KB = jnp.stack(KB_cols, axis=0)
        K_red = jnp.einsum("ncxyz,mcxyz->nm", B, KB)
        K_red = 0.5 * (K_red + K_red.T)  # symmetrise vs. drift

        # 4. Small dense eigenproblem; eigh returns ascending eigenvalues.
        eigvals, eigvecs = jnp.linalg.eigh(K_red)

        # 5. Rotate basis so each row is an eigenvector (ascending order).
        # B_new[m] = Σ_n eigvecs[n, m] B[n]
        B = jnp.einsum("nm,ncxyz->mcxyz", eigvecs, B)

    return eigvals, B


# ---------------------------------------------------------------------------
# ElasticityEigenmodes — concrete MechanicalModel
# ---------------------------------------------------------------------------


@autoinit
class ElasticityEigenmodes(MechanicalModel):
    """Density-dependent mechanical modes via a matrix-free eigensolve.

    Each call to :meth:`equilibrium_displacement` runs the subspace
    iteration on the current ρ (restricted to the Si subdomain — see the
    module docstring), projects the supplied body-force field onto the
    lowest modes, and returns the displacement.

    Parameters
    ----------
    young_modulus:
        Material Young's modulus ``E`` of the fully-dense solid (Pa).
    poisson_ratio:
        Material Poisson ratio.  0.27 is reasonable for (100) Si.
    density:
        Mass density of the fully-dense solid (kg/m³).
    voxel_size_m:
        Physical voxel spacing.
    free_dof_mask:
        Bool ``(3, Nx, Ny, Nz)``, ``True`` = free DOF.  Build with
        :func:`build_dof_mask`.
    n_modes:
        Number of eigenmodes used in the modal projection.
    n_subspace:
        Subspace iteration block size.  Defaults to ``2 * n_modes``; a
        slightly larger block dampens convergence stalls when modes are
        nearly degenerate.
    p_stiffness, eps_stiffness_min:
        SIMP stiffness penalty exponent and floor.
    eps_mass_min:
        Mass interpolation floor.
    n_subspace_iter:
        Number of outer subspace iterations.  6-10 typical; warm-start from
        analytical modes converges in fewer.
    cg_iterations, cg_tol:
        Inner CG parameters for the ``K^{-1}`` step.
    initial_subspace:
        Optional warm-start basis ``(n_subspace, 3, Nx, Ny, Nz)``.  If
        ``None``, a deterministic basis seeded with axis-aligned unit
        vectors (modulated by a low-frequency profile along each axis) is
        used.  For best results, pass the analytical modes from
        :func:`doubly_clamped_beam_modes`.
    rho_mask_threshold:
        Density threshold ``τ`` for the Si-restriction masks: a cell
        contributes to the elastic body iff ``ρ > τ``.  The caller's
        ``free_dof_mask`` anchors / symmetry planes are preserved
        (intersection only narrows the free subspace).  Default ``0.5``
        is appropriate for post-projection densities hardened toward
        {0, 1}.
    """

    young_modulus: float = frozen_field()
    poisson_ratio: float = frozen_field(default=0.27)
    density: float = frozen_field()
    voxel_size_m: tuple[float, float, float] = frozen_field()
    free_dof_mask: jax.Array = frozen_field()
    n_modes: int = frozen_field(default=1)
    n_subspace: int | None = frozen_field(default=None)

    p_stiffness: float = frozen_field(default=3.0)
    eps_stiffness_min: float = frozen_field(default=1e-3)
    eps_mass_min: float = frozen_field(default=1e-6)

    n_subspace_iter: int = frozen_field(default=8)
    cg_iterations: int = frozen_field(default=100)
    cg_tol: float = frozen_field(default=1e-5)

    initial_subspace: jax.Array | None = frozen_field(default=None)

    rho_mask_threshold: float = frozen_field(default=0.5)

    # ------------------------------------------------------------------

    def _si_masks(self, rho: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Compute the element-level and node-level Si masks for ``rho``.

        Returns ``(element_mask, node_mask)``:

        * ``element_mask`` shape ``(Nx-1, Ny-1, Nz-1)``, ``True`` iff all
          8 corner cells satisfy ``ρ > rho_mask_threshold``.  Multiplies
          the strain energy so only fully-Si elements contribute
          stiffness — boundary mixed elements drop out and the Si surface
          is traction-free.
        * ``node_mask`` shape ``(Nx, Ny, Nz)``, ``True`` iff the node is a
          corner of at least one fully-Si element.  Used to extend the
          caller's free-DOF mask so air nodes don't appear as
          zero-eigenvalue rigid-body modes.

        Both masks are wrapped in :func:`jax.lax.stop_gradient` — the
        comparison ``ρ > τ`` is non-differentiable, and gradients reach
        the modes through the SIMP-scaled Lamé / density on Si cells.
        """
        nx, ny, nz = rho.shape
        si_corner = (rho > self.rho_mask_threshold).astype(jnp.bool_)
        # Element mask: AND of the 8 corner indicators.
        elem = si_corner[0 : nx - 1, 0 : ny - 1, 0 : nz - 1]
        for bx in (0, 1):
            for by in (0, 1):
                for bz in (0, 1):
                    if (bx, by, bz) == (0, 0, 0):
                        continue
                    elem = jnp.logical_and(
                        elem,
                        si_corner[bx : nx - 1 + bx, by : ny - 1 + by, bz : nz - 1 + bz],
                    )
        # Node mask: True iff any of the 8 surrounding elements is fully-Si.
        # Scatter-OR the element mask back to (Nx, Ny, Nz).
        node = jnp.zeros((nx, ny, nz), dtype=jnp.bool_)
        for bx in (0, 1):
            for by in (0, 1):
                for bz in (0, 1):
                    node = node.at[bx : bx + nx - 1, by : by + ny - 1, bz : bz + nz - 1].set(
                        jnp.logical_or(
                            node[bx : bx + nx - 1, by : by + ny - 1, bz : bz + nz - 1],
                            elem,
                        )
                    )
        return jax.lax.stop_gradient(elem), jax.lax.stop_gradient(node)

    def _lame_parameters(self, rho: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Per-cell Lamé μ, λ from SIMP-scaled Young's modulus."""
        scale = _simp_stiffness_scale(rho, self.eps_stiffness_min, self.p_stiffness)
        E_field = scale * self.young_modulus
        nu = self.poisson_ratio
        lame_mu = E_field / (2.0 * (1.0 + nu))
        lame_lambda = E_field * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        return lame_mu, lame_lambda

    def _density_field(self, rho: jax.Array) -> jax.Array:
        return _linear_mass_scale(rho, self.eps_mass_min) * self.density

    def _block_size(self) -> int:
        return self.n_subspace if self.n_subspace is not None else max(2, 2 * self.n_modes)

    def _default_initial_basis(self, rho_shape: tuple[int, int, int]) -> jax.Array:
        """Construct a deterministic warm-start basis when none is supplied.

        Each axis carries a one-half-cycle cosine profile, scaled into
        each of the three displacement components in turn.  The
        orthogonality won't be exact but it gives the subspace iteration
        a structured start over random noise.
        """
        nx, ny, nz = rho_shape
        ax = jnp.cos(jnp.pi * (jnp.arange(nx) + 0.5) / nx)
        ay = jnp.cos(jnp.pi * (jnp.arange(ny) + 0.5) / ny)
        az = jnp.cos(jnp.pi * (jnp.arange(nz) + 0.5) / nz)
        # Bell-shape ⇒ +0.5 ramp instead of cosine that goes negative.
        bx = 0.5 * (1.0 - jnp.cos(jnp.pi * (jnp.arange(nx) + 0.5) / nx * 2.0))
        by = 0.5 * (1.0 - jnp.cos(jnp.pi * (jnp.arange(ny) + 0.5) / ny * 2.0))
        bz = 0.5 * (1.0 - jnp.cos(jnp.pi * (jnp.arange(nz) + 0.5) / nz * 2.0))
        bell = bx[:, None, None] * by[None, :, None] * bz[None, None, :]
        ramps = [
            ax[:, None, None] * jnp.ones_like(bell),
            ay[None, :, None] * jnp.ones_like(bell),
            az[None, None, :] * jnp.ones_like(bell),
        ]
        k = self._block_size()
        vectors: list[jax.Array] = []
        for i in range(k):
            comp = i % 3
            spatial = ramps[i % 3] if i < 3 else (bell * ramps[i % 3])
            vec = jnp.zeros((3, nx, ny, nz), dtype=jnp.float32)
            vec = vec.at[comp].set(spatial.astype(jnp.float32))
            vectors.append(vec)
        return jnp.stack(vectors, axis=0)

    def _prepare_initial_basis(self, rho_shape: tuple[int, int, int]) -> jax.Array:
        k = self._block_size()
        if self.initial_subspace is None:
            return self._default_initial_basis(rho_shape)
        # User-supplied warm-start; pad with default vectors if too few.
        supplied = self.initial_subspace.astype(jnp.float32)
        if supplied.shape[0] >= k:
            return supplied[:k]
        deficit = k - supplied.shape[0]
        filler = self._default_initial_basis(rho_shape)[:deficit]
        return jnp.concatenate([supplied, filler], axis=0)

    # ------------------------------------------------------------------

    def compute_modes(self, rho: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(eigvals, mode_shapes)`` for the current ρ.

        ``eigvals`` shape ``(n_modes,)`` in units of (rad/s)², ascending.
        ``mode_shapes`` shape ``(n_modes, 3, Nx, Ny, Nz)``, physical
        M-orthonormal (``∫ ρ φ_i φ_j dV = δ_ij``).

        Internally the eigensolver runs in non-dimensional units to keep
        CG well-conditioned: ``K`` is divided by ``E · L_voxel`` (a
        characteristic stiffness-diagonal magnitude) and ``M`` by
        ``ρ · V_voxel`` (the per-cell mass).  Eigenvalues come out in
        non-dimensional units ``λ = ω² · M_scale / K_scale`` and are
        rescaled back to ``(rad/s)²`` on the way out.  Mode shapes pick up
        a ``1/√M_scale`` factor so the physical M-orthonormality holds.
        """
        rho = jnp.clip(rho, 0.0, 1.0).astype(jnp.float32)
        lame_mu, lame_lambda = self._lame_parameters(rho)
        density_field = self._density_field(rho)

        # Si masks for the per-call Si subdomain (track ρ).  element_mask
        # zeroes the strain energy of any element with an air corner; node_mask
        # extends the free-DOF mask so air nodes don't show up as
        # zero-eigenvalue rigid-body modes.
        nx, ny, nz = int(rho.shape[0]), int(rho.shape[1]), int(rho.shape[2])
        rho_shape: tuple[int, int, int] = (nx, ny, nz)
        if nx >= 2 and ny >= 2 and nz >= 2:
            element_mask, node_mask = self._si_masks(rho)
        else:
            # 1-cell-axis fallback: _strain_energy switches to the collocated
            # form (no element concept), so pass a dummy element_mask and
            # restrict DOFs per-cell via ρ directly.
            element_mask = jnp.ones((max(1, nx - 1), max(1, ny - 1), max(1, nz - 1)), dtype=jnp.bool_)
            node_mask = jax.lax.stop_gradient(rho > self.rho_mask_threshold)
        eff_mask = jnp.logical_and(self.free_dof_mask, jnp.broadcast_to(node_mask[None], (3, *rho_shape)))

        # Numerical scaling (see docstring).  Without this, K and M differ by
        # ~20 orders of magnitude on typical photonic-MEMS voxel grids and
        # the inverse-iteration CG underflows to zero in float32.
        voxel_volume = self.voxel_size_m[0] * self.voxel_size_m[1] * self.voxel_size_m[2]
        L_min = float(min(self.voxel_size_m))
        K_scale = float(self.young_modulus) * L_min
        M_scale = float(self.density) * float(voxel_volume)
        inv_K_scale = jnp.asarray(1.0 / K_scale, dtype=jnp.float32)
        inv_M_scale = jnp.asarray(1.0 / M_scale, dtype=jnp.float32)

        def K_op_scaled(u: jax.Array) -> jax.Array:
            return (
                stiffness_operator_3d(u, lame_mu, lame_lambda, self.voxel_size_m, eff_mask, element_mask) * inv_K_scale
            )

        def M_op_scaled(u: jax.Array) -> jax.Array:
            return mass_operator_3d(u, density_field, self.voxel_size_m, eff_mask) * inv_M_scale

        B_init = self._prepare_initial_basis(rho_shape)
        B_init = B_init * eff_mask[None].astype(B_init.dtype)

        eigvals_dimless, eigvecs_scaled = subspace_iteration(
            K_op_scaled,
            M_op_scaled,
            B_init,
            n_iter=self.n_subspace_iter,
            cg_iterations=self.cg_iterations,
            cg_tol=self.cg_tol,
        )

        # Rescale.  Modes are ⟨·, M_scaled ·⟩-orthonormal so
        # ⟨φ_scaled, M φ_scaled⟩ = M_scale; dividing by √M_scale gives
        # physical M-orthonormality.
        eigvals_phys = eigvals_dimless * jnp.asarray(K_scale / M_scale, dtype=jnp.float32)
        eigvecs_phys = eigvecs_scaled * jnp.asarray(
            1.0 / jnp.sqrt(jnp.asarray(M_scale, dtype=jnp.float32)), dtype=jnp.float32
        )
        return eigvals_phys[: self.n_modes], eigvecs_phys[: self.n_modes]

    def equilibrium_displacement(
        self,
        rho: jax.Array,
        force: jax.Array,
    ) -> jax.Array:
        """Project the given body force onto the lowest modes and return ``u``.

        Modes are M-orthonormal so the modal mass is unity and the modal
        stiffness equals the eigenvalue; the modal coordinate reduces to
        ``q_i = ⟨φ_i, f⟩ / ω_i²``.
        """
        eigvals, mode_shapes = self.compute_modes(rho)
        voxel_volume = self.voxel_size_m[0] * self.voxel_size_m[1] * self.voxel_size_m[2]
        modal_force = jnp.einsum("ncxyz,cxyz->n", mode_shapes, force) * voxel_volume
        q = modal_force / jnp.maximum(eigvals, 1e-30)
        return jnp.einsum("n,ncxyz->cxyz", q, mode_shapes)
