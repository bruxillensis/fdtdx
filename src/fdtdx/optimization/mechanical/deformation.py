"""First-order shape-derivative deformation of a permittivity array.

Implements the small-displacement perturbation

.. math::

    \\varepsilon_{\\text{def}}(x) \\approx \\varepsilon(x) - u(x) \\cdot \\nabla \\varepsilon(x),

valid when ``|u| ≪`` the simulation voxel size — i.e. when the moving
boundary shifts by less than a grid step.  This is the regime of pre-pull-in
MEMS phase shifters: 5–30 nm displacement at 25 nm voxel pitch.

The displacement field lives on the **design** (matrix-voxel) grid; the
permittivity array lives on the **simulation** (Yee) grid.  We upsample with
the same ``expand_matrix`` helper that :mod:`apply_params` uses to write
device permittivity into the sim grid, then pad with zeros outside the
device's grid_slice so the deformation is strictly local.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from fdtdx.core.misc import expand_matrix

__all__ = [
    "apply_displacement_to_permittivity",
    "apply_finite_displacement_to_permittivity",
]


def _centered_diff(arr: jax.Array, axis: int) -> jax.Array:
    """Centered finite difference along ``axis`` — used in place of
    :func:`jnp.gradient` so the return type is unambiguously ``jax.Array``.
    Boundary cells wrap; safe here because the displacement field is zero
    outside the device's grid_slice."""
    return 0.5 * (jnp.roll(arr, -1, axis=axis) - jnp.roll(arr, 1, axis=axis))


def _upsample_displacement(
    displacement_design: jax.Array,
    grid_points_per_voxel: tuple[int, int, int],
    sim_grid_shape: tuple[int, int, int],
    sim_slice: tuple[slice, slice, slice],
) -> jax.Array:
    """Upsample a (3, *design) displacement field to (3, Nx, Ny, Nz) on the sim grid.

    Zero outside ``sim_slice``.  The padding makes the deformation a strictly
    local operation — fields outside the device region see no perturbation.
    """
    upsampled_components = []
    for c in range(3):
        comp = displacement_design[c]
        if comp.ndim == 2:
            comp = comp[:, :, None]
        # expand_matrix expects shape (Nx_d, Ny_d, Nz_d) → (Nx_full, Ny_full, Nz_full).
        comp_full = expand_matrix(comp, grid_points_per_voxel)
        # Pad with zeros to the full sim grid.
        padded = jnp.zeros(sim_grid_shape, dtype=comp_full.dtype)
        padded = padded.at[sim_slice].set(comp_full)
        upsampled_components.append(padded)
    return jnp.stack(upsampled_components, axis=0)


def apply_displacement_to_permittivity(
    inv_permittivities_sim: jax.Array,
    displacement_design: jax.Array,
    grid_points_per_voxel: tuple[int, int, int],
    sim_grid_shape: tuple[int, int, int],
    sim_slice: tuple[slice, slice, slice],
    voxel_size_m: tuple[float, float, float],
) -> jax.Array:
    """Return ``inv_permittivities`` after a first-order rigid-body shift by ``u``.

    Parameters
    ----------
    inv_permittivities_sim:
        Sim-grid array ``(num_components, Nx, Ny, Nz)`` as stored in
        :class:`ArrayContainer`.  Only the diagonal entries (component indices
        0, 4, 8 for 9-component, or 0..n-1 for isotropic/diagonal) are
        perturbed; off-diagonal entries (anisotropy) are passed through
        unchanged.
    displacement_design:
        Displacement vector field on the device's design-voxel grid,
        ``(3, *design_grid_shape)`` in metres.
    grid_points_per_voxel:
        ``Device.single_voxel_grid_shape`` — how many sim cells fit in one
        design voxel along each axis.
    sim_grid_shape:
        Full sim-grid shape ``(Nx, Ny, Nz)``.
    sim_slice:
        Where the device sits inside the sim grid (``Device.grid_slice``).
    voxel_size_m:
        Sim-grid voxel size in metres along each axis.  Used to scale the
        gradient so the perturbation is dimensionally consistent.
    """
    # Convert inverse permittivity to permittivity for differencing; will
    # invert again at the end.
    eps_sim = 1.0 / jnp.clip(inv_permittivities_sim, 1e-30, None)

    u_sim = _upsample_displacement(
        displacement_design, grid_points_per_voxel, sim_grid_shape, sim_slice
    )  # (3, Nx, Ny, Nz)

    # ∂ε/∂x_a, divided by voxel size, gives a per-meter gradient.
    # Operate on each component channel separately; the spatial axes are the
    # last three of inv_permittivities_sim.
    def shift_one_component(eps_c: jax.Array) -> jax.Array:
        grad_eps = jnp.stack(
            [
                _centered_diff(eps_c, ax) / voxel_size_m[ax]
                for ax in range(3)
            ],
            axis=0,
        )
        # ε_def = ε - u · ∇ε
        return eps_c - jnp.sum(u_sim * grad_eps, axis=0)

    eps_def = jax.vmap(shift_one_component, in_axes=0, out_axes=0)(eps_sim)
    # Guard against negative-eps overshoot in regions where the gradient
    # is steep relative to the chosen voxel pitch.
    eps_def = jnp.maximum(eps_def, 0.5 * jnp.min(eps_sim))
    return 1.0 / eps_def


def apply_finite_displacement_to_permittivity(
    inv_permittivities_sim: jax.Array,
    displacement_design: jax.Array,
    grid_points_per_voxel: tuple[int, int, int],
    sim_grid_shape: tuple[int, int, int],
    sim_slice: tuple[slice, slice, slice],
    voxel_size_m: tuple[float, float, float],
) -> jax.Array:
    """Finite (large-displacement) counterpart of
    :func:`apply_displacement_to_permittivity`.

    The first-order form ``ε(x) → ε(x) − u·∇ε`` is a Taylor expansion valid
    only for ``|u| ≪`` one voxel.  Near pull-in a MEMS actuator deflects a
    sizeable fraction of the gap — often several voxels — where the
    first-order term produces wildly over-/under-shot ε (clamped to a
    sub-unity floor) that makes the FDTD diverge.  This routine instead
    performs an exact (linearly-interpolated) Lagrangian resample
    ``ε_def(x) = ε(x − u)`` via :func:`fdtdx.advect_density`, which stays
    physical (Si ↔ air) for arbitrarily large ``u`` and is differentiable.

    Same signature/units as :func:`apply_displacement_to_permittivity`
    (drop-in replacement).
    """
    from fdtdx.optimization.mechanical.pullin import advect_density

    eps_sim = 1.0 / jnp.clip(inv_permittivities_sim, 1e-30, None)
    u_sim = _upsample_displacement(
        displacement_design, grid_points_per_voxel, sim_grid_shape, sim_slice
    )  # (3, Nx, Ny, Nz), metres, zero outside the device slice
    disp_vox = jnp.stack(
        [u_sim[a] / voxel_size_m[a] for a in range(3)], axis=0
    )  # per-axis displacement in voxels

    def shift_one_component(eps_c: jax.Array) -> jax.Array:
        return advect_density(eps_c, disp_vox, order=1)

    eps_def = jax.vmap(shift_one_component, in_axes=0, out_axes=0)(eps_sim)
    return 1.0 / jnp.clip(eps_def, 1e-30, None)
