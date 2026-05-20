"""Analytical Euler-Bernoulli modes for a uniform doubly-clamped beam.

``doubly_clamped_beam_modes`` returns closed-form flexural mode shapes,
modal masses/stiffnesses and natural frequencies for a uniform rectangular
fixed-fixed beam.  It is a physics oracle (e.g. the elasticity unit tests
compare the FEM fundamental against the Euler-Bernoulli value) and a
quick back-of-the-envelope frequency estimate — not a mechanical model.
The production solver is
:class:`~fdtdx.optimization.mechanical.elasticity.ElasticityEigenmodes`.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

__all__ = [
    "doubly_clamped_beam_modes",
]


# Euler-Bernoulli roots for a doubly-clamped (fixed-fixed) beam.
# cos(β) cosh(β) = 1   →   β_n ≈ 4.73004, 7.85320, 10.99561, ...
# σ_n = (cosh β_n − cos β_n) / (sinh β_n − sin β_n).
_DOUBLY_CLAMPED_BETA = (4.730040744862704, 7.853204624095838, 10.995607838001671)
_DOUBLY_CLAMPED_SIGMA = (0.982502214576507, 1.000777311907267, 0.999966450125308)


def _doubly_clamped_mode_1d(xi: np.ndarray, n: int) -> np.ndarray:
    """Mass-normalized n-th flexural mode shape on ξ ∈ [0, 1].

    ``φ_n(ξ) = cosh(β_n ξ) − cos(β_n ξ) − σ_n (sinh(β_n ξ) − sin(β_n ξ))``,
    rescaled so ``∫₀¹ φ_n² dξ = 1``.  Zero at ξ ∈ {0, 1} with zero slope.
    """
    beta = _DOUBLY_CLAMPED_BETA[n]
    sigma = _DOUBLY_CLAMPED_SIGMA[n]
    raw = np.cosh(beta * xi) - np.cos(beta * xi) - sigma * (np.sinh(beta * xi) - np.sin(beta * xi))
    # Trapezoidal mass normalization on the supplied grid.
    integrand = raw * raw
    if xi.size > 1:
        dx = np.diff(xi)
        integral = float(np.sum(0.5 * (integrand[:-1] + integrand[1:]) * dx))
    else:
        integral = 0.0
    norm = np.sqrt(integral)
    if norm == 0.0:
        return raw
    return raw / norm


def doubly_clamped_beam_modes(
    *,
    length_m: float,
    width_m: float,
    thickness_m: float,
    young_modulus: float,
    density: float,
    grid_shape: tuple[int, int, int],
    beam_axis: int = 0,
    deflection_axis: int = 2,
    n_modes: int = 1,
) -> dict[str, Any]:
    """Build analytical mode shapes / modal stiffnesses for a uniform doubly-
    clamped Si beam.

    The beam runs along ``beam_axis`` of the device's design-voxel grid; the
    static deflection is along ``deflection_axis``.  The width axis is the
    remaining axis (in-plane lateral).  Modes are uniform across width and
    thickness — this is the standard Euler-Bernoulli approximation valid when
    ``thickness ≪ length`` and ``width ≪ length``.

    Returns a dict with keys:

    - ``mode_shapes`` — ``(n_modes, 3, *grid_shape)``, only the
      ``deflection_axis`` component is non-zero
    - ``modal_stiffness`` — ``(n_modes,)``, in N/m for unit-amplitude modes
    - ``modal_mass`` — ``(n_modes,)``, in kg
    - ``voxel_volume_m3`` — float, ΔV for converting ``Σ φ·f`` → ``∫ φ·f dV``
    - ``natural_frequencies_hz`` — ``(n_modes,)``, for diagnostics

    The kinematic normalization is ``∫ φ²(s) ds = 1`` along the beam axis
    (with ``s = x / L``), so ``K_i = ω_i² · m_i`` where ``m_i`` is the modal
    mass on the same normalization.
    """
    if beam_axis == deflection_axis:
        raise ValueError("beam_axis and deflection_axis must differ")
    if beam_axis not in (0, 1, 2) or deflection_axis not in (0, 1, 2):
        raise ValueError("axes must be one of 0, 1, 2")
    if n_modes < 1 or n_modes > len(_DOUBLY_CLAMPED_BETA):
        raise ValueError(f"n_modes must be in [1, {len(_DOUBLY_CLAMPED_BETA)}], got {n_modes}")

    nx, ny, nz = grid_shape
    grid = (nx, ny, nz)
    n_beam = grid[beam_axis]
    width_axis = ({0, 1, 2} - {beam_axis, deflection_axis}).pop()

    # ξ ∈ [0, 1] along beam (uniform sampling on the design grid).
    xi = np.linspace(0.0, 1.0, n_beam, endpoint=True)

    # Cross-section properties (beam local frame).
    cross_section_area = width_m * thickness_m  # m^2
    # Bending around the deflection axis: I = b·h³/12 where b is the in-plane
    # width perpendicular to deflection, h is the dimension along deflection.
    if deflection_axis == 2:  # z deflection
        I_bending = width_m * thickness_m**3 / 12.0
    else:
        # If deflection is in-plane the "thickness" here is the in-plane dim;
        # the user supplies whichever cross-section makes physical sense.
        I_bending = thickness_m * width_m**3 / 12.0
    mass_per_length = density * cross_section_area  # kg/m

    # Voxel volume — for converting a discrete ``Σ φ·f`` into ``∫ φ·f dV``.
    voxel_volume_m3 = (length_m * width_m * thickness_m) / float(nx * ny * nz)

    mode_shapes = np.zeros((n_modes, 3, nx, ny, nz), dtype=np.float32)
    modal_stiffness = np.zeros((n_modes,), dtype=np.float32)
    modal_mass = np.zeros((n_modes,), dtype=np.float32)
    natural_freqs = np.zeros((n_modes,), dtype=np.float32)

    for n in range(n_modes):
        phi_1d = _doubly_clamped_mode_1d(xi, n).astype(np.float32)
        # Build the 3D field: ψ uniform across width & deflection axes.
        psi_per_axis: list[Any] = [None, None, None]
        psi_per_axis[beam_axis] = phi_1d.reshape(tuple(n_beam if a == beam_axis else 1 for a in range(3)))
        psi_3d = np.ones(grid, dtype=np.float32) * psi_per_axis[beam_axis]
        mode_shapes[n, deflection_axis] = psi_3d

        # Modal mass: ∫ ρA · φ² dx = ρA · L · ∫₀¹ φ²(ξ) dξ = ρA · L · 1
        m_i = mass_per_length * length_m * 1.0
        # Natural frequency (rad/s): ω_n = (β_n / L)² · √(E·I / ρA)
        beta = _DOUBLY_CLAMPED_BETA[n]
        omega = (beta / length_m) ** 2 * np.sqrt(young_modulus * I_bending / mass_per_length)
        k_i = omega**2 * m_i

        modal_mass[n] = m_i
        natural_freqs[n] = omega / (2.0 * np.pi)
        modal_stiffness[n] = k_i

    return {
        "mode_shapes": jnp.asarray(mode_shapes),
        "modal_stiffness": jnp.asarray(modal_stiffness),
        "modal_mass": jnp.asarray(modal_mass),
        "voxel_volume_m3": float(voxel_volume_m3),
        "natural_frequencies_hz": jnp.asarray(natural_freqs),
    }
