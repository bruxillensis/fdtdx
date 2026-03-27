"""
Slot waveguide phase shift analysis.

Sweeps the slot gap of a silicon slot waveguide, computes the TE fundamental mode
effective index at each gap using fdtdx's mode solver, and calculates the propagation
length required for a π phase shift relative to a nominal gap size.

Geometry (cross-section, propagation along x):
    SiO2 cladding everywhere, two Si rails separated by a variable air/SiO2 gap.

         z
         ^
         |   |← rail →|← gap →|← rail →|
         |   ┌────────┐       ┌────────┐
   wg ───┤   │   Si   │  gap  │   Si   │
         |   └────────┘       └────────┘
         +────────────────────────────────> y
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt

import fdtdx
from fdtdx.core.physics.modes import compute_mode


def build_cross_section(
    gap: float,
    rail_width: float,
    wg_height: float,
    clad_y: float,
    clad_z: float,
    resolution: float,
    eps_core: float,
    eps_clad: float,
) -> jax.Array:
    """Build inverse permittivity cross-section for a slot waveguide.

    Args:
        gap:        Gap width between the two Si rails (m).
        rail_width: Width of each Si rail (m).
        wg_height:  Height of the Si rails (m).
        clad_y:     Lateral cladding extent on each side (m).
        clad_z:     Vertical cladding above and below (m).
        resolution: Grid spacing (m).
        eps_core:   Relative permittivity of the rail material.
        eps_clad:   Relative permittivity of the cladding/gap material.

    Returns:
        inv_permittivities: shape (1, 1, ny, nz) — isotropic, x-propagation.
    """
    total_y = clad_y + rail_width + gap + rail_width + clad_y
    total_z = clad_z + wg_height + clad_z

    ny = int(round(total_y / resolution))
    nz = int(round(total_z / resolution))

    # Fill with cladding
    inv_eps = np.full((1, 1, ny, nz), 1.0 / eps_clad, dtype=np.float32)

    # Waveguide layer z indices
    iz0 = int(round(clad_z / resolution))
    iz1 = iz0 + int(round(wg_height / resolution))

    # Left rail y indices
    iy_l0 = int(round(clad_y / resolution))
    iy_l1 = iy_l0 + int(round(rail_width / resolution))

    # Right rail y indices (starts right after the gap)
    iy_r0 = iy_l1 + int(round(gap / resolution))
    iy_r1 = iy_r0 + int(round(rail_width / resolution))

    inv_eps[0, 0, iy_l0:iy_l1, iz0:iz1] = 1.0 / eps_core
    inv_eps[0, 0, iy_r0:iy_r1, iz0:iz1] = 1.0 / eps_core

    return jnp.array(inv_eps)


def compute_slot_neff(
    gap: float,
    rail_width: float,
    wg_height: float,
    clad_y: float,
    clad_z: float,
    resolution: float,
    eps_core: float,
    eps_clad: float,
    frequency: float,
) -> complex:
    inv_eps = build_cross_section(
        gap=gap,
        rail_width=rail_width,
        wg_height=wg_height,
        clad_y=clad_y,
        clad_z=clad_z,
        resolution=resolution,
        eps_core=eps_core,
        eps_clad=eps_clad,
    )
    _, _, neff = compute_mode(
        frequency=frequency,
        inv_permittivities=inv_eps,
        inv_permeabilities=1.0,
        resolution=resolution,
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    return complex(neff)


def main(
    wavelength: float = 1.55e-6,
    nominal_gap: float = 200e-9,
    gap_min: float = 20e-9,
    gap_max: float = 200e-9,
    n_gaps: int = 16,
    rail_width: float = 200e-9,
    wg_height: float = 220e-9,
    n_core: float = 3.48,   # Silicon
    n_clad: float = 1,   # SiO2
    resolution: float = 10e-9,
    clad_y: float = 600e-9,
    clad_z: float = 500e-9,
    save_plot: bool = True,
):
    frequency = fdtdx.constants.c / wavelength
    eps_core = n_core ** 2
    eps_clad = n_clad ** 2

    gap_values = np.linspace(gap_min, gap_max, n_gaps)

    # --- Nominal reference ---
    logger.info(f"Computing nominal mode at gap = {nominal_gap * 1e9:.0f} nm ...")
    neff_nominal = compute_slot_neff(
        gap=nominal_gap,
        rail_width=rail_width,
        wg_height=wg_height,
        clad_y=clad_y,
        clad_z=clad_z,
        resolution=resolution,
        eps_core=eps_core,
        eps_clad=eps_clad,
        frequency=frequency,
    )
    logger.info(f"  neff (nominal) = {neff_nominal.real:.5f} + {neff_nominal.imag:.2e}j")

    # --- Gap sweep ---
    neff_real = []
    delta_neff = []
    L_pi_um = []

    for i, gap in enumerate(gap_values):
        logger.info(f"[{i+1}/{n_gaps}] gap = {gap * 1e9:.1f} nm")
        neff = compute_slot_neff(
            gap=gap,
            rail_width=rail_width,
            wg_height=wg_height,
            clad_y=clad_y,
            clad_z=clad_z,
            resolution=resolution,
            eps_core=eps_core,
            eps_clad=eps_clad,
            frequency=frequency,
        )
        dn = neff.real - neff_nominal.real
        L_pi = wavelength / (2.0 * abs(dn)) if abs(dn) > 1e-7 else np.inf

        neff_real.append(neff.real)
        delta_neff.append(dn)
        L_pi_um.append(L_pi * 1e6)  # convert to μm

        logger.info(
            f"  neff = {neff.real:.5f}, Δneff = {dn:+.5f}, L_π = {L_pi * 1e6:.1f} μm"
        )

    neff_real = np.array(neff_real)
    delta_neff = np.array(delta_neff)
    L_pi_um = np.array(L_pi_um)
    gap_nm = gap_values * 1e9
    nominal_nm = nominal_gap * 1e9

    # --- Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(gap_nm, neff_real, "b-o", markersize=4)
    axes[0].axvline(nominal_nm, color="r", linestyle="--", label=f"Nominal ({nominal_nm:.0f} nm)")
    axes[0].axhline(neff_nominal.real, color="r", linestyle=":", alpha=0.5)
    axes[0].set_xlabel("Gap (nm)")
    axes[0].set_ylabel("Re(neff)")
    axes[0].set_title("TE Mode Effective Index vs Gap")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(gap_nm, delta_neff * 1e3, "g-o", markersize=4)
    axes[1].axvline(nominal_nm, color="r", linestyle="--", label=f"Nominal ({nominal_nm:.0f} nm)")
    axes[1].axhline(0, color="k", linestyle="-", linewidth=0.8)
    axes[1].set_xlabel("Gap (nm)")
    axes[1].set_ylabel("Δneff (×10⁻³)")
    axes[1].set_title("Δneff relative to nominal gap")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    finite_mask = np.isfinite(L_pi_um)
    axes[2].plot(gap_nm[finite_mask], L_pi_um[finite_mask], "m-o", markersize=4)
    axes[2].axvline(nominal_nm, color="r", linestyle="--", label=f"Nominal ({nominal_nm:.0f} nm)")
    axes[2].set_xlabel("Gap (nm)")
    axes[2].set_ylabel("L_π (μm)")
    axes[2].set_title("π Phase Shift Length vs Gap")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.suptitle(
        f"Si slot waveguide  |  λ = {wavelength * 1e9:.0f} nm  |  "
        f"rail {rail_width * 1e9:.0f} nm × {wg_height * 1e9:.0f} nm  |  "
        f"nominal gap = {nominal_nm:.0f} nm",
        fontsize=11,
    )
    plt.tight_layout()

    if save_plot:
        out_path = "slot_waveguide_phase_shift.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {out_path}")

    plt.show()


if __name__ == "__main__":
    seed = 0
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    key = jax.random.PRNGKey(seed)
    main()
