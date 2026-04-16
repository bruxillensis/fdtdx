"""
Adjoint optimization of a silicon / silicon nitride downward Gaussian emitter.

The device couples light from a 300nm-wide input silicon waveguide into a
downward-emitted Gaussian beam launched into the buried oxide (no output
waveguide — this variant optimizes only for the farfield Gaussian).

Stack-up (bottom to top, z-axis) — simulation volume covers oxide only:
    1. Buried oxide  SiO2       -- PML below absorbs downward emission
    2. Silicon device layer      (220 nm)  -- input waveguide + 4x4 um optimizable device
    3. Interlayer oxide SiO2     (150 nm)
    4. Silicon nitride Si3N4     (400 nm)  -- 4x4 um optimizable device
    5. Top cladding SiO2

The Si substrate and freespace above are excluded to reduce memory; PMLs absorb
downward- and upward-propagating light at the boundaries. PEC at min_y enforces
the TE mirror symmetry so only the y>=0 half is simulated.

The script defines two Device regions (one in the Si layer, one in the SiN layer)
that are jointly optimized via gradient descent. The figure of merit is the sum
of two terms, both in [0, 1]:
(a) flux_efficiency_down: fraction of input Poynting flux directed downward, and
(b) gaussian_overlap: y-polarized Gaussian shape match at the focal plane, where
    the near-field phasor is angular-spectrum-propagated to the focal plane
    before computing the coherent overlap.
The additive form gives the optimizer a useful gradient on either term alone;
solutions that win on both simultaneously are implicitly rewarded.
"""

import re
import sys
import time
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import optax
import pytreeclass as tc
from loguru import logger

import fdtdx

# ---------------------------------------------------------------------------
# Material constants
# ---------------------------------------------------------------------------
PERMITTIVITY_SI = fdtdx.constants.relative_permittivity_silicon  # 12.25 (n~3.5)
PERMITTIVITY_SIO2 = fdtdx.constants.relative_permittivity_silica  # 2.25  (n~1.5)
PERMITTIVITY_SIN = 4.0  # Si3N4 at 1550 nm (n~2.0)
PERMITTIVITY_AIR = fdtdx.constants.relative_permittivity_air  # 1.0


def angular_spectrum_propagate(
    field_2d: jax.Array,
    dz_cells: float,
    n_medium: float,
    wavelength_cells: float,
) -> jax.Array:
    """Propagate a 2D complex field by distance dz using the angular spectrum method.

    Args:
        field_2d: complex array of shape (Nx, Ny).
        dz_cells: propagation distance in grid cells (positive = away from source).
        n_medium: refractive index of the propagation medium.
        wavelength_cells: free-space wavelength in grid cells.

    Returns:
        Propagated complex field of same shape (Nx, Ny).
    """
    nx, ny = field_2d.shape
    k = 2.0 * jnp.pi * n_medium / wavelength_cells

    # Spatial frequency grids (cycles per cell → rad per cell)
    kx = jnp.fft.fftfreq(nx) * 2.0 * jnp.pi
    ky = jnp.fft.fftfreq(ny) * 2.0 * jnp.pi
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")

    # kz: propagating modes have k² > kx²+ky²; evanescent modes get kz=0 (decay, don't propagate)
    kt_sq = kx_grid**2 + ky_grid**2
    kz = jnp.sqrt(jnp.maximum(k**2 - kt_sq, 0.0))

    spectrum = jnp.fft.fft2(field_2d)
    propagated_spectrum = spectrum * jnp.exp(1j * kz * dz_cells)
    return jnp.fft.ifft2(propagated_spectrum)


def _propagate_phasor_to_focal_plane(
    phasor_2d: jax.Array,
    grid_shape_x: int,
    grid_shape_y: int,
    propagation_distance_cells: float,
    x_offset_cells: float,
    beam_waist_cells: float,
    n_medium: float,
    wavelength_cells: float,
) -> tuple[jax.Array, int]:
    """Mirror half-domain phasor, zero-pad, propagate to focal plane.

    Assumes the half-domain is bounded at y=0 by a PEC symmetry plane (tangential
    E = 0). Under that boundary the transverse components behave as:
        Ex (tangential): odd about y=0  → sign-flipped when mirroring
        Ey (normal):     even about y=0 → unchanged when mirroring
    The phasor's first component axis must be ordered (Ex, Ey).

    Zero-pads in x to accommodate both the lateral shift from a tilted propagation
    and the beam width at the focal plane, preventing FFT wrap-around artifacts.

    Args:
        phasor_2d: complex array of shape (num_components, Nx, Ny) — half y-domain.
        grid_shape_x: number of grid cells in x.
        grid_shape_y: number of grid cells in y (half-domain).
        propagation_distance_cells: distance to propagate in grid cells.
        x_offset_cells: lateral shift of focal point in x (used to size the padding).
        beam_waist_cells: target Gaussian waist at the focal plane, in grid cells.
        n_medium: refractive index.
        wavelength_cells: free-space wavelength in grid cells.

    Returns:
        Tuple of (propagated field of shape (num_components, Nx_padded, Ny), x_pad)
        where x_pad is the number of cells padded on each side of x.
    """
    num_components = phasor_2d.shape[0]
    # Mirror about y=0 for full domain with per-component parity under PEC:
    # Ex (component 0) flips sign, Ey (component 1) is unchanged.
    sign_per_component = jnp.array([-1.0, 1.0]).reshape(-1, 1, 1)
    reflected = phasor_2d[:, :, :0:-1] * sign_per_component
    mirrored = jnp.concatenate([reflected, phasor_2d], axis=2)  # (C, Nx, 2*Ny-1)

    # Zero-pad x to cover both the tilt shift AND the beam footprint at focus.
    x_pad = int(abs(x_offset_cells) + 4.0 * beam_waist_cells + 20)
    mirrored = jnp.pad(mirrored, ((0, 0), (x_pad, x_pad), (0, 0)))

    propagated_components = []
    for c in range(num_components):
        prop = angular_spectrum_propagate(
            mirrored[c],
            dz_cells=propagation_distance_cells,
            n_medium=n_medium,
            wavelength_cells=wavelength_cells,
        )
        propagated_components.append(prop)
    propagated_full = jnp.stack(propagated_components, axis=0)

    # Crop back to the y>=0 half
    return propagated_full[:, :, grid_shape_y - 1 :], x_pad


def gaussian_overlap(
    phasor: jax.Array,
    grid_shape_y: int,
    grid_shape_x: int,
    beam_waist_cells: float,
    x_offset_cells: float,
    propagation_distance_cells: float,
    n_medium: float,
    wavelength_cells: float,
    propagation_sign: float = 1.0,
) -> jax.Array:
    """Overlap of a propagated phasor field with a y-polarized Gaussian at its waist.

    The near-field phasor recorded at the detector is propagated to the focal plane
    via the angular spectrum method, then compared with the target Gaussian at its
    waist (focus). Evaluating at the waist avoids near-field artifacts and removes
    the curvature term from the target: the Gaussian is exp(-ρ²/w₀²) with a tilt
    carrier exp(ik·x·sinθ).

    The target mode is y-polarized (ψ = (0, g)) because the input waveguide is a
    TE (Ey-dominant) mode. The numerator matches only Ey to g, while the denominator
    includes |Ex|² + |Ey|², so any power leaked into Ex counts against the overlap.

    Uses the normalized complex inner product:
        η = |⟨Ey, g⟩|² / ((|Ex|² + |Ey|²) · |g|²)

    Args:
        phasor: complex array of shape (num_freq, num_components, Nx, Ny, 1).
            Components must be ordered (Ex, Ey).
        grid_shape_y: number of grid cells in y (half-domain, PEC at min_y).
        grid_shape_x: number of grid cells in x.
        beam_waist_cells: Gaussian beam waist w₀ at the focal point, in grid cells.
        x_offset_cells: x-position of the focal point relative to detector center, in grid cells.
        propagation_distance_cells: distance from detector to focal plane, in grid cells.
        n_medium: refractive index of the medium (e.g. 1.5 for SiO₂).
        wavelength_cells: free-space wavelength λ₀ in grid cells (= λ₀ / resolution).
        propagation_sign: +1.0 for upward emission (focus above detector),
                          -1.0 for downward emission (focus below detector).

    Returns:
        Real scalar overlap in [0, 1].
    """
    phasor_2d = phasor[0, :, :, :, 0]  # (num_components, Nx, Ny)
    prop_field, x_pad = _propagate_phasor_to_focal_plane(
        phasor_2d, grid_shape_x, grid_shape_y,
        propagation_distance_cells=propagation_sign * propagation_distance_cells,
        x_offset_cells=x_offset_cells,
        beam_waist_cells=beam_waist_cells,
        n_medium=n_medium,
        wavelength_cells=wavelength_cells,
    )

    nx_padded = grid_shape_x + 2 * x_pad
    xs = jnp.arange(nx_padded) - (nx_padded - 1) / 2.0
    ys = jnp.arange(grid_shape_y)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")

    k = 2.0 * jnp.pi * n_medium / wavelength_cells
    theta = jnp.arctan2(x_offset_cells, propagation_distance_cells)
    rho_sq = (xx - x_offset_cells) ** 2 + yy**2

    amplitude = jnp.exp(-rho_sq / beam_waist_cells**2)
    phase = k * xx * jnp.sin(theta)
    gauss = amplitude * jnp.exp(1j * phase)

    # Match only Ey to the target; Ex contributes only to the denominator so that
    # polarization leakage costs the device overlap.
    ey_propagated = prop_field[1]
    overlap_c = jnp.sum(ey_propagated * jnp.conj(gauss))
    overlap_power = jnp.abs(overlap_c) ** 2

    field_power = jnp.sum(jnp.abs(prop_field) ** 2)  # |Ex|² + |Ey|²
    gauss_power = jnp.sum(jnp.abs(gauss) ** 2)

    return overlap_power / jnp.maximum(field_power * gauss_power, 1e-30)


def plot_downward_profile(
    phasor_field: jax.Array,
    grid_shape_y: int,
    grid_shape_x: int,
    beam_waist_cells: float,
    x_offset_cells: float,
    propagation_distance_cells: float,
    n_medium: float,
    wavelength_cells: float,
    resolution: float,
    save_path,
):
    """Plot the propagated downward field profile against the target Gaussian at the focal plane."""
    phasor_2d = phasor_field[0, :, :, :, 0]  # (num_components, Nx, Ny)
    prop_field, x_pad = _propagate_phasor_to_focal_plane(
        phasor_2d, grid_shape_x, grid_shape_y,
        propagation_distance_cells=-propagation_distance_cells,  # downward
        x_offset_cells=x_offset_cells,
        beam_waist_cells=beam_waist_cells,
        n_medium=n_medium,
        wavelength_cells=wavelength_cells,
    )

    nx_padded = grid_shape_x + 2 * x_pad
    xs = jnp.arange(nx_padded) - (nx_padded - 1) / 2.0
    ys = jnp.arange(grid_shape_y)
    xx, yy = jnp.meshgrid(xs, ys, indexing="ij")

    # Target Gaussian at focal plane (waist — no curvature)
    k = 2.0 * jnp.pi * n_medium / wavelength_cells
    theta = jnp.arctan2(x_offset_cells, propagation_distance_cells)
    rho_sq = (xx - x_offset_cells) ** 2 + yy**2
    amplitude = jnp.exp(-rho_sq / beam_waist_cells**2)
    phase = k * xx * jnp.sin(theta)
    gauss = amplitude * jnp.exp(1j * phase)

    # Simulated field: sum intensity over components
    sim_intensity = jnp.sum(jnp.abs(prop_field) ** 2, axis=0)  # (Nx_crop, Ny)
    gauss_intensity = jnp.abs(gauss) ** 2

    # Normalize both to unit total power (L2 norm), matching the overlap metric
    sim_norm = sim_intensity / jnp.maximum(sim_intensity.sum(), 1e-30)
    gauss_norm = gauss_intensity / jnp.maximum(gauss_intensity.sum(), 1e-30)

    # Convert grid cells to microns
    xs_um = float(resolution * 1e6) * xs
    ys_um = float(resolution * 1e6) * ys

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 1D cut along x at y=0
    ax = axes[0]
    ax.plot(xs_um, sim_norm[:, 0], label="Simulated", linewidth=1.5)
    ax.plot(xs_um, gauss_norm[:, 0], "--", label="Target Gaussian", linewidth=1.5)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("Intensity (unit total power)")
    ax.set_title("x-cut at y=0")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2D simulated
    ax = axes[1]
    extent = [float(ys_um[0]), float(ys_um[-1]), float(xs_um[0]), float(xs_um[-1])]
    im = ax.imshow(
        sim_norm, aspect="auto", origin="lower", extent=extent, cmap="inferno"
    )
    ax.set_xlabel("y (μm)")
    ax.set_ylabel("x (μm)")
    ax.set_title("Simulated |E|²")
    plt.colorbar(im, ax=ax)

    # 2D target
    ax = axes[2]
    im = ax.imshow(
        gauss_norm, aspect="auto", origin="lower", extent=extent, cmap="inferno"
    )
    ax.set_xlabel("y (μm)")
    ax.set_ylabel("x (μm)")
    ax.set_title("Target Gaussian |E|²")
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def _find_latest_seed_iter(params_dir: Path, device_names: list[str]) -> int:
    """Return the highest iter_idx for which every device has a saved params file.

    Ignores iter_idx == -1 (the pre-optimization random init).
    """
    pattern = re.compile(r"^params_(-?\d+)_(.+)\.npy$")
    iters_per_device: dict[str, set[int]] = {n: set() for n in device_names}
    for p in params_dir.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        iter_idx = int(m.group(1))
        rest = m.group(2)
        # Match longest device name first so "Si_Layer_0" isn't caught by "Si_Layer_0_somekey"
        for dev_name in sorted(device_names, key=len, reverse=True):
            if rest == dev_name or rest.startswith(dev_name + "_"):
                iters_per_device[dev_name].add(iter_idx)
                break
    common = set.intersection(*iters_per_device.values()) if iters_per_device else set()
    common.discard(-1)
    if not common:
        missing = [n for n, s in iters_per_device.items() if not s]
        raise FileNotFoundError(
            f"No common iteration across all devices in {params_dir}. "
            f"Devices with no matching files: {missing}"
        )
    return max(common)


def _load_seed_params(
    params: fdtdx.ParameterContainer,
    seed_dir: Path,
    iter_idx: int | None,
) -> fdtdx.ParameterContainer:
    """Overwrite `params` values with arrays loaded from `{seed_dir}/params_{iter}_{name}.npy`.

    The returned container has the same keys/shapes/dtypes as the input; only values change.
    """
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Seed directory does not exist: {seed_dir}")

    device_names = list(params.keys())
    if iter_idx is None:
        iter_idx = _find_latest_seed_iter(seed_dir, device_names)
        logger.info(f"Auto-selected latest seed iter_idx={iter_idx} from {seed_dir}")
    else:
        logger.info(f"Loading seed iter_idx={iter_idx} from {seed_dir}")

    new_params: fdtdx.ParameterContainer = {}
    for name, current in params.items():
        if isinstance(current, dict):
            loaded: dict[str, jax.Array] = {}
            for k, v in current.items():
                path = seed_dir / f"params_{iter_idx}_{name}_{k}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing seed file: {path}")
                arr = np.load(path)
                if tuple(arr.shape) != tuple(v.shape):
                    raise ValueError(
                        f"Shape mismatch for {name}[{k}]: seed {arr.shape} vs expected {v.shape}"
                    )
                loaded[k] = jnp.asarray(arr, dtype=v.dtype)
            new_params[name] = loaded
        else:
            path = seed_dir / f"params_{iter_idx}_{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing seed file: {path}")
            arr = np.load(path)
            if tuple(arr.shape) != tuple(current.shape):
                raise ValueError(
                    f"Shape mismatch for {name}: seed {arr.shape} vs expected {current.shape}"
                )
            new_params[name] = jnp.asarray(arr, dtype=current.dtype)
        logger.info(f"  loaded seed for device '{name}'")
    return new_params


def main(
    seed: int,
    evaluation: bool,
    backward: bool,
    seed_from: str | None = None,
    seed_iter: int | None = None,
):
    logger.info(f"{seed=}")

    exp_logger = fdtdx.Logger(
        experiment_name="antenna_emitter",
        name=None,
    )
    key = jax.random.PRNGKey(seed=seed)

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)
    resolution = 25e-9  # 25 nm grid

    config = fdtdx.SimulationConfig(
        time=200e-15,
        resolution=resolution,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")
    logger.info(f"{config.max_travel_distance=}")

    # Gradient config for adjoint
    if not evaluation or backward:
        gradient_config = fdtdx.GradientConfig(
            recorder=fdtdx.Recorder(
                modules=[
                    fdtdx.LinearReconstructEveryK(k=5),
                    fdtdx.DtypeConversion(dtype=jnp.float8_e4m3fnuz),
                ]
            )
        )
        config = config.aset("gradient_config", gradient_config)

    placement_constraints, object_list = [], []

    # ------------------------------------------------------------------
    # Layer thicknesses
    # ------------------------------------------------------------------
    t_box = 1.0e-6          # buried oxide (PML below absorbs downward emission)
    t_si = 220e-9           # silicon device layer
    t_spacer = 150e-9       # interlayer oxide
    t_sin = 400e-9          # silicon nitride
    t_topclad = 750e-9      # top cladding oxide
    t_det_above_box_bottom = 0.35e-6  # downward detector sits this far above the BOX bottom (above PML)

    t_oxide_total = t_box + t_si + t_spacer + t_sin + t_topclad
    total_z = t_oxide_total  # no freespace — detectors are inside the oxide

    # Device / emitter dimensions
    device_length_x = 4.0e-6   # device size along propagation
    device_width_y = 4.0e-6    # device size in transverse direction
    waveguide_width = 300e-9   # input waveguide width
    waveguide_margin_in = 3.0e-6   # straight input waveguide length on the −x side
    waveguide_margin_out = 2.0e-6  # empty space on +x side — enough to capture
                                   # the downward beam at the detector plane;
                                   # farfield wrap-around is handled by x_pad.

    n_si_etch_levels = 3  # number of etch levels in Si device layer

    total_x = device_length_x + waveguide_margin_in + waveguide_margin_out
    # PEC at min_y exploits TE mirror symmetry → simulate only half the y-domain
    total_y = device_width_y / 2 + 4.0e-6

    # ------------------------------------------------------------------
    # Simulation volume
    # ------------------------------------------------------------------
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(total_x, total_y, total_z),
    )
    object_list.append(volume)

    # PML boundaries (PEC at min_y for TE symmetry → halves y-domain)
    # PEC is correct for TE (Ey-dominant): tangential E (Ex, Ez) = 0 at y=0,
    # while Ey (normal to boundary) is free and maximum at the symmetry plane.
    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=10,
        override_types={"min_y": "pec"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    mat_si = fdtdx.Material(permittivity=PERMITTIVITY_SI)
    mat_sio2 = fdtdx.Material(permittivity=PERMITTIVITY_SIO2)

    # ------------------------------------------------------------------
    # Layer stack (placed bottom-up relative to volume)
    # Si substrate and freespace above are excluded from the simulation volume
    # to reduce memory. PML at the bottom absorbs downward-propagating light.
    # ------------------------------------------------------------------
    # SiO2 oxide stack fills the full volume except the freespace above.
    # Si and SiN only exist in the device regions and waveguide.
    oxide_stack = fdtdx.UniformMaterialObject(
        name="Oxide_stack",
        partial_real_shape=(None, None, t_oxide_total),
        material=mat_sio2,
        color=fdtdx.colors.XKCD_ORANGE,
    )
    placement_constraints.append(
        oxide_stack.place_relative_to(volume, axes=2, own_positions=-1, other_positions=-1)
    )
    object_list.append(oxide_stack)


    # ------------------------------------------------------------------
    # Optimizable devices
    # ------------------------------------------------------------------
    voxel_size = resolution  # match resolution

    # Device in Si layer: patterns Si vs SiO2
    # 3 separate layers for etch levels — fabrication constraint (no floating Si)
    # is enforced via a penalty in the loss function rather than PillarDiscretization.
    si_device_materials = {
        "SiO2": mat_sio2,
        "Silicon": mat_si,
    }
    # Etch levels at 90nm, 150nm, 220nm → layer thicknesses (bottom to top)
    si_etch_levels = [90e-9, 150e-9, 220e-9]
    si_layer_thicknesses = [
        si_etch_levels[0],                                    # 90 nm
        si_etch_levels[1] - si_etch_levels[0],                # 60 nm
        si_etch_levels[2] - si_etch_levels[1],                # 70 nm
    ]

    si_layers: list[fdtdx.Device] = []
    z_offset = 0.0
    for i in range(n_si_etch_levels):
        t_layer = si_layer_thicknesses[i]
        layer = fdtdx.Device(
            name=f"Si_Layer_{i}",
            partial_real_shape=(device_length_x, device_width_y / 2, t_layer),
            materials=si_device_materials,
            param_transforms=[
                fdtdx.GaussianSmoothing2D(std_discrete=3),
                fdtdx.SubpixelSmoothedProjection(),
            ],
            partial_voxel_real_shape=(voxel_size, voxel_size, t_layer),
        )
        # Each layer sits at BOX top + cumulative offset for its etch level
        # y-edge at PEC boundary (min_y), left edge at waveguide_margin_in from volume min_x
        placement_constraints.extend([
            layer.place_relative_to(
                oxide_stack, axes=2, own_positions=-1, other_positions=-1,
                margins=t_box + z_offset,
            ),
            layer.place_relative_to(
                volume, axes=0, own_positions=-1, other_positions=-1,
                margins=waveguide_margin_in,
            ),
            layer.place_relative_to(volume, axes=1, own_positions=-1, other_positions=-1),
        ])
        si_layers.append(layer)
        object_list.append(layer)
        z_offset += t_layer

    # Device in SiN layer: patterns SiN vs SiO2
    sin_device_materials = {
        "SiO2": mat_sio2,
        "SiN": fdtdx.Material(permittivity=PERMITTIVITY_SIN),
    }
    sin_device = fdtdx.Device(
        name="SiN_Device",
        partial_real_shape=(device_length_x, device_width_y / 2, t_sin),
        materials=sin_device_materials,
        param_transforms=[
            fdtdx.GaussianSmoothing2D(std_discrete=3),
            fdtdx.SubpixelSmoothedProjection(),
        ],
        partial_voxel_real_shape=(voxel_size, voxel_size, t_sin),
    )
    # SiN device sits at: oxide_stack bottom + t_box + t_si + t_spacer
    # y-edge at PEC boundary (min_y), left edge at waveguide_margin_in from volume min_x
    placement_constraints.extend([
        sin_device.place_relative_to(
            oxide_stack, axes=2, own_positions=-1, other_positions=-1,
            margins=t_box + t_si + t_spacer,
        ),
        sin_device.place_relative_to(
            volume, axes=0, own_positions=-1, other_positions=-1,
            margins=waveguide_margin_in,
        ),
        sin_device.place_relative_to(volume, axes=1, own_positions=-1, other_positions=-1),
    ])
    object_list.append(sin_device)

    # ------------------------------------------------------------------
    # Input waveguide (300 nm wide Si, waveguide_margin_in long, left of device)
    # ------------------------------------------------------------------
    waveguide_in = fdtdx.UniformMaterialObject(
        name="Waveguide_in",
        partial_real_shape=(waveguide_margin_in, waveguide_width / 2, t_si),
        material=mat_si,
        color=fdtdx.colors.XKCD_LIGHT_BLUE,
    )
    placement_constraints.extend([
        waveguide_in.place_relative_to(volume, axes=1, own_positions=-1, other_positions=-1),
        waveguide_in.place_relative_to(si_layers[0], axes=0, own_positions=1, other_positions=-1),
        waveguide_in.place_relative_to(
            oxide_stack, axes=2, own_positions=-1, other_positions=-1,
            margins=t_box,
        ),
    ])
    object_list.append(waveguide_in)

    # ------------------------------------------------------------------
    # Mode source (TE fundamental in Si waveguide, propagating +x)
    # Z-extent limited to oxide stack region to exclude Si substrate from
    # mode computation (otherwise the solver finds the substrate slab mode).
    # ------------------------------------------------------------------
    source = fdtdx.ModePlaneSource(
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, None, t_oxide_total),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        direction="+",
        mode_index=0,
        filter_pol="te",
    )
    placement_constraints.extend([
        source.place_relative_to(
            volume,
            axes=0,
            own_positions=-1,
            other_positions=-1,
            grid_margins=bound_cfg.thickness_grid_minx + 4,
        ),
        source.place_relative_to(
            oxide_stack,
            axes=2,
            own_positions=-1,
            other_positions=-1,
        ),
    ])
    object_list.append(source)

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------
    # Input Poynting flux (for normalizing downward emission)
    flux_in_detector = fdtdx.PoyntingFluxDetector(
        name="in_flux",
        partial_grid_shape=(1, None, None),
        direction="+",
        switch=fdtdx.OnOffSwitch(
            fixed_on_time_steps=all_time_steps[-2 * period_steps :]
        ),
    )
    placement_constraints.append(
        flux_in_detector.place_relative_to(
            volume, axes=0, own_positions=-1, other_positions=-1,
            grid_margins=bound_cfg.thickness_grid_minx + 6,
        )
    )
    object_list.append(flux_in_detector)

    # Downward flux in the BOX (-z direction)
    flux_down_detector = fdtdx.PoyntingFluxDetector(
        name="down_flux",
        partial_grid_shape=(None, None, 1),
        direction="-",
        fixed_propagation_axis=2,
        switch=fdtdx.OnOffSwitch(
            fixed_on_time_steps=all_time_steps[-2 * period_steps :]
        ),
    )
    placement_constraints.append(
        flux_down_detector.place_relative_to(
            oxide_stack, axes=2, own_positions=-1, other_positions=-1,
            margins=t_det_above_box_bottom,
        )
    )
    object_list.append(flux_down_detector)

    # Phasor detector for Gaussian overlap (same plane as downward flux)
    phasor_detector = fdtdx.PhasorDetector(
        name="phasor_down",
        partial_grid_shape=(None, None, 1),
        wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength),),
        components=("Ex", "Ey"),
        switch=fdtdx.OnOffSwitch(
            period=period,
            start_time=0.75 * config.time,
            on_for_periods=3,
        ),
    )
    placement_constraints.append(
        phasor_detector.place_relative_to(
            oxide_stack, axes=2, own_positions=-1, other_positions=-1,
            margins=t_det_above_box_bottom,
        )
    )
    object_list.append(phasor_detector)

    # Energy detector (last step, for diagnostics)
    energy_last_step = fdtdx.EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[-1]),
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])
    object_list.append(energy_last_step)

    exclude_object_list: list[fdtdx.SimulationObject] = [energy_last_step]

    if evaluation:
        video_detector = fdtdx.EnergyDetector(
            name="video",
            as_slices=True,
            switch=fdtdx.OnOffSwitch(interval=10),
            exact_interpolation=True,
            num_video_workers=10,
        )
        placement_constraints.extend([*video_detector.same_position_and_size(volume)])
        exclude_object_list.append(video_detector)
        object_list.append(video_detector)
        if backward:
            backward_video_detector = fdtdx.EnergyDetector(
                name="backward_video",
                as_slices=True,
                inverse=True,
                switch=fdtdx.OnOffSwitch(interval=10),
                exact_interpolation=True,
                num_video_workers=10,
            )
            placement_constraints.extend(
                [*backward_video_detector.same_position_and_size(volume)]
            )
            exclude_object_list.append(backward_video_detector)
            object_list.append(backward_video_detector)

    # ------------------------------------------------------------------
    # Place all objects
    # ------------------------------------------------------------------
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )

    if seed_from is not None:
        params = _load_seed_params(params, Path(seed_from), seed_iter)

    start_idx = 0

    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------
    epochs = 501
    if not evaluation:
        schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=0.003,
            end_value=0.0003,
            warmup_steps=15,
            decay_steps=round(0.9 * epochs),
        )
        optimizer = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule)
        optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)
        opt_state: optax.OptState = optimizer.init(params)

    # Beta schedule for binarization
    # Two-phase ramp: linear 0.1→50 for 90% of epochs, then linear 50→200 for the
    # remaining 10%.  Avoids the previous jnp.inf jump that destabilized the last
    # few epochs.
    def custom_schedule(idx: chex.Numeric) -> chex.Numeric:
        phase1_end = round(0.9 * epochs)
        phase1 = optax.linear_schedule(0.1, 50, phase1_end)
        phase2 = optax.linear_schedule(50, 200, epochs - phase1_end)
        return jax.lax.cond(
            idx < phase1_end,
            lambda: phase1(idx),
            lambda: phase2(idx - phase1_end),
        )

    # ------------------------------------------------------------------
    # Save setup
    # ------------------------------------------------------------------
    exp_logger.savefig(
        exp_logger.cwd,
        "setup.png",
        fdtdx.plot_setup(
            config=config,
            objects=objects,
            exclude_object_list=exclude_object_list,
        ),
    )

    changed_voxels = exp_logger.log_params(
        iter_idx=-1,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
        beta=custom_schedule(start_idx),
    )

    _, tmp, _ = fdtdx.apply_params(arrays, objects, params, key, beta=custom_schedule(start_idx))
    tmp.sources[0].plot(exp_logger.cwd / "figures" / "mode.png")  # type: ignore

    # ------------------------------------------------------------------
    # Objective weights
    # ------------------------------------------------------------------
    fab_weight = 0.0  # no-floating-Si penalty (disabled while we stabilize the Gaussian objective)

    # Gaussian target parameters for overlap computation.
    # The focal point sits below the center of the silicon device layer. The
    # overlap function propagates the phasor to the focal plane (beam waist) via
    # angular spectrum before computing the coherent Ey-vs-Gaussian inner product.
    # propagation_sign=-1 in gaussian_overlap indicates downward emission.
    n_sio2 = float(jnp.sqrt(PERMITTIVITY_SIO2))

    focal_distance_from_si = 10.0e-6  # focal distance measured from Si layer center
    si_center_height = t_box + t_si / 2  # Si center above oxide_stack bottom
    det_height = t_det_above_box_bottom   # detector plane above oxide_stack bottom
    focal_z_distance = focal_distance_from_si - (si_center_height - det_height)

    # Diffraction-limited beam waist: w₀ = √(λ₀ · f / (π · n))
    # This is the tightest Gaussian focus at distance f (Rayleigh range = f).
    focal_beam_waist = float(jnp.sqrt(wavelength * focal_distance_from_si / (jnp.pi * n_sio2)))
    focal_x_offset = 0.0  # lateral offset of focal point from detector center (0 = straight down)

    beam_waist_cells = focal_beam_waist / resolution
    x_offset_cells = focal_x_offset / resolution
    propagation_distance_cells = focal_z_distance / resolution
    wavelength_cells = wavelength / resolution

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------
    def loss_func(
        params: fdtdx.ParameterContainer,
        arrays: fdtdx.ArrayContainer,
        key: jax.Array,
        idx: int,
    ):
        arrays, new_objects, info = fdtdx.apply_params(
            arrays, objects, params, key, beta=custom_schedule(idx)
        )

        final_state = fdtdx.run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )

        _, arrays = final_state

        # --- Downward flux (Poynting-based) ---
        total_in_flux = arrays.detector_states[flux_in_detector.name]["poynting_flux"].sum()
        total_down_flux = arrays.detector_states[flux_down_detector.name]["poynting_flux"].sum()
        flux_efficiency_down = total_down_flux / jnp.maximum(total_in_flux, 1e-30)

        # --- Gaussian overlap (shape quality of downward emission) ---
        phasor_data = arrays.detector_states[phasor_detector.name]["phasor"]
        # phasor shape: (1, num_freq, num_components, Nx, Ny, 1); leading dim is latent time
        phasor_field = phasor_data[0]  # (num_freq, num_components, Nx, Ny, 1)
        phasor_grid_x = phasor_field.shape[2]
        phasor_grid_y = phasor_field.shape[3]
        overlap = gaussian_overlap(
            phasor_field,
            phasor_grid_y,
            phasor_grid_x,
            beam_waist_cells=beam_waist_cells,
            x_offset_cells=x_offset_cells,
            propagation_distance_cells=propagation_distance_cells,
            n_medium=n_sio2,
            wavelength_cells=wavelength_cells,
            propagation_sign=-1.0,  # downward emission: focus is below the detector
        )

        # Fabrication penalty: no floating Si (upper etch level needs support below).
        # fab_weight is 0 while we stabilize the Gaussian objective — kept for monitoring.
        fab_penalty = 0.0
        for i in range(n_si_etch_levels - 1):
            diff = params[si_layers[i + 1].name] - params[si_layers[i].name]
            fab_penalty = fab_penalty + jnp.mean(jnp.maximum(0.0, diff) ** 2)

        # Additive objective, both terms in [0, 1]. Each term alone gives the
        # optimizer a useful gradient even when the other is ~0, unlike the
        # multiplicative form we had previously.
        objective = 0.5 * flux_efficiency_down + 0.5 * overlap - fab_weight * fab_penalty

        if evaluation and backward:
            _, arrays = fdtdx.full_backward(
                state=final_state,
                objects=new_objects,
                config=config,
                key=key,
                record_detectors=True,
                reset_fields=True,
            )

        new_info = {
            "total_in_flux": total_in_flux,
            "total_down_flux": total_down_flux,
            "flux_efficiency_down": flux_efficiency_down,
            "gaussian_overlap": overlap,
            "fab_penalty": fab_penalty,
            "objective": objective,
            **info,
        }
        return -objective, (arrays, new_info)

    # ------------------------------------------------------------------
    # JIT compilation
    # ------------------------------------------------------------------
    compile_start_time = time.time()
    print("Started Compilation...")
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    idx_dummy_arr = jnp.asarray(start_idx, dtype=jnp.float32)
    if evaluation:
        jitted_loss = (
            jax.jit(loss_func, donate_argnames=["arrays"])
            .lower(params, arrays, key, idx_dummy_arr)
            .compile()
        )
    else:
        jitted_loss = (
            jax.jit(jax.value_and_grad(loss_func, has_aux=True), donate_argnames=["arrays"])
            .lower(params, arrays, key, idx_dummy_arr)
            .compile()
        )
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)
    print(f"Finished Compilation in {compile_delta_time:.1f} seconds")

    # ------------------------------------------------------------------
    # Optimization loop
    # ------------------------------------------------------------------
    optim_task_id = exp_logger.progress.add_task(
        "Optimization", total=1 if evaluation else epochs
    )
    for epoch in range(start_idx, start_idx + 1 if evaluation else epochs):
        run_start_time = time.time()
        key, subkey = jax.random.split(key)
        idx_arr = jnp.asarray(epoch, dtype=jnp.float32)

        if evaluation:
            loss, (arrays, info) = jitted_loss(params, arrays, subkey, idx_arr)
        else:
            (loss, (arrays, info)), grads = jitted_loss(params, arrays, subkey, idx_arr)

            updates, opt_state = optimizer.update(grads, opt_state, params)  # type: ignore
            info["lr"] = opt_state.inner_opt_state.hyperparams["learning_rate"]
            params = optax.apply_updates(params, updates)
            params = jax.tree_util.tree_map(lambda p: jnp.clip(p, 0, 1), params)
            info["grad_norm"] = optax.global_norm(grads)
            info["update_norm"] = optax.global_norm(updates)

        runtime_delta = time.time() - run_start_time
        info["runtime"] = runtime_delta
        info["loss"] = loss

        if evaluation:
            logger.info(f"{compile_delta_time=}")
            logger.info(f"{runtime_delta=}")

        changed_voxels = exp_logger.log_params(
            iter_idx=epoch,
            params=params,
            objects=objects,
            export_stl=True,
            export_figure=True,
            beta=custom_schedule(epoch),
        )
        info["changed_voxels"] = changed_voxels

        exp_logger.log_detectors(
            iter_idx=epoch, objects=objects, detector_states=arrays.detector_states
        )

        exp_logger.write(info)
        exp_logger.progress.update(optim_task_id, advance=1)

        # Plot downward profile vs target Gaussian
        phasor_data = arrays.detector_states[phasor_detector.name]["phasor"]
        phasor_field = phasor_data[0]  # (num_freq, num_components, Nx, Ny, 1)
        plot_downward_profile(
            phasor_field=phasor_field,
            grid_shape_y=phasor_field.shape[3],
            grid_shape_x=phasor_field.shape[2],
            beam_waist_cells=beam_waist_cells,
            x_offset_cells=x_offset_cells,
            propagation_distance_cells=propagation_distance_cells,
            n_medium=n_sio2,
            wavelength_cells=wavelength_cells,
            resolution=resolution,
            save_path=exp_logger.cwd / "figures" / f"downward_profile_{epoch:04d}.png",
        )

        if epoch % 25 == 0:
            logger.info(
                f"Epoch {epoch}: loss={float(loss):.4f} "
                f"flux_down={float(info['flux_efficiency_down']):.4f} "
                f"gauss_overlap={float(info['gaussian_overlap']):.4f}"
            )


if __name__ == "__main__":
    seed = 0
    evaluation = False
    backward = False
    seed_from: str | None = None
    seed_iter: int | None = None
    if len(sys.argv) > 1:
        seed = int(sys.argv[1])
    if len(sys.argv) > 2:
        evaluation = sys.argv[2].lower() in ("true", "1", "eval")
    if len(sys.argv) > 3:
        backward = sys.argv[3].lower() in ("true", "1")
    if len(sys.argv) > 4:
        seed_from = sys.argv[4]
    if len(sys.argv) > 5:
        seed_iter = None if sys.argv[5].lower() in ("latest", "auto", "") else int(sys.argv[5])
    main(
        seed=seed,
        evaluation=evaluation,
        backward=backward,
        seed_from=seed_from,
        seed_iter=seed_iter,
    )
