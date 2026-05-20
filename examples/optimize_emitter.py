"""
Adjoint optimization of a silicon / silicon nitride downward Gaussian emitter.

Couples a 300 nm Si input waveguide into a downward-emitted Gaussian beam
launched into the buried oxide. The device has no output waveguide — it only
optimizes for the farfield Gaussian below the chip.

Stack-up (bottom to top, z) — simulation volume covers oxide only:
    1. Buried oxide  SiO2                     -- PML below absorbs downward emission
    2. Silicon device layer      (220 nm)     -- input waveguide + 3-level etched device
    3. Interlayer oxide SiO2     (150 nm)
    4. Silicon nitride           (400 nm)     -- optimizable device
    5. Top cladding SiO2

PEC at min_y exploits the TE mirror symmetry, so only the y>=0 half is simulated.

This script drives the optimization via :class:`fdtdx.Optimization`. The loss is
built from two objectives and two kinds of constraint:

Objectives (maximized):
  (a) flux_efficiency_down = P_down / P_source, the fraction of the injected
      source power radiated downward through the BOX.
  (b) gaussian_overlap, the coherent E_y overlap of the downward near-field
      (angular-spectrum propagated to the focal plane) with a target
      diffraction-limited Gaussian beam.

Constraints (minimized):
  (c) back_reflection_ratio = P_back / P_source, the fraction of source power
      scattered back into the input waveguide. Measured by a detector inside
      the TFSF *scattered-field* region behind the mode source — incident wave
      is cancelled there, so only reflections are captured.
  (d) MinLineSpace(140 nm) on each optimizable device, enforcing a 140 nm
      minimum feature size and minimum gap.
  (e) NoFloatingMaterial across the 3 etched Si layers — upper layers may not
      carry solid material above a void in the layer below (release-etch rule).

Source power normalization: a TFSF mode source injects a known P_inject into
the +x total-field region. A +x-facing detector inside the total-field region
measures P_forward = P_inject − P_back, so using P_forward alone as the
denominator allows efficiency > 1 whenever the device reflects. Here we recover
P_source = P_forward + P_back via a dedicated back-reflection detector in the
scattered-field region, keeping every efficiency in [0, 1] by construction.
"""

import sys
import time
from pathlib import Path

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


def angular_spectrum_propagate(
    field_2d: jax.Array,
    dz_cells: float,
    n_medium: float,
    wavelength_cells: float,
) -> jax.Array:
    """Propagate a 2D complex field by distance dz using the angular spectrum method."""
    nx, ny = field_2d.shape
    k = 2.0 * jnp.pi * n_medium / wavelength_cells

    kx = jnp.fft.fftfreq(nx) * 2.0 * jnp.pi
    ky = jnp.fft.fftfreq(ny) * 2.0 * jnp.pi
    kx_grid, ky_grid = jnp.meshgrid(kx, ky, indexing="ij")

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
    """Mirror half-domain phasor under PEC parity, zero-pad, propagate to focal plane.

    The first component axis must be ordered (Ex, Ey). Under PEC at y=0, Ex
    (tangential) is odd and Ey (normal) is even; this sets the mirror signs.
    """
    num_components = phasor_2d.shape[0]
    sign_per_component = jnp.array([-1.0, 1.0]).reshape(-1, 1, 1)
    reflected = phasor_2d[:, :, :0:-1] * sign_per_component
    mirrored = jnp.concatenate([reflected, phasor_2d], axis=2)

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

    The normalized complex inner product
        η = |⟨Ey, g⟩|² / ((|Ex|² + |Ey|²) · |g|²)
    ensures any power leaked into Ex counts against the overlap.
    """
    phasor_2d = phasor[0, :, :, :, 0]
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

    ey_propagated = prop_field[1]
    overlap_c = jnp.sum(ey_propagated * jnp.conj(gauss))
    overlap_power = jnp.abs(overlap_c) ** 2

    field_power = jnp.sum(jnp.abs(prop_field) ** 2)
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
    """Plot the propagated downward field profile against the target Gaussian."""
    phasor_2d = phasor_field[0, :, :, :, 0]
    prop_field, x_pad = _propagate_phasor_to_focal_plane(
        phasor_2d, grid_shape_x, grid_shape_y,
        propagation_distance_cells=-propagation_distance_cells,
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

    sim_intensity = jnp.sum(jnp.abs(prop_field) ** 2, axis=0)
    gauss_intensity = jnp.abs(gauss) ** 2

    sim_norm = sim_intensity / jnp.maximum(sim_intensity.sum(), 1e-30)
    gauss_norm = gauss_intensity / jnp.maximum(gauss_intensity.sum(), 1e-30)

    xs_um = float(resolution * 1e6) * xs
    ys_um = float(resolution * 1e6) * ys

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(xs_um, sim_norm[:, 0], label="Simulated", linewidth=1.5)
    ax.plot(xs_um, gauss_norm[:, 0], "--", label="Target Gaussian", linewidth=1.5)
    ax.set_xlabel("x (μm)")
    ax.set_ylabel("Intensity (unit total power)")
    ax.set_title("x-cut at y=0")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    extent = [float(ys_um[0]), float(ys_um[-1]), float(xs_um[0]), float(xs_um[-1])]
    im = ax.imshow(sim_norm, aspect="auto", origin="lower", extent=extent, cmap="inferno")
    ax.set_xlabel("y (μm)")
    ax.set_ylabel("x (μm)")
    ax.set_title("Simulated |E|²")
    plt.colorbar(im, ax=ax)

    ax = axes[2]
    im = ax.imshow(gauss_norm, aspect="auto", origin="lower", extent=extent, cmap="inferno")
    ax.set_xlabel("y (μm)")
    ax.set_ylabel("x (μm)")
    ax.set_title("Target Gaussian |E|²")
    plt.colorbar(im, ax=ax)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main(args):
    seed = args.seed_rng
    evaluation = args.evaluation
    backward = args.backward
    seed_from = args.seed_from
    seed_iter = args.seed_iter
    resume_from = args.resume_from

    logger.info(f"{seed=}")

    exp_logger = fdtdx.Logger(experiment_name="antenna_emitter", name=None)
    key = jax.random.PRNGKey(seed=seed)

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)
    resolution = 25e-9

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

    # Gradient config (omit for pure forward evaluation)
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

    placement_constraints: list = []
    object_list: list = []

    # ------------------------------------------------------------------
    # Layer thicknesses
    # ------------------------------------------------------------------
    t_box = 1.0e-6
    t_si = 220e-9
    t_spacer = 150e-9
    t_sin = 400e-9
    t_topclad = 750e-9
    t_det_above_box_bottom = 0.35e-6

    t_oxide_total = t_box + t_si + t_spacer + t_sin + t_topclad
    total_z = t_oxide_total

    device_length_x = 4.0e-6
    device_width_y = 4.0e-6
    waveguide_width = 300e-9
    waveguide_margin_in = 3.0e-6
    waveguide_margin_out = 2.0e-6

    n_si_etch_levels = 3

    total_x = device_length_x + waveguide_margin_in + waveguide_margin_out
    total_y = device_width_y / 2 + 4.0e-6

    # ------------------------------------------------------------------
    # Simulation volume + boundaries
    # ------------------------------------------------------------------
    volume = fdtdx.SimulationVolume(partial_real_shape=(total_x, total_y, total_z))
    object_list.append(volume)

    bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(
        thickness=10,
        override_types={"min_y": "pec"},
    )
    bound_dict, c_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
    placement_constraints.extend(c_list)
    object_list.extend(list(bound_dict.values()))

    # ------------------------------------------------------------------
    # Materials + oxide stack
    # ------------------------------------------------------------------
    mat_si = fdtdx.Material(permittivity=PERMITTIVITY_SI)
    mat_sio2 = fdtdx.Material(permittivity=PERMITTIVITY_SIO2)

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
    voxel_size = resolution

    si_device_materials = {"SiO2": mat_sio2, "Silicon": mat_si}
    si_etch_levels = [90e-9, 150e-9, 220e-9]
    si_layer_thicknesses = [
        si_etch_levels[0],
        si_etch_levels[1] - si_etch_levels[0],
        si_etch_levels[2] - si_etch_levels[1],
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

    all_device_names = [layer.name for layer in si_layers] + [sin_device.name]

    # ------------------------------------------------------------------
    # Input waveguide
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
    # Mode source (TFSF)
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
            volume, axes=0, own_positions=-1, other_positions=-1,
            grid_margins=bound_cfg.thickness_grid_minx + 4,
        ),
        source.place_relative_to(
            oxide_stack, axes=2, own_positions=-1, other_positions=-1,
        ),
    ])
    object_list.append(source)

    # ------------------------------------------------------------------
    # Detectors
    # ------------------------------------------------------------------
    # Forward flux in the total-field region (a few cells downstream of source).
    # This is P_inject − P_back since reflected waves also pass through it.
    flux_in_detector = fdtdx.PoyntingFluxDetector(
        name="in_flux",
        partial_grid_shape=(1, None, None),
        direction="+",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-2 * period_steps:]),
    )
    placement_constraints.append(
        flux_in_detector.place_relative_to(
            volume, axes=0, own_positions=-1, other_positions=-1,
            grid_margins=bound_cfg.thickness_grid_minx + 6,
        )
    )
    object_list.append(flux_in_detector)

    # Back-reflection flux in the TFSF scattered-field region, a few cells
    # UPSTREAM of the source. The TFSF correction cancels the incident wave
    # there so only reflected power is captured. direction="-" reports the
    # backward flux magnitude as a positive number. Used both for the
    # back-reflection constraint and to reconstruct P_source.
    back_flux_detector = fdtdx.PoyntingFluxDetector(
        name="back_flux",
        partial_grid_shape=(1, None, None),
        direction="-",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-2 * period_steps:]),
    )
    placement_constraints.append(
        back_flux_detector.place_relative_to(
            volume, axes=0, own_positions=-1, other_positions=-1,
            grid_margins=bound_cfg.thickness_grid_minx + 2,
        )
    )
    object_list.append(back_flux_detector)

    flux_down_detector = fdtdx.PoyntingFluxDetector(
        name="down_flux",
        partial_grid_shape=(None, None, 1),
        direction="-",
        fixed_propagation_axis=2,
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-2 * period_steps:]),
    )
    placement_constraints.append(
        flux_down_detector.place_relative_to(
            oxide_stack, axes=2, own_positions=-1, other_positions=-1,
            margins=t_det_above_box_bottom,
        )
    )
    object_list.append(flux_down_detector)

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
    # Place everything on the grid
    # ------------------------------------------------------------------
    key, subkey = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=object_list,
        config=config,
        constraints=placement_constraints,
        key=subkey,
    )

    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))

    # ------------------------------------------------------------------
    # Gaussian target parameters
    # ------------------------------------------------------------------
    n_sio2 = float(jnp.sqrt(PERMITTIVITY_SIO2))
    focal_distance_from_si = 10.0e-6
    si_center_height = t_box + t_si / 2
    det_height = t_det_above_box_bottom
    focal_z_distance = focal_distance_from_si - (si_center_height - det_height)

    focal_beam_waist = float(jnp.sqrt(wavelength * focal_distance_from_si / (jnp.pi * n_sio2)))
    focal_x_offset = 0.0

    beam_waist_cells = focal_beam_waist / resolution
    x_offset_cells = focal_x_offset / resolution
    propagation_distance_cells = focal_z_distance / resolution
    wavelength_cells = wavelength / resolution

    # ------------------------------------------------------------------
    # Beta schedule for SubpixelSmoothedProjection binarization.
    # Two-phase linear ramp: 0.1→50 over 90% of epochs, then 50→200 for the
    # remaining 10%.
    # ------------------------------------------------------------------
    epochs = 501
    phase1_end = round(0.9 * epochs)

    def beta_schedule(epoch: jax.Array) -> jax.Array:
        epoch_f = jnp.asarray(epoch, dtype=jnp.float32)
        t1 = jnp.clip(epoch_f / float(phase1_end), 0.0, 1.0)
        phase1_val = 0.1 + (50.0 - 0.1) * t1
        t2 = jnp.clip((epoch_f - phase1_end) / float(max(epochs - phase1_end, 1)), 0.0, 1.0)
        phase2_val = 50.0 + (200.0 - 50.0) * t2
        return jnp.where(epoch_f < phase1_end, phase1_val, phase2_val)

    # ------------------------------------------------------------------
    # simulate_fn (shared between optimization and evaluation)
    # ------------------------------------------------------------------
    def simulate_fn(params, arrays, objects, config, key, epoch):
        arrays, new_objects, _ = fdtdx.apply_params(
            arrays, objects, params, key, beta=beta_schedule(epoch)
        )
        final_state = fdtdx.run_fdtd(
            arrays=arrays, objects=new_objects, config=config, key=key,
        )
        _, arrays = final_state
        if evaluation and backward:
            _, arrays = fdtdx.full_backward(
                state=final_state, objects=new_objects, config=config, key=key,
                record_detectors=True, reset_fields=True,
            )
        return arrays

    # ------------------------------------------------------------------
    # Metric helpers — shared by the objectives and the evaluation path.
    # P_source is reconstructed as P_forward + P_back so all ratios stay in [0, 1]
    # regardless of how reflective the device is.
    # ------------------------------------------------------------------
    def _power_budget(arrays):
        in_fwd = arrays.detector_states[flux_in_detector.name]["poynting_flux"].sum()
        back = arrays.detector_states[back_flux_detector.name]["poynting_flux"].sum()
        down = arrays.detector_states[flux_down_detector.name]["poynting_flux"].sum()
        # Clip P_back to >=0 to guard against numerical jitter driving the
        # denominator of the ratios below the unbiased source power.
        back = jnp.maximum(back, 0.0)
        source = jnp.maximum(in_fwd + back, 1e-30)
        return in_fwd, back, down, source

    def _overlap_from_arrays(arrays):
        phasor_field = arrays.detector_states[phasor_detector.name]["phasor"][0]
        return gaussian_overlap(
            phasor_field,
            grid_shape_y=phasor_field.shape[3],
            grid_shape_x=phasor_field.shape[2],
            beam_waist_cells=beam_waist_cells,
            x_offset_cells=x_offset_cells,
            propagation_distance_cells=propagation_distance_cells,
            n_medium=n_sio2,
            wavelength_cells=wavelength_cells,
            propagation_sign=-1.0,
        )

    # ------------------------------------------------------------------
    # Objectives & constraints
    # ------------------------------------------------------------------
    def flux_down_metric(*, arrays, **_unused):
        in_fwd, back, down, source = _power_budget(arrays)
        ratio = down / source
        info = {
            "total_in_fwd_flux": in_fwd,
            "total_back_reflection": back,
            "total_down_flux": down,
            "total_source_power": source,
        }
        return ratio, info

    def gaussian_overlap_metric(*, arrays, **_unused):
        return _overlap_from_arrays(arrays), {}

    def back_reflection_metric(*, arrays, **_unused):
        _, back, _, source = _power_budget(arrays)
        return back / source, {}

    objectives = (
        fdtdx.FunctionObjective(
            name="flux_down",
            schedule=fdtdx.ConstantSchedule(value=0.5),
            fn=flux_down_metric,
        ),
        fdtdx.FunctionObjective(
            name="overlap",
            schedule=fdtdx.ConstantSchedule(value=0.5),
            fn=gaussian_overlap_metric,
        ),
    )

    constraints: list = [
        fdtdx.FunctionConstraint(
            name="back_reflection",
            schedule=fdtdx.ConstantSchedule(value=0.3),
            fn=back_reflection_metric,
        ),
        # No floating Si: upper etch levels may not carry material above a
        # void in the layer directly below (fab rule — unsupported features
        # fall off during release etch). Ramped in so the optimizer can
        # explore freely early and tighten to the fab rule as binarization
        # sets in.
        fdtdx.NoFloatingMaterial(
            name="no_floating_si",
            device_stack_names=tuple(layer.name for layer in si_layers),
            schedule=fdtdx.LinearSchedule(
                epoch_start=0,
                epoch_end=round(0.9 * epochs),
                start_value=0.0,
                end_value=1.0,
            ),
        ),
    ]
    for dev_name in all_device_names:
        constraints.append(
            fdtdx.MinLineSpace(
                name=f"line_space_{dev_name}",
                device_name=dev_name,
                min_line_width_m=140e-9,
                min_space_m=140e-9,
                schedule=fdtdx.LinearSchedule(
                    epoch_start=0,
                    epoch_end=round(0.9 * epochs),
                    start_value=0.0,
                    end_value=1.0,
                ),
            )
        )

    # ------------------------------------------------------------------
    # Save setup & initial params snapshot
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
    exp_logger.log_params(
        iter_idx=-1,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
        beta=beta_schedule(jnp.asarray(0.0, dtype=jnp.float32)),
    )
    _, tmp, _ = fdtdx.apply_params(arrays, objects, params, key, beta=beta_schedule(jnp.asarray(0.0)))
    tmp.sources[0].plot(exp_logger.cwd / "figures" / "mode.png")  # type: ignore

    # ------------------------------------------------------------------
    # Evaluation branch: one forward pass, plot, exit.
    # ------------------------------------------------------------------
    if evaluation:
        if seed_from is not None:
            params = fdtdx.load_seed_params(
                Path(seed_from), params,
                iter_idx=(None if seed_iter in (None, "latest", "auto", "") else int(seed_iter)),
            )

        compile_start = time.time()
        print("Started Compilation...")
        jit_task_id = exp_logger.progress.add_task("JIT", total=None)
        jitted_sim = (
            jax.jit(simulate_fn, donate_argnames=["arrays"])
            .lower(params, arrays, objects, config, key, jnp.asarray(0.0, dtype=jnp.float32))
            .compile()
        )
        compile_dt = time.time() - compile_start
        exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)
        print(f"Finished Compilation in {compile_dt:.1f} seconds")

        key, subkey = jax.random.split(key)
        run_start = time.time()
        arrays = jitted_sim(params, arrays, objects, config, subkey, jnp.asarray(0.0))
        run_dt = time.time() - run_start

        in_fwd, back, down, source = _power_budget(arrays)
        overlap = _overlap_from_arrays(arrays)
        info = {
            "runtime": run_dt,
            "compile_time": compile_dt,
            "total_in_fwd_flux": float(in_fwd),
            "total_back_reflection": float(back),
            "total_down_flux": float(down),
            "total_source_power": float(source),
            "flux_efficiency_down": float(down / source),
            "back_reflection_ratio": float(back / source),
            "gaussian_overlap": float(overlap),
        }
        logger.info(info)
        exp_logger.log_detectors(iter_idx=0, objects=objects, detector_states=arrays.detector_states)
        exp_logger.log_params(iter_idx=0, params=params, objects=objects,
                              export_stl=True, export_figure=True,
                              beta=beta_schedule(jnp.asarray(0.0)))
        exp_logger.write({k: np.asarray(v) for k, v in info.items()})

        phasor_field = arrays.detector_states[phasor_detector.name]["phasor"][0]
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
            save_path=exp_logger.cwd / "figures" / "downward_profile_eval.png",
        )
        return

    # ------------------------------------------------------------------
    # Optimization branch
    # ------------------------------------------------------------------
    schedule: optax.Schedule = optax.warmup_cosine_decay_schedule(
        init_value=1e-5,
        peak_value=0.003,
        end_value=0.0003,
        warmup_steps=15,
        decay_steps=round(0.9 * epochs),
    )
    optimizer = optax.inject_hyperparams(optax.nadam)(learning_rate=schedule)
    optimizer = optax.MultiSteps(optimizer, every_k_schedule=1)

    opt = fdtdx.Optimization(
        objects=objects,
        arrays=arrays,
        params=params,
        config=config,
        simulate_fn=simulate_fn,
        optimizer=optimizer,
        objectives=objectives,
        constraints=tuple(constraints),
        total_epochs=epochs,
        param_clip=(0.0, 1.0),
        logger=exp_logger,
        log_every=1,
        checkpoint_every=50,
    )

    resolved_seed_iter: int | None = None
    if seed_iter is not None and seed_iter not in ("latest", "auto", ""):
        resolved_seed_iter = int(seed_iter)

    key, run_key = jax.random.split(key)
    final = opt.run(
        key=run_key,
        seed_from=seed_from,
        seed_iter=resolved_seed_iter,
        resume_from=resume_from,
    )

    # Final profile plot against the target Gaussian
    phasor_field = final.arrays.detector_states[phasor_detector.name]["phasor"][0]
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
        save_path=exp_logger.cwd / "figures" / f"downward_profile_final.png",
    )


if __name__ == "__main__":
    parser = fdtdx.build_arg_parser(description="Downward Gaussian emitter optimization")
    # Backwards-compat: allow positional-style args like the old CLI.
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        argv = ["--seed-rng", sys.argv[1]]
        if len(sys.argv) > 2:
            if sys.argv[2].lower() in ("true", "1", "eval"):
                argv.append("--evaluation")
        if len(sys.argv) > 3 and sys.argv[3].lower() in ("true", "1"):
            argv.append("--backward")
        if len(sys.argv) > 4:
            argv += ["--seed-from", sys.argv[4]]
        if len(sys.argv) > 5:
            argv += ["--seed-iter", sys.argv[5]]
        args = parser.parse_args(argv)
    else:
        args = parser.parse_args()
    main(args)
