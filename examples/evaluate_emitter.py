"""
Evaluate an optimized emitter design with an extended downward region.

This script loads saved parameters from an optimization run of optimize_emitter.py
and re-simulates the device in a taller simulation volume so the downward-emitted
Gaussian beam can propagate to its focal point.

Usage:
    python examples/evaluate_emitter.py <params_dir> [seed]

    params_dir: path to the device/params/ directory from an optimization run
                e.g. outputs/nobackup/2026-04-06/antenna_emitter/12-00-00.000000/device/params
    seed:       random seed (default 0)
"""

import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import pytreeclass as tc
from loguru import logger

import fdtdx

# ---------------------------------------------------------------------------
# Material constants (same as optimize_emitter.py)
# ---------------------------------------------------------------------------
PERMITTIVITY_SI = fdtdx.constants.relative_permittivity_silicon
PERMITTIVITY_SIO2 = fdtdx.constants.relative_permittivity_silica
PERMITTIVITY_SIN = 4.0
PERMITTIVITY_AIR = fdtdx.constants.relative_permittivity_air


def main(params_dir: str, seed: int = 0):
    logger.info(f"{seed=}")
    logger.info(f"Loading params from: {params_dir}")
    params_path = Path(params_dir)

    exp_logger = fdtdx.Logger(
        experiment_name="antenna_emitter_eval",
        name=None,
    )
    key = jax.random.PRNGKey(seed=seed)

    # ------------------------------------------------------------------
    # Simulation parameters
    # ------------------------------------------------------------------
    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)
    resolution = 25e-9

    config = fdtdx.SimulationConfig(
        time=300e-15,  # longer run time so the beam reaches the focus
        resolution=resolution,
        dtype=jnp.float32,
        courant_factor=0.99,
    )

    period_steps = round(period / config.time_step_duration)
    logger.info(f"{config.time_steps_total=}")
    logger.info(f"{period_steps=}")

    placement_constraints, object_list = [], []

    # ------------------------------------------------------------------
    # Layer thicknesses — extended downward to show the focal point
    # ------------------------------------------------------------------
    t_si = 220e-9
    t_spacer = 150e-9
    t_sin = 400e-9
    t_topclad = 750e-9

    # Original BOX was 1 μm. The focal point is ~10 μm below the Si layer.
    # Extend the buried oxide so the focus is visible inside the simulation.
    focal_distance_from_si = 10.0e-6
    t_box = focal_distance_from_si + 2.0e-6  # extra 2 μm margin below focus

    t_oxide_total = t_box + t_si + t_spacer + t_sin + t_topclad
    total_z = t_oxide_total

    # Device / emitter dimensions (must match optimize_emitter.py)
    device_length_x = 4.0e-6
    device_width_y = 4.0e-6
    waveguide_width = 300e-9
    waveguide_margin_x = 3.0e-6
    n_si_etch_levels = 3

    total_x = device_length_x + 2 * waveguide_margin_x
    total_y = device_width_y / 2 + 4.0e-6

    # ------------------------------------------------------------------
    # Simulation volume
    # ------------------------------------------------------------------
    volume = fdtdx.SimulationVolume(
        partial_real_shape=(total_x, total_y, total_z),
    )
    object_list.append(volume)

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
    # Layer stack
    # ------------------------------------------------------------------
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
    # Optimizable devices (frozen — params loaded from file)
    # ------------------------------------------------------------------
    voxel_size = resolution

    si_device_materials = {
        "SiO2": mat_sio2,
        "Silicon": mat_si,
    }
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
                fdtdx.TanhProjection(),
                fdtdx.StandardToInversePermittivityRange(),
            ],
            partial_voxel_real_shape=(voxel_size, voxel_size, t_layer),
        )
        placement_constraints.extend([
            layer.place_relative_to(
                oxide_stack, axes=2, own_positions=-1, other_positions=-1,
                margins=t_box + z_offset,
            ),
            layer.place_at_center(volume, axes=0),
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
        sin_device.place_at_center(volume, axes=0),
        sin_device.place_relative_to(volume, axes=1, own_positions=-1, other_positions=-1),
    ])
    object_list.append(sin_device)

    # ------------------------------------------------------------------
    # Waveguides
    # ------------------------------------------------------------------
    waveguide_in = fdtdx.UniformMaterialObject(
        name="Waveguide_in",
        partial_real_shape=(waveguide_margin_x, waveguide_width / 2, t_si),
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

    waveguide_out = fdtdx.UniformMaterialObject(
        name="Waveguide_out",
        partial_real_shape=(waveguide_margin_x, waveguide_width / 2, t_si),
        material=mat_si,
        color=fdtdx.colors.XKCD_LIGHT_BLUE,
    )
    placement_constraints.extend([
        waveguide_out.place_relative_to(volume, axes=1, own_positions=-1, other_positions=-1),
        waveguide_out.place_relative_to(si_layers[0], axes=0, own_positions=-1, other_positions=1),
        waveguide_out.place_relative_to(
            oxide_stack, axes=2, own_positions=-1, other_positions=-1,
            margins=t_box,
        ),
    ])
    object_list.append(waveguide_out)

    # ------------------------------------------------------------------
    # Mode source
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
    # Energy detector — video for the full volume (shows beam propagation to focus)
    video_detector = fdtdx.EnergyDetector(
        name="video",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(interval=10),
        exact_interpolation=True,
        num_video_workers=10,
    )
    placement_constraints.extend([*video_detector.same_position_and_size(volume)])
    object_list.append(video_detector)

    # Energy at last step for static figure
    energy_last_step = fdtdx.EnergyDetector(
        name="energy_last_step",
        as_slices=True,
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[-1]),
    )
    placement_constraints.extend([*energy_last_step.same_position_and_size(volume)])
    object_list.append(energy_last_step)

    exclude_object_list = [energy_last_step, video_detector]

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

    logger.info(tc.tree_summary(arrays, depth=2))
    print(tc.tree_diagram(config, depth=4))

    # ------------------------------------------------------------------
    # Load optimized parameters
    # ------------------------------------------------------------------
    # Find the highest epoch saved for each device
    device_names = [f"Si_Layer_{i}" for i in range(n_si_etch_levels)] + ["SiN_Device"]
    for dname in device_names:
        # Find all param files for this device and pick the latest epoch
        param_files = sorted(params_path.glob(f"params_*_{dname}.npy"))
        if not param_files:
            logger.error(f"No saved params found for {dname} in {params_path}")
            sys.exit(1)
        latest = param_files[-1]
        logger.info(f"Loading {dname} from {latest.name}")
        params[dname] = jnp.load(str(latest))

    # Apply with high beta to get the binarized design
    beta = 200.0

    # ------------------------------------------------------------------
    # Save setup figure
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
        iter_idx=0,
        params=params,
        objects=objects,
        export_stl=True,
        export_figure=True,
        beta=beta,
    )

    # ------------------------------------------------------------------
    # Run simulation
    # ------------------------------------------------------------------
    def forward(params, arrays, key):
        arrays, new_objects, info = fdtdx.apply_params(
            arrays, objects, params, key, beta=beta
        )
        final_state = fdtdx.run_fdtd(
            arrays=arrays,
            objects=new_objects,
            config=config,
            key=key,
        )
        _, arrays = final_state
        return arrays, info

    compile_start_time = time.time()
    print("Started Compilation...")
    jit_task_id = exp_logger.progress.add_task("JIT", total=None)
    jitted_forward = (
        jax.jit(forward, donate_argnames=["arrays"])
        .lower(params, arrays, key)
        .compile()
    )
    compile_delta_time = time.time() - compile_start_time
    exp_logger.progress.update(jit_task_id, total=1, completed=1, refresh=True)
    print(f"Finished Compilation in {compile_delta_time:.1f} seconds")

    run_start_time = time.time()
    key, subkey = jax.random.split(key)
    arrays, info = jitted_forward(params, arrays, subkey)
    runtime_delta = time.time() - run_start_time

    logger.info(f"{compile_delta_time=:.1f}s")
    logger.info(f"{runtime_delta=:.1f}s")

    exp_logger.log_detectors(
        iter_idx=0, objects=objects, detector_states=arrays.detector_states
    )

    info["runtime"] = runtime_delta
    exp_logger.write(info)

    logger.info(f"Output saved to: {exp_logger.cwd}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python examples/evaluate_emitter.py <params_dir> [seed]")
        print("  params_dir: path to device/params/ from an optimization run")
        sys.exit(1)

    params_dir = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    main(params_dir=params_dir, seed=seed)
