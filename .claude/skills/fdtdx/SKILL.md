---
name: fdtdx
description: FDTDX framework knowledge — JAX-based FDTD simulation patterns, Yee grid conventions, pytree immutability, constraint system, gradient strategies, and inverse design workflows. Use when writing or modifying fdtdx code.
user-invocable: false
---

# FDTDX Framework Knowledge

## Immutability & PyTree Pattern

Every class inherits from `TreeClass` (wraps `pytreeclass.TreeClass`). Objects are **frozen JAX pytrees** — never mutate in place.

**Always use `.aset()` for updates:**
```python
# Single field
config = config.aset("gradient_config", grad_cfg)

# Nested path
obj = obj.aset("nested->field", value)
```

**Field types** (from `core/jax/pytrees.py`):
- `field()` — standard mutable pytree leaf (KW_ONLY by default)
- `frozen_field()` — excluded from pytree traversal (metadata, not differentiated)
- `private_field()` — not in `__init__`, set after construction
- `frozen_private_field()` — frozen + private combined
- `@autoinit` — decorator that auto-generates `__init__` from type hints

**Key implication:** Since objects are pytrees, they can flow through `jax.jit`, `jax.grad`, `jax.vmap`, etc. Use `frozen_field` for anything that should NOT be traced/differentiated (names, config flags, etc.).

## Simulation Pipeline

The canonical execution order is always:

```python
# 1. Define objects and constraints
volume = fdtdx.SimulationVolume(partial_real_shape=(Lx, Ly, Lz))
source = fdtdx.GaussianPlaneSource(...)
detector = fdtdx.PoyntingFluxDetector(...)
# ... add constraints ...

# 2. Resolve constraints and initialize arrays
objects, arrays, params, config, key = fdtdx.place_objects(
    object_list=object_list,
    config=config,
    constraints=constraints,
    key=key,
)

# 3. Apply device parameters to permittivity arrays
arrays, objects, info = fdtdx.apply_params(arrays, objects, params, key)

# 4. Run simulation
state = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)
time_step, arrays = state

# 5. (Optional) Backward pass for gradient recording
_, arrays = fdtdx.full_backward(state=state, objects=objects, config=config, key=key)
```

**`place_objects()`** resolves all constraints iteratively (up to 1000 iterations), places objects on the grid, initializes E/H/PML field arrays, material arrays, detector states, and recording state.

**`apply_params()`** runs the device parameter transformation pipeline and writes resulting permittivities into the array container. For CONTINUOUS output, uses linear interpolation between materials. For DISCRETE output, uses straight-through estimator (STE).

**`run_fdtd()`** dispatches to either `reversible_fdtd()` or `checkpointed_fdtd()` based on `config.gradient_config`.

## Yee Grid Conventions

Axis mapping: 0=x, 1=y, 2=z. Field arrays have shape `(3, Nx, Ny, Nz)` where index 0 is the component index.

**Staggered field positions (Taflove convention):**
```python
E_x: (i+1/2, j,     k    )     H_x: (i,     j+1/2, k+1/2)
E_y: (i,     j+1/2, k    )     H_y: (i+1/2, j,     k+1/2)
E_z: (i,     j,     k+1/2)     H_z: (i+1/2, j+1/2, k    )
```

**Leapfrog time stepping:** E at integer steps, H at half-steps. Single time step order:
1. Update E fields (curl of H)
2. Update H fields (curl of E)
3. Inject sources
4. Record detectors

**Detector interpolation:** E and H are co-located at the E_z grid point `(i, j, k+1/2)` via multi-point averaging before recording.

## Field Normalization

FDTDX uses **eta0-normalized H fields** — the impedance of free space (eta0 ~ 376.73 Ohm) is absorbed into the field update equations rather than appearing explicitly.

**Update equations (isotropic, lossless):**
```
E^(n+1) = E^n + c * curl(H) * inv_permittivity
H^(n+1/2) = H^(n-1/2) - c * curl(E) * inv_permeability
```
where `c = courant_number = courant_factor / sqrt(3)` (default: 0.99/sqrt(3) ~ 0.571).

**With conductivity (lossy):**
```
factor_E = 1 - c * sigma_E * eta0 * inv_eps / 2
E = factor_E * E + c * curl(H) * inv_eps
E = E / (1 + c * sigma_E * eta0 * inv_eps / 2)

factor_H = 1 - c * sigma_H / eta0 * inv_mu / 2
H = factor_H * H - c * curl(E) * inv_mu
H = H / (1 + c * sigma_H / eta0 * inv_mu / 2)
```

Note the asymmetry: sigma_E multiplied by eta0, sigma_H divided by eta0.

**With dispersion (ADE correction):** After the lossless/lossy E update but before the final divide by `(1 + c*sigma_E*eta0*inv_eps/2)`, add the per-pole polarization increment. For each pole `p`:
```text
P_p^(n+1) = c1_p * P_p^n + c2_p * P_p^(n-1) + c3_p * E^n
E        += inv_eps * sum_p (P_p^n - P_p^(n+1))
```
`P` is stored normalized as `P/eps_0`, so it has the same units as `E` and no eta0 factor enters. The reverse-time update in `update_E_reverse` inverts this recurrence (`c2 ~ -1` in the physical regime keeps the inversion numerically stable).

## Material System

Materials store **inverse** permittivity/permeability (`inv_permittivities`, `inv_permeabilities`) to avoid division in the hot loop.

**Internal representation:** Always 9-tuple `(xx, xy, xz, yx, yy, yz, zx, zy, zz)` for the full 3x3 tensor, but array sizing adapts:
- **Isotropic** (all objects scalar): 1-component arrays
- **Diagonally anisotropic** (any object has 3 components): 3-component arrays
- **Fully anisotropic** (any object has 3x3 tensor): 9-component arrays

This is determined globally — if ANY object is anisotropic, ALL material arrays expand. The detection happens in `ObjectContainer` properties like `all_objects_isotropic_permittivity`.

**Conductivity is scaled by resolution** during initialization in `_init_arrays` (`src/fdtdx/fdtd/initialization.py`).

**Material fields:**
- `permittivity`, `permeability`, `electric_conductivity`, `magnetic_conductivity` — 9-tuples (scalar/3-tuple/nested-3x3 inputs auto-normalized).
- `dispersion: DispersionModel | None` — attaches an ADE dispersion model. When set, `permittivity` is the high-frequency permittivity ε∞ and the full ε(ω) = ε∞ + χ(ω).
- `is_dispersive` property → True iff `dispersion` has at least one pole.

## Dispersive Materials (ADE)

Linear dispersion is implemented via the Auxiliary Differential Equation (ADE) method in `src/fdtdx/dispersion.py`. A `DispersionModel` is a sum of 2nd-order poles, each solving `p̈ + γ ṗ + ω₀² p = K E` for a normalized polarization `p = P/ε₀`.

**Pole classes** (all inherit from `Pole`, stored as `frozen_field` inside `DispersionModel`):
- `LorentzPole(resonance_frequency, damping, delta_epsilon)` — `χ(ω) = Δε·ω₀² / (ω₀² − ω² − iγω)`
- `DrudePole(plasma_frequency, damping)` — `χ(ω) = −ωₚ² / (ω² + iγω)` (special case ω₀ = 0)
- New pole types: subclass `Pole` and expose `omega_0`, `gamma`, `coupling_sq` (K = Δε·ω₀² for Lorentz, ωₚ² for Drude).

**Discrete-time recurrence** (central differences, evaluated once at setup via `compute_pole_coefficients(poles, dt)`):
```text
p^(n+1) = c1·p^n + c2·p^(n-1) + c3·E^n
c1 = (2 − ω₀²·dt²) / (1 + γ·dt/2)
c2 = −(1 − γ·dt/2) / (1 + γ·dt/2)
c3 =  (K·dt²)      / (1 + γ·dt/2)
```
Stability needs `γ·dt < 2`; physically `γ·dt ≪ 1`, so `c2 ≈ −1` and the reverse-time inversion in `update_E_reverse` is well-conditioned.

**ArrayContainer fields** (all `None` unless any object is dispersive):
- `dispersive_P_curr`, `dispersive_P_prev` — shape `(num_poles, 3, Nx, Ny, Nz)`, field-dtype (complex if `use_complex_fields`). Not differentiable (state-only; `None` cotangent in both gradient paths).
- `dispersive_c1`, `dispersive_c2`, `dispersive_c3` — shape `(num_poles, 1, Nx, Ny, Nz)` (middle axis broadcasts over field components). Config dtype. Differentiable: cotangents flow through them in both the reversible and checkpointed paths.
- `dispersive_inv_c2` — cached `1/c2`, closure-captured and `stop_gradient`'d so gradients flow through `c2` only and don't double-count.

**Leading pole axis size:** `objects.max_num_dispersive_poles` — the max pole count across all `UniformMaterialObject`, `Device`, `StaticMultiMaterialObject`. Materials with fewer poles get zero-padded slots, so non-dispersive cells automatically contribute zero. `UniformMaterialObject` always writes the full zero-padded coefficient stack into its `grid_slice`, so a non-dispersive object placed over a dispersive one cleanly clears stale coefficients.

**Restriction:** Dispersive materials cannot currently be combined with fully anisotropic (off-diagonal) permittivity tensors — `_init_arrays` raises `NotImplementedError`. Isotropic and diagonally anisotropic ε are both fine.

**Devices with dispersive materials:** `apply_params` interpolates ADE coefficients the same way it interpolates `inv_permittivities` — linearly between the two bracketing materials for `CONTINUOUS` output, straight-through-estimator for `DISCRETE`. This is not equivalent to interpolating the pole *parameters*, but it keeps gradients smooth for inverse design.

**Evaluating χ(ω) / ε(ω) from stored coefficients** (useful in sources, detectors, setup-time analysis):
- `susceptibility_from_coefficients(c1, c2, c3, omega, dt)` → JAX complex array of per-cell χ(ω), summed over poles.
- `effective_inv_permittivity(inv_eps, c1, c2, c3, omega, dt)` → real 1/Re(ε∞ + χ(ω)); used by sources to sample the true medium at the carrier frequency (imaginary part is already handled by the ADE loop — injecting it would double-count).
- `compute_eps_spectrum_from_coefficients(c1, c2, c3, inv_eps_inf, omegas, dt, weights=None)` → host-side numpy; volume-averaged complex ε(ω) spectrum for a block of cells.
- `compute_impedance_corrected_temporal_profile(raw_samples, dt, eps_spectrum, eps_center)` → applies the FIR filter `G(ω) = √(ε(ω)/ε(ω_c))` to an E-side temporal profile, producing the H-side profile for broadband TFSF injection.

## Constraint System

Objects are positioned relative to each other via constraint objects. Key constraint builders on `SimulationObject`:

```python
# Position relative to another object
obj.place_relative_to(other, axes=(2,), own_positions=("x1",), other_positions=("x2",))
obj.place_at_center(volume)
obj.place_above(other, margin=0.5e-6)
obj.face_to_face_same_side(other, axis=2, position="x2", margin=0.1e-6)

# Size constraints
obj.same_size(volume, axes=(0, 1))
obj.size_relative_to(volume, axes=(2,), proportions=(0.5,))
obj.extend_to(boundary, axis=2, side="x2")

# Grid-level positioning
obj.set_grid_coordinates(axes=(2,), sides=("x1",), coordinates=(10,))
```

**Constraint types:** `PositionConstraint`, `SizeConstraint`, `SizeExtensionConstraint`, `GridCoordinateConstraint`, `RealCoordinateConstraint`.

Objects specify `partial_real_shape` (meters) or `partial_grid_shape` (voxels) with `None` for unconstrained dimensions that will be resolved by constraints.

## Boundary Conditions

**PML (Perfectly Matched Layer):** CPML formulation with polynomial-graded sigma, kappa, alpha profiles. Uses 6 auxiliary psi fields `(psi_Ex, psi_Ey, psi_Ez, psi_Hx, psi_Hy, psi_Hz)` each with shape `(Nx, Ny, Nz)`. PML breaks time-reversal symmetry, so interface fields must be recorded for reversible gradients.

**PEC / PMC:** Zero tangential E (PEC) or H (PMC) at boundary. Applied via `apply_field_reset()`.

**Bloch Boundary:** Phase-shifted periodic conditions with `bloch_vector=(kx, ky, kz)`. When any k component is nonzero, complex fields are required. `PeriodicBoundary` is an alias for `BlochBoundary` with `bloch_vector=(0,0,0)`.

**BoundaryConfig helper:**
```python
bound_cfg = fdtdx.BoundaryConfig.from_uniform_bound(boundary_type="pml", thickness=10)
bound_dict, constraint_list = fdtdx.boundary_objects_from_config(bound_cfg, volume)
```

## Simulation Symmetry

Mirror-symmetry exploitation: build the **full** model, set `config.symmetry`, and fdtdx runs the reduced half/quarter/octant internally (up to 8× less memory/compute), then you unfold results back to the full domain. Implemented in `src/fdtdx/fdtd/symmetry.py`; the FDTD time loop is untouched (the symmetry plane is just an added PEC/PMC wall object).

**Encoding** — `symmetry: tuple[int, int, int]` on `SimulationConfig`, order `(x, y, z)`:
- `0` = no symmetry on this axis
- `-1` = **PEC** mirror (electric wall) on the axis center plane — tangential E odd, normal E even
- `+1` = **PMC** mirror (magnetic wall) — tangential H odd, normal H even

Distinct from manually placing PEC/PMC via `BoundaryConfig` (that still works unchanged); `config.symmetry` is the additive auto-reduce path.

**Requirements / behavior:**
- Each symmetric axis **must resolve to an even cell count** (else `place_objects` raises `ValueError`) — guarantees an exact split and cell-for-cell unfold.
- The **upper half is kept** so the plane lands at the reduced domain's min edge (matching the mode solver's "wall at min edge" convention). Objects are clipped to that half during `place_objects`; centered objects keep their upper half, objects entirely in the discarded half are dropped (with a warning).
- The min-side boundary on each symmetric axis is replaced by the PEC/PMC wall; the far (max) side keeps whatever the user set (use PML there, not periodic).
- `ModePlaneSource` / `ModeOverlapDetector` get their mode-solver `symmetry` 2-tuple **auto-derived** (PMC→1, PEC/none→0 on the two transverse axes) unless explicitly set.
- The user must place objects symmetrically about the center plane — asymmetric models are warned about but not corrected (true of every FDTD symmetry feature).

**Usage:**
```python
config = fdtdx.SimulationConfig(resolution=..., time=..., symmetry=(0, -1, 1))  # PEC y-plane, PMC z-plane
# ... build the FULL volume/sources/detectors/boundaries as usual ...
objects, arrays, params, config, _ = fdtdx.place_objects(...)   # reduced internally
arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key)
_, arrays = fdtdx.run_fdtd(arrays=arrays, objects=objects, config=config, key=key)  # runs on reduced domain

# Unfold to full domain (explicit, post-processing — NOT auto-run by run_fdtd):
full = fdtdx.unfold_detector_states(arrays, objects, config)     # full-domain detector_states
E_full = fdtdx.unfold_fields(arrays.fields.E, config.symmetry, "E")  # (3, Nx, Ny, Nz)
```

**Unfold helpers** (`fdtdx.unfold_fields`, `fdtdx.unfold_detector_states`, `fdtdx.unfold_source_mode`, `fdtdx.unfold_array`):
- `unfold_fields(field, symmetry, field_type)` — reconstruct a full `(3, Nx, Ny, Nz)` E/H array via per-component parity mirror. The general escape hatch — derive any quantity from the full fields.
- `unfold_detector_states(arrays, objects, config)` — pure post-processing that rebuilds each detector's full-domain output from its stored reduced output + parity (no in-loop cost, no flags). Spatial outputs are mirrored per component; `reduce_volume` sums/means are rescaled per component (even doubles/keeps, **odd vanishes**); `as_slices` energy planes are mirrored in-plane.
- `unfold_source_mode(source, config)` → `(E_full, H_full)` — reconstruct the full-domain mode profile a `ModePlaneSource` *injects* (its solved-on-the-reduced-cross-section `_E`/`_H`). Unfolds only the transverse axes (the propagation axis is never a symmetry plane). Run `apply_params` first. For the fields *recorded during the run*, prefer a detector on the source plane + `unfold_detector_states`.
- **Guardrails:** unfolding a non-symmetric model (`symmetry=(0,0,0)`) raises `ValueError`; `place_objects` warns that results are on the reduced domain until unfolded.
- **Not unfoldable:** `DiffractiveDetector` raises `NotImplementedError` (its diffraction-order basis depends on domain size — unfold the fields and recompute instead).
- **Mode-overlap S-params** are already correct on the reduced domain (source + detector share the reduced plane), so they need no unfolding.

**Mode sources are fully wired:** under symmetry, a `ModePlaneSource`'s cross-section is clipped to the reduced grid, its mode-solver `symmetry` 2-tuple is auto-derived from `config.symmetry`, and `compute_mode` solves/injects the half/quarter mode with the matching PEC/PMC wall. Use `unfold_source_mode` to inspect the reconstructed full profile.

**Gradient note:** the differentiable simulation runs on the reduced domain (correct and cheaper); unfolding is a post-hoc step on the output arrays.

## Gradient Strategies

**Reversible FDTD** (`method="reversible"`):
- Exploits time-reversibility of Maxwell's equations
- O(1) field memory, O(T) boundary memory (PML interfaces only)
- Uses `@jax.custom_vjp` — forward pass runs simulation recording boundaries, backward pass reconstructs fields in reverse
- Requires a `Recorder` with optional compression modules (e.g., `DtypeConversion(dtype=jnp.bfloat16)`)
- Differentiable primals: `inv_permittivities`, `inv_permeabilities`, and (when present) `dispersive_c1/c2/c3`. Conductivity arrays and `dispersive_inv_c2` are closure-captured non-primals; `dispersive_P_curr/prev` thread through as state-only primals with `None` cotangent.
- Dispersive reverse update: the ADE recurrence `P^(n+1) = c1·P^n + c2·P^(n-1) + c3·E^n` is algebraically inverted to recover `P^(n-1)` (see `update_E_reverse`). For lossy + dispersive + conductive cells the reverse E update subtracts `inv_eps * sum(P^n − P^(n+1))` before dividing by the loss factor.

**Checkpointed FDTD** (`method="checkpointed"`):
- Standard gradient checkpointing via `eqxi.while_loop(kind="checkpointed")`
- Configurable memory/compute tradeoff via `num_checkpoints`
- Dispersive coefficients flow gradient naturally through the tape.

**Setup pattern:**
```python
recorder = fdtdx.Recorder(modules=[fdtdx.DtypeConversion(dtype=jnp.bfloat16)])
gradient_config = fdtdx.GradientConfig(method="reversible", recorder=recorder)
config = config.aset("gradient_config", gradient_config)
```

## Device & Parameter Transformations

Devices are optimizable regions with a parameter transformation pipeline:

```python
device = fdtdx.Device(
    materials={"air": air, "si": silicon},
    param_transforms=[
        fdtdx.StandardToInversePermittivityRange(),
        fdtdx.GaussianSmoothing2D(sigma=1.0),
        fdtdx.TanhProjection(beta=4.0),
        fdtdx.ClosestIndex(),
    ],
    partial_real_shape=(...),
    partial_voxel_real_shape=(...),  # voxel grid can differ from sim grid
)
```

**Pipeline order:** projection -> smoothing -> discretization -> discrete post-processing -> symmetry.

**Parameter types flow:** `CONTINUOUS` (float values interpolating between materials) -> `DISCRETE` (integer material indices). The STE (straight-through estimator) bridges discrete forward with continuous gradients.

**Voxel indirection:** Devices have their own voxel grid independent of the simulation grid, allowing coarse optimization on a fine simulation mesh.

## Sources

**TFSF (Total-Field/Scattered-Field):** Plane sources inject fields at a boundary offset +0.25 on the Yee grid along the propagation axis.

**Source types:**
- `UniformPlaneSource` — uniform amplitude across plane
- `GaussianPlaneSource` — Gaussian beam profile with configurable `radius`
- `ModePlaneSource` — injects a computed waveguide mode profile
- `PointDipoleSource` — point dipole with configurable `polarization` axis (0/1/2) plus optional `azimuth_angle`/`elevation_angle` (degrees) to tilt off-axis; also `source_type` ∈ {"electric","magnetic"}.

**Temporal profiles:** `SingleFrequencyProfile` (CW) or `GaussianPulseProfile` (pulsed).

**On/Off control:** `OnOffSwitch` pre-computes boolean arrays for the entire simulation duration during `place_on_grid()`.

**`SimulationObject.apply()` signature** — `apply_params` passes dispersive coefficients through to every object:
```python
def apply(self, *, key, inv_permittivities, inv_permeabilities,
          dispersive_c1=None, dispersive_c2=None, dispersive_c3=None): ...
```
Coefficient arrays are passed with `stop_gradient` (matching how `inv_permittivities` is passed to source apply) — the FDTD VJP itself still differentiates through them, this only avoids gradient noise from the source amplitude path. Objects that don't use them (detectors, boundaries, uniform material objects) just `del` the kwargs; sources use them to sample the real medium at the carrier frequency.

**Carrier-frequency impedance in dispersive media:** Sources inside a dispersive background call `effective_inv_permittivity(...)` to get `1/Re(ε∞ + χ(ω_c))` before computing impedance and energy normalization — otherwise they would use only ε∞ and inject with the wrong amplitude ratio. This happens in `LinearlyPolarizedPlaneSource.apply`, `ModePlaneSource.apply`, and `PointDipoleSource.apply`. `PointDipoleSource` additionally uses `_contract_orientation` (einsum over the flattened 9-tensor) so off-diagonal ε coupling is picked up correctly for tilted dipoles. `ModeOverlapDetector.apply` uses the same correction so the reference mode profile is solved against ε(ω_c).

**Broadband TFSF correction** (`_build_dispersive_H_filter` in `src/fdtdx/objects/sources/tfsf.py`): When a source sits in a dispersive medium and its `temporal_profile` is wideband (e.g. `GaussianPulseProfile`), the η(ω_c) rescale alone leaks unphysical reflections at off-carrier frequencies. `TFSFPlaneSource` precomputes a filtered H-side temporal profile `s_H(t)` with spectrum `S(ω)·√(ε(ω)/ε(ω_c))` (stored in `_temporal_H_filter`, shape `(time_steps_total,)`) and looks it up per step via `jnp.interp` at the Yee half-step offset. Non-dispersive case leaves `_temporal_H_filter = None` and the inner loop falls back to the raw `temporal_profile.get_amplitude` call — so non-dispersive behavior is bit-identical.

Bulk ε(ω) is averaged uniformly over the source cells — correct for `LinearlyPolarizedPlaneSource`, a first-order approximation for `ModePlaneSource` (captures bulk dispersion of the guiding medium, not geometric modal dispersion).

## Detectors

All detectors use `OnOffSwitch` for temporal gating. State is stored as `DetectorState = Dict[str, Array]`.

- `FieldDetector` — records raw E/H field components
- `EnergyDetector` — records electromagnetic energy density
- `PoyntingFluxDetector` — records directional power flow (key for transmission/reflection)
- `PhasorDetector` — records complex phasor amplitudes at specific frequencies
- `DiffractiveDetector` — records complex diffraction efficiencies per order
- `ModeOverlapDetector` — computes overlap integral with a guided mode (inherits from PhasorDetector)

**Accessing results:** `arrays.detector_states["name"]["key"]`

All state arrays have a leading time dimension: `(num_time_steps_on, ...)`. Use index `-1` for the final accumulated value.

**FieldDetector** — key: `"fields"`
- `reduce_volume=False`: `(T, num_components, nx, ny, nz)`
- `reduce_volume=True`: `(T, num_components)`

**EnergyDetector** — key: `"energy"` or slice keys
- `as_slices=False, reduce_volume=False`: `(T, nx, ny, nz)`
- `as_slices=False, reduce_volume=True`: `(T, 1)` (scalar)
- `as_slices=True`: three keys `"XY Plane"` `(T, nx, ny)`, `"XZ Plane"` `(T, nx, nz)`, `"YZ Plane"` `(T, ny, nz)` — cannot combine with `reduce_volume=True`

**PoyntingFluxDetector** — key: `"poynting_flux"`
- Default (`reduce_volume=True`, scalar): `(T, 1)` — total flux through surface
- `keep_all_components=True, reduce_volume=True`: `(T, 3)`
- `reduce_volume=False`: `(T, nx, ny, nz)` or `(T, 3, nx, ny, nz)` with `keep_all_components`

**PhasorDetector** — key: `"phasor"`, dtype: complex
- Time dim is always 1 (frequency-domain accumulation)
- `reduce_volume=False`: `(1, num_wavelengths, num_components, nx, ny, nz)`
- `reduce_volume=True`: `(1, num_wavelengths, num_components)`
- Component index matches order of the `components` tuple

**DiffractiveDetector** — key: `"diffractive"`, dtype: complex
- Time dim is always 1
- Shape: `(1, num_frequencies, num_orders)`

**ModeOverlapDetector** — inherits PhasorDetector, always uses all 6 field components. Use `compute_overlap_to_mode()` to get the scalar overlap. In a dispersive medium, the reference mode is solved against `effective_inv_permittivity` at the detector's carrier frequency (same correction as `ModePlaneSource`), so the overlap compares against ε(ω_c) rather than ε∞.

## Optimization Suite

`fdtdx.optimization` (re-exported from `fdtdx`) is the inverse-design driver. Three core abstractions:

- **`LossTerm`** — a `TreeClass` contributing `sign * schedule(epoch) * raw` to the loss. `compute(*, params, objects, arrays, config, epoch) → (scalar, info_dict)`. `__call__` wraps the raw metric with the schedule and auto-injects `{name}_raw`, `{name}_weight`, `{name}_contrib` into the info dict.
- **`Objective`** (subclass of LossTerm) — driver calls with `sign=-1`. Use for quantities to MAXIMISE (efficiency, overlap).
- **`Constraint`** (subclass of LossTerm) — driver calls with `sign=+1`. Use for penalties to MINIMISE (fab violations, reflections).

Every term takes `name: str` (frozen, used as info-dict prefix) and `schedule: WeightSchedule` (default `ConstantSchedule(value=1.0)`).

### Weight Schedules (`fdtdx.optimization.schedules`)

All JIT-safe (no Python branching on traced epoch). Shared fields: `epoch_start` (default 0), `epoch_end` (default `None` = unbounded). Returns `0.0` outside `[epoch_start, epoch_end]`.

- **`ConstantSchedule(value=1.0)`** — constant inside window.
- **`LinearSchedule(start_value=0.0, end_value=1.0)`** — linear ramp.
- **`ExponentialSchedule(start_value=1e-3, end_value=1.0)`** — log-linear; both bounds must be > 0.
- **`CosineSchedule(start_value=0.0, end_value=1.0)`** — cosine-eased with zero derivative at endpoints.
- **`OnOffSchedule(value=1.0)`** — binary on/off (semantically == Constant, documents hard-switch intent).

**Default fab-rule / connectivity constraints to a ramp AND delay the ramp start until β has hardened** — not `epoch_start=0`. DRC ("feature thinner than X") and connectivity ("is it one piece") are *meaningless on the initial gray ρ≈0.5 field* and, applied from epoch 0, dominate the gradient so the optimizer never finds a guiding/physics-satisfying structure. Use e.g. `LinearSchedule(epoch_start=round(0.4*epochs), epoch_end=round(0.95*epochs), start_value=0.0, end_value=1.0)` so they switch on only after the projection β ramp has substantially binarized the design. Physics objectives/constraints (flux, overlap, phase, back-reflection) stay `ConstantSchedule` from epoch 0. Always sanity-check each term's raw magnitude in `metrics.csv` (`{name}_raw`) and size `end_value` so its `{name}_contrib` is comparable to the primary objective — a term whose raw is orders of magnitude larger silently becomes the only thing optimized.

### Wrappers for One-Off Terms

- **`FunctionObjective(name=..., schedule=..., fn=callable)`** — wraps a user callable `fn(*, params, objects, arrays, config, epoch) → (scalar, info_dict)`. No subclassing needed.
- **`FunctionConstraint(...)`** — same, contributes with `sign=+1`.

### Built-In Manufacturing Constraints

All evaluate the device density via the shared helper `_eval_device_density(device, params[name])` (in `constraints/physics.py`), which calls `device(leaf, beta=_HARDEN_BETA)` (`_HARDEN_BETA = 50.0`) and falls back to a no-kwarg call on `TypeError`. This is required because a device whose param pipeline contains a β-projection (`TanhProjection`, `SubpixelSmoothedProjection`) raises if called without `beta` — so DRC/connectivity/OPC constraints (and the `Optimization` driver's `logger.log_params`) all evaluate the *as-fabricated* near-binary design at a hardened β, not the soft latent. They use morphology primitives from `fdtdx.optimization.utils.morphology`; device's `single_voxel_real_shape[0]` gives the XY pitch used to translate meters → odd voxel kernel.

- **`MinLineSpace(device_name, min_line_width_m, min_space_m, beta=8.0, eta_erode=0.75, eta_dilate=0.25)`** — Sigmund-Wang filter-and-project. Penalises solid features thinner than `min_line_width_m` and gaps narrower than `min_space_m`: `mean(thin_line²) + mean(thin_gap²)`. Info keys: `thin_line`, `thin_gap`.
- **`MinInclusion(inner_device_name, outer_device_name, min_margin_m, beta=8.0, eta=0.75)`** — inner device must lie inside outer device eroded by `min_margin_m`. Last 2 axes (XY) must match; Z broadcasts. Info: `max_violation`, `mean_violation`.
- **`NoFloatingMaterial(device_stack_names: tuple[str, ...])`** — for a bottom→top stack of etched layers, penalises `mean(ReLU(ρ_above - ρ_below)²)` summed over adjacent pairs. All layers must share shape. Info: `max_excess`, `penalty`. Use this for multi-etch-level devices where upper layers need support below.

### Connectivity

- **`VirtualTemperatureConnectivity(device_name, source_mask, drain_mask, kappa_min=1e-3, kappa_max=1.0, p=3.0, cg_iterations=200, cg_tol=1e-6)`** — solves `-∇·(κ(ρ) ∇T) = source_mask` with `T=0` on `drain_mask` via CG with Jacobi preconditioner. `κ(ρ) = κ_min + (κ_max-κ_min)·ρᵖ` (SIMP). Masks must match `device.ndim` or be 2D/3D (stencil auto-selected). The penalty is the mean source-cell temperature **normalized by the fully-disconnected baseline** (the same solve with κ ≡ κ_min, which is ρ-independent → solved once internally and `stop_gradient`'d). So the raw penalty is intrinsically **~[0,1]**: ≈1 fully disconnected, →0 once ρ forms a continuous source→drain path. Use a normal **O(1) weight** (no tiny magic scale). Info keys: `{name}_penalty_norm` (the returned scalar), `{name}_disconnect_ref`, `{name}_temp_mean_source`, `{name}_temp_max`. (Pre-normalization the raw was an un-normalized ~1e3–1e4 mean temperature that silently dominated the loss — don't reintroduce a hand-tuned weight to compensate.)

### Lithography / OPC

- **`LithographyModel(wavelength_m=193e-9, numerical_aperture=0.85, sigma_inner=0.0, sigma_outer=0.7, sigma_transition=0.0, resist_threshold=0.3, resist_sharpness=50.0, num_kernels=20, source_grid_points=41)`** — Hopkins partially-coherent imaging with SOCS compression (top-K eigenpairs of the TCC) + sigmoid resist. Two-stage life cycle:
  1. Config-only until `.prepare(grid_shape, voxel_pitch_m)` → new model with frozen kernel/eigenvalue leaves. Eigendecomposition runs on numpy at prepare time; fails if voxel pitch is too coarse for NA/λ.
  2. Prepared model supplies `aerial_image(design) → intensity` and `forward(design) → (printed, aerial)` (printed = sigmoid(sharpness·(aerial - threshold))). Broadcasts over leading axes; last 2 axes are spatial.
- **`OPCConstraint(device_name, litho_model, target_design=None)`** — `mean((printed - target)²)` where target is `ρ` itself (self-consistency) when `target_design is None`, else the supplied array. Use `OPCConstraint.for_device(device, litho_model, name=..., target_design=...)` to prepare the model on the device's XY grid automatically. Info: `aerial_max`, `aerial_mean`, `mismatch`.

### Physics-Base for Custom Constraints

- **`PhysicsConstraint`** (abstract) — fetches density via `device_name`, clips to `[0, 1]`, squeezes trailing singletons, hands `rho` to `build_penalty(rho) → (scalar, info)`. Extend for density-only constraints.
- **`LinearSteadyStatePDEConstraint`** (abstract) — solves `A(ρ) u = b(ρ)` with implicit-diff CG (gradients flow through the CG solve via JAX's built-in VJP). Subclasses override `operator(u, rho)` (must be symmetric), `rhs(rho)`, `preconditioner_diag(rho) | None`, and `penalty(u, rho) → (scalar, info)`. Fields: `cg_iterations=200`, `cg_tol=1e-6`. `VirtualTemperatureConnectivity` is the reference example.

### Slow-Physics / Multiphysics Co-simulation

Electrostatics, linear elasticity, modal analysis, pull-in continuation, and the EOM coupling **are no longer part of fdtdx**. They live in the sibling package **`fdtdx-multiphysics`** (import as `fdtdx_multiphysics`, conventional alias `mp`). Use that package for any quasi-static physics that rides alongside the FDTD: `mp.PoissonSolver`, `mp.ElasticityEigenmodes`, `mp.ElectroMechanicalFEM`, `mp.pull_in_voltage`, `mp.advect_density`, `mp.apply_displacement_to_permittivity`, `mp.apply_finite_displacement_to_permittivity`, etc. fdtdx exposes only the EM building blocks they consume (the constants module, the `Device` / `apply_params` / `run_fdtd` machinery, `ArrayContainer` for threading multiphysics outputs through the optimisation pytree).

See the `fdtdx-multiphysics` skill for the slow-physics API surface, and the `lightsail-pdk` `MEMSPhaseTunerCoupling` for an example orchestration pattern (CPU FEM run **outside** the JIT in a `pre_step_hook`, results threaded back into `arrays.detector_states`, ε-perturbation applied in a pure-JAX `simulate_fn`).

**fdtdx-side pipeline gotchas that remain relevant:** (1) `simulate_fn`'s returned `ArrayContainer` must have the **same detector_states pytree every epoch** — multi-voltage / multi-state schemes that add suffixed `detector_states` keys (or stash multiphysics outputs there) must pre-register them with zero arrays in the initial `arrays` *before* constructing `Optimization`, or the driver's compiled `loss_fn` fails on the threaded-back input. (2) `config.time` must let light traverse the *whole* device plus settling, or output detectors read a tiny perturbation-insensitive precursor. (3) `ModeOverlapDetector.compute_overlap` needs `.apply(...)` called once at setup to populate the reference mode (the driver passes pre-`apply` objects to metric callbacks); close the metric over a setup-prepared detector.

### `Optimization` Driver

```python
opt = fdtdx.Optimization(
    objects=objects,            # frozen_field — structural, not traced
    arrays=arrays,              # frozen_field — initial state (JIT-donation-safe)
    params=params,              # field — traced, primary gradient target
    config=config,              # frozen_field
    simulate_fn=simulate_fn,    # (params, arrays, objects, config, key, epoch) → arrays
    optimizer=optax.adam(...),  # any optax.GradientTransformation
    objectives=(...,),          # tuple of Objective
    constraints=(...,),         # tuple of Constraint
    total_epochs=500,
    param_clip=(0.0, 1.0),      # applied after every optax update
    logger=exp_logger,          # optional fdtdx.Logger
    log_every=1,
    checkpoint_every=50,
    checkpoint_dir=None,        # defaults to {logger.cwd}/checkpoints
    pre_step_hook=None,         # optional out-of-trace hook (see below)
    epoch_callback=None,        # optional per-logged-epoch monitor
)
final = opt.run(key=key, seed_from=..., seed_iter=..., resume_from=...)
```

- `simulate_fn` is where you call `fdtdx.apply_params(..., beta=beta_schedule(epoch))` and `fdtdx.run_fdtd(...)`. The driver doesn't hard-code those — users control beta, custom backward passes, etc.
- `loss_fn` (internal) is wrapped in `jax.value_and_grad` then `jax.jit`-compiled with `donate_argnames=["arrays"]`. **Compilation is deferred to the first epoch call** (epoch 0 bears the latency). A previous `jit_step.lower(...).compile()` pre-compile triggered `Computation compiled for X inputs but called with Y` on JAX ≥ 0.9 because the closure-captured `ObjectContainer` / `SimulationConfig` flattened into many `const_args` at lower time that the runtime `Compiled.call` could not re-thread.
- `seed_from` loads params only (resets optax state + epoch). `resume_from` restores params + optax state + epoch + RNG. They are mutually exclusive.
- Returns a new `Optimization` with updated `params` and `arrays`.

**`pre_step_hook(*, epoch, params, arrays, objects, optimization) → updated_arrays | None`** runs *before* each compiled step, with full Python freedom (host-side numpy/scipy work, file I/O, …). Use it to thread **out-of-trace** computations into the JIT region by mutating `arrays.detector_states` and returning the updated container. Canonical use case: CPU FEM solves from `fdtdx_multiphysics` — run `ElectroMechanicalFEM.continuation(rho)` on the host, pack `V_pi`/`v_branch`/`s_grid`/`lin_dir` into a `detector_states` slot, return `arrays.aset("detector_states", new_ds)`. Returning `None` keeps arrays unchanged (e.g. when a host transfer fails and you want to reuse last epoch's state). **Exceptions are NOT caught** — a broken pre-step is a hard error because it can desynchronise the gradient. The keys it writes must be pre-seeded with zero arrays in the initial `arrays` before constructing `Optimization`, else the first epoch fails on pytree mismatch.

**`epoch_callback(*, epoch, params, objects, arrays, info, optimization)`** fires every `log_every` epochs *after* logging, for live monitoring / status plots / early-stop bookkeeping. Exceptions inside it are caught and logged — a broken monitor must never kill a long optimization. Use it for matplotlib snapshots, push-metrics, custom convergence checks, etc.

### Utils

- **`fdtdx.build_arg_parser(description=...)`** → pre-populated `argparse.ArgumentParser` with `--seed-rng`, `--evaluation`, `--backward`, `--seed-from PATH`, `--seed-iter IDX|latest`, `--resume-from PATH`. Extend with script-specific flags before `parse_args()`.
- **`fdtdx.save_checkpoint(dir, *, epoch, params, opt_state, rng_key)`** / **`load_checkpoint(path, *, params_template, opt_state_template, rng_key_template=None)`** — full resume; writes `checkpoint_{epoch:06d}.eqx` + `.json`. `load_checkpoint` accepts either a dir (auto-picks latest) or a direct `.eqx` file.
- **`fdtdx.load_seed_params(seed_path, params_template, iter_idx=None)`** — reads `params_{iter}_{device}.npy` from a Logger output dir. Auto-selects the highest iter for which every device in `params_template` has a file. Values are replaced; shapes/dtypes must match the template.
- **Morphology** (`fdtdx.optimization.utils.morphology`, also re-exported): `box_filter_2d`, `gaussian_filter_2d`, `smooth_erosion(rho, k, beta=8.0, eta=0.75)`, `smooth_dilation(rho, k, beta=8.0, eta=0.25)`, `meters_to_odd_kernel(length_m, voxel_pitch_m)` (clamps to ≥3, rounds up to odd). All morphology ops act on the last two axes and broadcast over leading batch axes.

### Typical Script Flow

```python
parser = fdtdx.build_arg_parser()
args = parser.parse_args()

# ... construct objects, place_objects, define simulate_fn ...

objectives = (
    fdtdx.FunctionObjective(name="flux_eff", fn=flux_metric,
                            schedule=fdtdx.ConstantSchedule(0.5)),
    fdtdx.FunctionObjective(name="overlap", fn=overlap_metric,
                            schedule=fdtdx.ConstantSchedule(0.5)),
)
constraints = (
    fdtdx.MinLineSpace(name="ls", device_name="dev",
                       min_line_width_m=140e-9, min_space_m=140e-9,
                       schedule=fdtdx.LinearSchedule(
                           epoch_start=0, epoch_end=round(0.9*epochs),
                           start_value=0.0, end_value=1.0)),
)

opt = fdtdx.Optimization(..., objectives=objectives, constraints=constraints)
final = opt.run(key=key, seed_from=args.seed_from, resume_from=args.resume_from)
```

## Testing Patterns

**Three test tiers** (auto-marked via conftest.py):
- `unit` — individual components, no simulation runs
- `integration` — object placement, initialization, multi-component interaction
- `simulation` — full FDTD runs validating physics

**Physics validation pattern (two-run normalization):**
```python
# Reference run (e.g., all PML) and test run (e.g., with PEC) share a helper:
def _run(objects, constraints, config):
    key = jax.random.PRNGKey(0)
    obj_container, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key,
    )
    arrays, obj_container, _ = fdtdx.apply_params(arrays, obj_container, params, key)
    _, arrays = fdtdx.run_fdtd(arrays=arrays, objects=obj_container, config=config, key=key)
    return arrays

def _mean_flux(arrays, name):
    flux = np.array(arrays.detector_states[name]["poynting_flux"][:, 0])
    return float(np.mean(flux[-N_AVG_STEPS:]))

ref_flux = _mean_flux(_run(ref_objects, ref_constraints, config), "detector")
test_flux = _mean_flux(_run(test_objects, test_constraints, config), "detector")
transmission = test_flux / ref_flux
```

**Steady-state extraction:** Average over last N optical periods:
```python
steps_per_period = int(round(wavelength / (c0 * dt)))
n_avg = 10 * steps_per_period
steady_state = float(np.mean(flux[-n_avg:]))
```

**Gradient validation:**
```python
loss, grads = jax.value_and_grad(loss_fn)(params, arrays, objects, config, key)
assert jnp.isfinite(loss)
assert jnp.all(jnp.isfinite(grads))
```

**When simulation tests fail marginally:** Increase resolution (more cells per wavelength) rather than relaxing tolerances — the physics should converge, not the assertions weaken.

## Common Pitfalls

- **Forgetting `.aset()`**: Direct attribute assignment on TreeClass objects silently fails or raises. Always use `.aset()`.
- **Material array sizing is global**: Adding one anisotropic object forces ALL material arrays to expand. Check `ObjectContainer` isotropy properties.
- **PML + reversible gradients**: PML breaks time-reversal. Must set up `Recorder` and `recording_state` for boundary interfaces.
- **Complex fields**: Bloch boundaries with nonzero k-vector automatically require complex fields. Check `config.use_complex_fields`. When complex fields are in effect, ADE polarization arrays (`dispersive_P_curr/prev`) are also allocated as complex.
- **Conductivity scaling**: Conductivity values are multiplied by `config.resolution` during `_init_arrays()`. Don't pre-scale.
- **Inverse storage**: Material arrays store `1/epsilon` and `1/mu`, not epsilon and mu directly. For dispersive materials, `Material.permittivity` represents ε∞ only — the full ε(ω) must be reconstructed via the dispersion model.
- **Detector timing**: Detectors only record at timesteps where their `OnOffSwitch` is active. Check `switch` configuration if data appears missing.
- **donate_argnames**: When JIT-compiling simulation functions, use `donate_argnames=["arrays"]` to allow JAX to reuse array memory.
- **Dispersive + full anisotropic**: Not supported — `_init_arrays` raises `NotImplementedError`. Use diagonal anisotropy if you need directional ε alongside dispersion.
- **Dispersive pole count is max'd globally**: The `num_poles` leading axis size = `objects.max_num_dispersive_poles`. Adding one 3-pole material allocates 3 pole slots for every dispersive cell in the sim; non-dispersive cells still have their `c1/c2/c3` set to zero (ADE term vanishes) but consume array memory.
- **Dispersive source impedance**: Inside a dispersive medium, never use ε∞ as the source's effective permittivity — call `effective_inv_permittivity` at ω_c. Broadband pulses additionally need the `_temporal_H_filter` path to avoid TFSF leakage at off-carrier frequencies.
- **Stacking objects with mixed dispersion**: `UniformMaterialObject` always writes a full zero-padded pole-coefficient stack into its `grid_slice`, so placing a non-dispersive object over a dispersive one cleanly overwrites stale coefficients. Rely on this rather than assuming "no dispersion = leave coefficients alone".
- **WeightSchedule is zero outside `[epoch_start, epoch_end]`** — not just capped. If you use `LinearSchedule` as a generic epoch→value utility (e.g. for beta projection), remember that `schedule(epoch)` returns `0.0` for `epoch > epoch_end`, not the `end_value`. For an unbounded hold past the ramp, leave `epoch_end=None`.
- **Don't pass `fdtdx.apply_params` beta into `Optimization`** — the driver doesn't know about beta. Plumb beta through your `simulate_fn(params, arrays, objects, config, key, epoch)` callback: `arrays, objects, _ = fdtdx.apply_params(arrays, objects, params, key, beta=beta_schedule(epoch))`.
- **`LithographyModel` must be `.prepare(...)`d before use** — an un-prepared model has `kernels=None`; calling `aerial_image`/`forward` will crash. Prefer `OPCConstraint.for_device(device, model, ...)` which prepares automatically on the device's XY grid.
- **`Optimization.arrays` is `frozen_field`** — the initial reference stored on `self` is excluded from the pytree so JIT-donation inside `run()` doesn't break pytreeclass's copy-based `.aset` on the final return. Don't try to trace through `opt.arrays`; treat it as the initial state only.
- **`seed_from` vs `resume_from`** are mutually exclusive. `seed_from` loads params only (optax state and epoch reset to 0); `resume_from` restores params + optax + epoch + RNG.
- **Built-in constraints evaluate the device at a hardened β** (`_eval_device_density`, β=50): a device with a `TanhProjection`/`SubpixelSmoothedProjection` pipeline raises if called without `beta`. This is intentional (DRC/connectivity/OPC judge the as-fabricated near-binary design) — don't "fix" it by stripping the projection; if you add a new density-reading constraint, route through `_eval_device_density`, don't call `device(params[name])` directly.
- **`simulate_fn` must return a structurally identical `ArrayContainer` every epoch** — the driver compiles `loss_fn` once and threads the output back as the next input. Multi-pass schemes that add suffixed `detector_states` keys (or stash multiphysics outputs there) must pre-seed those keys (zeros) into the initial `arrays` *before* building `Optimization`, else epoch 1 fails with a pytree mismatch.
- **`VirtualTemperatureConnectivity` penalty is normalized ~[0,1]** (by the disconnected baseline) — use an O(1) weight.
- **Multiphysics symbols are not in fdtdx anymore.** If a script does `fdtdx.PoissonSolver`, `fdtdx.ElasticityEigenmodes`, `fdtdx.pull_in_voltage`, `fdtdx.apply_displacement_to_permittivity`, etc., that's pre-split code — install `fdtdx-multiphysics` and import from `fdtdx_multiphysics` instead.
- **Symmetry results look wrong / are half-size**: with `config.symmetry` set, `run_fdtd` returns *reduced-domain* arrays — you must call `fdtdx.unfold_detector_states` / `fdtdx.unfold_fields` to get full-domain results (see Simulation Symmetry). `place_objects` warns about this. Unfolding a non-symmetric model raises.
- **Symmetry needs even cells + symmetric model**: each symmetric axis must resolve to an even cell count (`place_objects` raises otherwise), and the user's full model must actually be mirror-symmetric about the center plane — asymmetric objects are only warned about. Use PML (not periodic) on the far side of a symmetric axis.
