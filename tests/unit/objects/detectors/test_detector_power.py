"""Unit tests for the detector power methods (flux_spectrum / transmission), synthetic, no FDTD run."""

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.config import SimulationConfig
from fdtdx.core.grid import UniformGrid
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.objects.detectors.phasor import PhasorDetector


def _placed_plane_detector():
    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = PhasorDetector(
        name="plane", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=False, scaling_mode="pulse"
    )
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    return det, config


def _arrays_with_phasor(det_name, phasor):
    return types.SimpleNamespace(detector_states={det_name: {"phasor": phasor}})


def test_flux_spectrum_known_plane_wave():
    """Ex=1, Hy=1 (forward +z plane wave) -> S_z = 0.5 per cell, summed over the plane."""
    det, _ = _placed_plane_detector()
    # phasor shape (1, num_freqs=1, 6, 4, 4, 1); set Ex (idx0) and Hy (idx4) to 1.
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64)
    phasor = phasor.at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)
    arrays = _arrays_with_phasor("plane", phasor)
    flux = det.flux_spectrum(arrays)
    spacing = 2e-7
    expected = 0.5 * (spacing**2) * (4 * 4)  # 0.5 * area_per_cell * num_cells
    # abs=0: these powers are ~3e-13, below pytest.approx's default 1e-12 absolute floor,
    # which would otherwise make the comparison vacuous (even a zero flux would pass).
    assert float(flux[0]) == pytest.approx(expected, rel=1e-5, abs=0)


def test_flux_spectrum_requires_six_components():
    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = PhasorDetector(
        name="p", wave_characters=(WaveCharacter(wavelength=1e-6),), components=("Ex",), reduce_volume=False
    )
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    arrays = _arrays_with_phasor("p", jnp.zeros((1, 1, 1, 4, 4, 1), dtype=jnp.complex64))
    with pytest.raises(ValueError, match="6 components"):
        det.flux_spectrum(arrays)


def test_transmission_divides_flux_by_injected():
    det, _ = _placed_plane_detector()
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64)
    phasor = phasor.at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)
    arrays = _arrays_with_phasor("plane", phasor)
    flux = det.flux_spectrum(arrays)
    source_stub = types.SimpleNamespace(
        injected_power_spectrum=lambda frequencies, apodization=None: jnp.full((len(frequencies),), 2.0)
    )
    t = det.transmission(arrays, source_stub)
    np.testing.assert_allclose(np.array(t), np.array(flux) / 2.0, rtol=1e-6)


def test_transmission_is_general_detector_capability():
    """transmission() lives on the Detector base, so any measured-power detector can use it."""
    from fdtdx.objects.detectors.detector import Detector

    assert "transmission" in vars(Detector)
    assert "measured_power_spectrum" in vars(Detector)


def test_closed_surface_box_reports_net_power_not_a_plane_error():
    """A closed box reports net surface power, overriding the plane-only inherited version.

    ``PhasorDetector.measured_power_spectrum`` routes through ``flux_spectrum``, which
    requires a single singleton axis; without the override a box would raise "expects a
    plane detector" instead of reporting the power it is built to measure.
    """
    from fdtdx.objects.detectors.poynting_flux import ClosedSurfacePhasorPoyntingFluxDetector

    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = ClosedSurfacePhasorPoyntingFluxDetector(
        name="box", wave_characters=(WaveCharacter(wavelength=1e-6),), scaling_mode="pulse"
    )
    det = det.place_on_grid(((0, 4), (0, 4), (0, 4)), config, jax.random.PRNGKey(0))

    # Zero fields on every face -> zero net power, but crucially a value rather than a raise.
    faces = {}
    for axis in range(3):
        for side in ("min", "max"):
            shape = [4, 4, 4]
            shape[axis] = 1
            faces[f"phasor_axis{axis}_{side}"] = jnp.zeros((1, 1, 6, *shape), dtype=jnp.complex64)
    arrays = types.SimpleNamespace(detector_states={"box": faces})

    power = det.measured_power_spectrum(arrays)
    assert power.shape == (1,)
    np.testing.assert_allclose(np.asarray(power), 0.0, atol=1e-12)


def _zero_box_faces(shape=(4, 4, 4)):
    faces = {}
    for axis in range(3):
        for side in ("min", "max"):
            face_shape = list(shape)
            face_shape[axis] = 1
            faces[f"phasor_axis{axis}_{side}"] = jnp.zeros((1, 1, 6, *face_shape), dtype=jnp.complex64)
    return faces


@pytest.mark.parametrize("scaling_mode", ["pulse", "continuous"])
def test_closed_surface_power_matches_the_plane_convention(scaling_mode):
    """A box whose only flux leaves through one face reports what a plane there reports.

    ``compute_net_flux`` keeps its own standalone convention (recorded phasors, and the 1/2
    time average only in continuous mode), but ``measured_power_spectrum`` feeds
    ``transmission``, whose denominator is a raw windowed-DFT injected power. The two paths
    must therefore agree. A zero-field box cannot see this: any constant times zero is zero.
    """
    from fdtdx.objects.detectors.poynting_flux import ClosedSurfacePhasorPoyntingFluxDetector

    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    wave_characters = (WaveCharacter(wavelength=1e-6),)

    box = ClosedSurfacePhasorPoyntingFluxDetector(
        name="box", wave_characters=wave_characters, scaling_mode=scaling_mode
    )
    box = box.place_on_grid(((0, 4), (0, 4), (0, 4)), config, jax.random.PRNGKey(0))
    faces = _zero_box_faces()
    # Forward +z plane wave (Ex=1, Hy=1) crossing the z-max face only.
    faces["phasor_axis2_max"] = faces["phasor_axis2_max"].at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)

    plane = PhasorDetector(
        name="plane", wave_characters=wave_characters, reduce_volume=False, scaling_mode=scaling_mode
    )
    plane = plane.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64).at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)

    box_power = float(box.measured_power_spectrum(types.SimpleNamespace(detector_states={"box": faces}))[0])
    plane_power = float(plane.flux_spectrum(_arrays_with_phasor("plane", phasor))[0])
    assert box_power > 0
    # Compare as a ratio: the absolute powers are ~3e-13, under pytest.approx's default 1e-12
    # absolute floor, so a direct comparison would accept any wrong constant factor.
    assert box_power / plane_power == pytest.approx(1.0, rel=1e-5), f"box={box_power:.6e} plane={plane_power:.6e}"


@pytest.mark.parametrize("stride", [2, 3, 5])
def test_flux_spectrum_is_invariant_to_dft_subsample(stride):
    """The recorded phasor already estimates the every-step DFT, so flux must not track stride.

    ``update()`` multiplies each kept sample by the stride precisely so the thinned sum matches
    every-step recording. Un-scaling by the whole ``_static_scale()`` undoes that compensation,
    and the quadratic ``E x H*`` squares it into a ``1 / stride**2`` deficit.
    """
    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64).at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)

    def flux(subsample):
        det = PhasorDetector(
            name="p",
            wave_characters=(WaveCharacter(wavelength=1e-6),),
            reduce_volume=False,
            scaling_mode="pulse",
            dft_subsample=subsample,
        )
        det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
        return float(det.flux_spectrum(_arrays_with_phasor("p", phasor))[0])

    strided, every_step = flux(stride), flux(1)
    # Ratio, not difference: these powers sit below pytest.approx's default 1e-12 absolute floor.
    assert strided / every_step == pytest.approx(1.0, rel=1e-6), (
        f"stride={stride}: {strided:.6e} vs every-step {every_step:.6e}"
    )


@pytest.mark.parametrize("direction,expected_sign", [("+", 1.0), ("-", -1.0)])
def test_phasor_poynting_flux_spectrum_honors_direction(direction, expected_sign):
    """``flux_spectrum`` must use the detector's own normal, as ``compute_poynting_flux`` does.

    They read the same phasors on the same object, so a sign disagreement means
    ``transmission`` (built on ``flux_spectrum``) contradicts the detector's own flux method.
    """
    import math

    from fdtdx.objects.detectors.poynting_flux import PhasorPoyntingFluxDetector

    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = PhasorPoyntingFluxDetector(
        name="d", wave_characters=(WaveCharacter(wavelength=1e-6),), direction=direction, scaling_mode="pulse"
    )
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    # Forward +z plane wave (Ex=1, Hy=1): S_z = +1 per cell before the detector's own convention.
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64).at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)

    flux = float(det.flux_spectrum(_arrays_with_phasor("d", phasor))[0])
    own = float(det.compute_poynting_flux({"phasor": phasor})[0])

    assert math.copysign(1.0, flux) == expected_sign, f"direction={direction!r} gave flux={flux:.4e}"
    assert math.copysign(1.0, flux) == math.copysign(1.0, own), (
        f"flux_spectrum={flux:.4e} disagrees in sign with compute_poynting_flux={own:.4e}"
    )
    # Magnitude is unchanged by the sign convention. abs=0: this is ~3e-13, under approx's floor.
    assert abs(flux) == pytest.approx(0.5 * (2e-7**2) * 16, rel=1e-5, abs=0)


def test_backward_monitor_reports_a_backward_wave_as_positive():
    """A ``direction="-"`` monitor is how you measure reflection; R must come out positive.

    Without the direction convention the caller has to wrap the result in ``abs()``, which also
    hides a genuinely wrong sign.
    """
    from fdtdx.objects.detectors.poynting_flux import PhasorPoyntingFluxDetector

    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = PhasorPoyntingFluxDetector(
        name="back", wave_characters=(WaveCharacter(wavelength=1e-6),), direction="-", scaling_mode="pulse"
    )
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    # Backward -z plane wave (Ex=1, Hy=-1): S_z = -1 per cell, i.e. +1 through this monitor.
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64).at[0, 0, 0].set(1.0).at[0, 0, 4].set(-1.0)
    source_stub = types.SimpleNamespace(
        injected_power_spectrum=lambda frequencies, apodization=None: jnp.full((len(frequencies),), 3.2e-13)
    )

    t = float(det.transmission(_arrays_with_phasor("back", phasor), source_stub)[0])
    assert t > 0, f"backward monitor reported a negative fraction for a backward wave: {t:.4f}"
    assert t == pytest.approx(1.0, rel=1e-5)


def test_detector_without_a_spectrum_cannot_do_transmission():
    """A detector that records no spectrum must say so, not fail obscurely downstream."""
    from fdtdx.objects.detectors.field import FieldDetector

    det = FieldDetector(name="f")
    with pytest.raises(NotImplementedError, match="does not implement measured_power_spectrum"):
        det.measured_power_spectrum(_arrays_with_phasor("f", jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64)))
    with pytest.raises(NotImplementedError, match="no frequency set"):
        det._default_transmission_frequencies()
    # No window to forward: only the phasor family carries an apodization.
    assert det._injection_apodization() is None


def test_flux_spectrum_rejects_reduce_volume():
    """reduce_volume collapses the plane, so there is nothing left to integrate over."""
    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = PhasorDetector(name="r", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=True)
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="reduce_volume=False"):
        det.flux_spectrum(_arrays_with_phasor("r", jnp.zeros((1, 1, 6), dtype=jnp.complex64)))


def test_flux_spectrum_rejects_a_non_plane_detector():
    """The surface integral needs exactly one singleton axis to define a normal."""
    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = PhasorDetector(name="v", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=False)
    det = det.place_on_grid(((0, 4), (0, 4), (0, 4)), config, jax.random.PRNGKey(0))
    with pytest.raises(ValueError, match="expects a plane detector"):
        det.flux_spectrum(_arrays_with_phasor("v", jnp.zeros((1, 1, 6, 4, 4, 4), dtype=jnp.complex64)))


def test_update_rejects_an_invalid_scaling_mode():
    """The recording loop refuses a mode it has no scale for, rather than accumulating garbage."""
    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = PhasorDetector(name="s", wave_characters=(WaveCharacter(wavelength=1e-6),), reduce_volume=False)
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    det = det.aset("scaling_mode", "bogus")
    fields = jnp.ones((3, 4, 4, 1))
    with pytest.raises(Exception, match="Invalid scaling mode"):
        det.update(jnp.array(0), fields, fields, det.init_state(), fields, 1.0)


def test_flux_spectrum_uses_per_cell_areas_on_a_non_uniform_grid():
    """On a resolved rectilinear grid the face area varies per cell and must be integrated as such.

    The uniform fallback (``spacing**2`` everywhere) would silently mis-weight such a plane.
    """
    import fdtdx

    spacing, ny = 2e-7, 4
    config = SimulationConfig(
        time=1e-13, grid=fdtdx.QuasiUniformGrid(dx=spacing, dy=spacing, dz=spacing), backend="cpu"
    )
    volume = fdtdx.SimulationVolume(partial_real_shape=(4 * spacing, ny * spacing, 4 * spacing))
    det = fdtdx.PhasorDetector(
        name="p",
        partial_grid_shape=(None, None, 1),
        wave_characters=(WaveCharacter(wavelength=1e-6),),
        reduce_volume=False,
        scaling_mode="pulse",
    )
    constraints = [
        det.same_size(volume, axes=(0, 1)),
        det.place_at_center(volume, axes=(0, 1)),
        det.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(1,)),
    ]
    oc, _, _, config, _ = fdtdx.place_objects(
        object_list=[volume, det], config=config, constraints=constraints, key=jax.random.PRNGKey(0)
    )
    assert config.resolved_grid is not None  # otherwise this exercises the uniform fallback
    placed = oc["p"]

    shape = placed.grid_shape
    phasor = jnp.zeros((1, 1, 6, *shape), dtype=jnp.complex64).at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)
    flux = float(placed.flux_spectrum(_arrays_with_phasor("p", phasor))[0])
    expected = 0.5 * float(jnp.sum(placed._face_area(2)))
    assert flux == pytest.approx(expected, rel=1e-5, abs=0)


def _constant_mode_function(*, coordinates, frequency, propagation_axis, inv_permittivity):
    """A trivial reference mode: uniform E along axis 0, H along axis 1."""
    del frequency, propagation_axis, inv_permittivity
    amp = jnp.ones(coordinates[0].shape, dtype=jnp.float32)
    mode_E = jnp.zeros((3, *amp.shape), dtype=jnp.float32).at[0].set(amp)
    mode_H = jnp.zeros((3, *amp.shape), dtype=jnp.float32).at[1].set(amp)
    return mode_E, mode_H


def _applied_mode_detector(scaling_mode="pulse"):
    from fdtdx.objects.detectors.mode import CustomModeOverlapDetector

    config = SimulationConfig(time=1e-13, grid=UniformGrid(spacing=2e-7), backend="cpu")
    det = CustomModeOverlapDetector(
        name="m",
        wave_characters=(WaveCharacter(wavelength=1e-6),),
        mode_function=_constant_mode_function,
        normalize=False,
        scaling_mode=scaling_mode,
    )
    det = det.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    det = det.apply(jax.random.PRNGKey(0), jnp.ones((3, 4, 4, 4), dtype=jnp.float32), 1.0)
    return det


def test_modal_transmission_requires_pulse_scaling():
    """Continuous scaling puts the modal power in a different convention than the injected power."""
    det = _applied_mode_detector(scaling_mode="continuous")
    with pytest.raises(ValueError, match="requires scaling_mode='pulse'"):
        det.modal_transmission(None, None)


def test_modal_transmission_is_quadratic_in_the_recorded_field():
    """Modal power is ``|overlap|^2``, so doubling the recorded phasor quadruples the fraction."""
    det = _applied_mode_detector()
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64).at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)
    source_stub = types.SimpleNamespace(
        injected_power_spectrum=lambda frequencies, apodization=None: jnp.full((len(frequencies),), 1e-12)
    )

    single = float(det.modal_transmission(_arrays_with_phasor("m", phasor), source_stub)[0])
    double = float(det.modal_transmission(_arrays_with_phasor("m", 2 * phasor), source_stub)[0])
    assert single > 0
    assert double / single == pytest.approx(4.0, rel=1e-4)


def test_modal_transmission_divides_by_the_injected_power():
    """The injected power is the denominator, so twice the injection halves the fraction."""
    det = _applied_mode_detector()
    phasor = jnp.zeros((1, 1, 6, 4, 4, 1), dtype=jnp.complex64).at[0, 0, 0].set(1.0).at[0, 0, 4].set(1.0)
    arrays = _arrays_with_phasor("m", phasor)

    def stub(scale):
        return types.SimpleNamespace(
            injected_power_spectrum=lambda frequencies, apodization=None: jnp.full((len(frequencies),), scale)
        )

    weak = float(det.modal_transmission(arrays, stub(1e-12))[0])
    strong = float(det.modal_transmission(arrays, stub(2e-12))[0])
    assert strong / weak == pytest.approx(0.5, rel=1e-4)
