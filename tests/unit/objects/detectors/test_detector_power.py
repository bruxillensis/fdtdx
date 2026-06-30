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
    assert float(flux[0]) == pytest.approx(expected, rel=1e-5)


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
    # and it agrees with the detector's own net-flux method
    np.testing.assert_allclose(np.asarray(power), np.asarray(det.compute_net_flux(faces)), atol=1e-12)
