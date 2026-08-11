"""Unit tests for the analytic ``Source.injected_power_spectrum``.

Placement and ``apply_params`` populate ``source._E`` / ``source._H``, which is all the
analytic power needs — no time stepping, so these stay unit tests.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import fdtdx
from fdtdx.core.wavelength import WaveCharacter
from fdtdx.core.window import TukeyWindow
from fdtdx.objects.sources.source import Source

_WL = 1e-6
_SPACING = 5e-8


def _place_plane_source(direction="+", amplitude=1.0, cells=8, dispersive=False):
    """Place + apply a uniform plane source and return it (no FDTD run)."""
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_SPACING), time=2e-14, dtype=jnp.float32)
    wave = WaveCharacter(wavelength=_WL)
    volume = fdtdx.SimulationVolume(partial_real_shape=(cells * _SPACING, cells * _SPACING, cells * _SPACING))
    source = fdtdx.UniformPlaneSource(
        name="src",
        partial_grid_shape=(None, None, 1),
        wave_character=wave,
        temporal_profile=fdtdx.GaussianPulseProfile(
            center_wave=wave, spectral_width=WaveCharacter(wavelength=3e-6)
        ),
        direction=direction,
        fixed_E_polarization_vector=(1, 0, 0),
        static_amplitude_factor=amplitude,
    )
    constraints = [
        source.same_size(volume, axes=(0, 1)),
        source.place_at_center(volume, axes=(0, 1)),
        source.set_grid_coordinates(axes=(2,), sides=("-",), coordinates=(cells // 2,)),
    ]
    objects = [volume, source]
    if dispersive:
        # A Lorentz medium makes apply() build a separate H-side temporal profile for
        # broadband impedance matching, which the injected power must then use. The resonance
        # sits well above the 1 um carrier (~1.9e15 rad/s) so eps stays positive there; inside a
        # reststrahlen band eps < 0 makes the impedance sqrt(mu/eps) -- and hence the placed _H --
        # NaN, which is a property of the medium rather than of the power calculation.
        slab = fdtdx.UniformMaterialObject(
            name="slab",
            material=fdtdx.Material(
                permittivity=2.0,
                dispersion=fdtdx.DispersionModel(
                    poles=(fdtdx.LorentzPole(resonance_frequency=8e15, damping=5e12, delta_epsilon=0.5),)
                ),
            ),
        )
        constraints.append(slab.same_size(volume))
        objects.append(slab)
    key = jax.random.PRNGKey(0)
    oc, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objects, config=config, constraints=constraints, key=key
    )
    arrays, oc, _ = fdtdx.apply_params(arrays, oc, params, key)
    return oc["src"], config


def _freqs():
    return jnp.asarray([WaveCharacter(wavelength=_WL).get_frequency()])


def test_base_source_has_no_analytic_power():
    """The base contract refuses and points at the measured alternative.

    Every concrete source overrides this today, so the base implementation is reached only
    through ``super()`` — but a new source type that forgets to override lands here.
    """
    source, _ = _place_plane_source()
    with pytest.raises(NotImplementedError, match="ClosedSurfacePhasorPoyntingFluxDetector"):
        Source.injected_power_spectrum(source, _freqs())


def test_point_dipole_refuses_analytic_power():
    """A dipole's radiated power is environment-dependent, so no closed form exists."""
    dipole = fdtdx.PointDipoleSource(
        name="d",
        partial_grid_shape=(1, 1, 1),
        wave_character=WaveCharacter(wavelength=_WL),
        temporal_profile=fdtdx.SingleFrequencyProfile(),
        polarization=0,
    )
    with pytest.raises(NotImplementedError, match="Purcell"):
        dipole.injected_power_spectrum(_freqs())


class TestPlaneSourceInjectedPower:
    """The analytic TFSF injected power, exercised without a simulation."""

    def test_forward_source_injects_positive_power(self):
        source, _ = _place_plane_source(direction="+")
        power = source.injected_power_spectrum(_freqs())
        assert power.shape == (1,)
        assert float(power[0]) > 0

    def test_backward_source_flips_the_sign(self):
        """Power is a signed +axis flux, so a ``-`` source injects along -z."""
        forward, _ = _place_plane_source(direction="+")
        backward, _ = _place_plane_source(direction="-")
        assert float(forward.injected_power_spectrum(_freqs())[0]) > 0
        assert float(backward.injected_power_spectrum(_freqs())[0]) < 0

    def test_power_is_quadratic_in_amplitude(self):
        """Power goes as E x H, so doubling the amplitude quadruples it."""
        single, _ = _place_plane_source(amplitude=1.0)
        double, _ = _place_plane_source(amplitude=2.0)
        p1 = float(single.injected_power_spectrum(_freqs())[0])
        p2 = float(double.injected_power_spectrum(_freqs())[0])
        assert p2 / p1 == pytest.approx(4.0, rel=1e-4)

    def test_full_rectangular_window_matches_no_window(self):
        """An ``alpha=0`` Tukey spanning the record is the rectangular gate — an exact no-op."""
        source, config = _place_plane_source()
        end = (config.time_steps_total - 1) * config.time_step_duration
        rect = TukeyWindow(start_time=0.0, end_time=end, alpha=0.0)
        plain = float(source.injected_power_spectrum(_freqs())[0])
        windowed = float(source.injected_power_spectrum(_freqs(), apodization=rect)[0])
        assert windowed == pytest.approx(plain, rel=1e-4)

    def test_tapered_window_removes_signal(self):
        """A real taper must reduce the injected power — that is what cancels in transmission."""
        source, config = _place_plane_source()
        end = (config.time_steps_total - 1) * config.time_step_duration
        tapered = TukeyWindow(start_time=0.0, end_time=end, alpha=0.5)
        plain = float(source.injected_power_spectrum(_freqs())[0])
        windowed = float(source.injected_power_spectrum(_freqs(), apodization=tapered)[0])
        assert 0 < windowed < plain

    def test_spectrum_peaks_near_the_pulse_center(self):
        """The injected spectrum tracks the temporal profile, not just the spatial profile."""
        source, _ = _place_plane_source()
        center = float(WaveCharacter(wavelength=_WL).get_frequency())
        freqs = jnp.asarray([0.4 * center, center, 2.5 * center])
        power = np.asarray(source.injected_power_spectrum(freqs))
        assert power[1] > power[0] and power[1] > power[2]


def test_injected_power_requires_apply_params():
    """The analytic power reads the placed spatial profile, which only apply_params builds."""
    config = fdtdx.SimulationConfig(grid=fdtdx.UniformGrid(spacing=_SPACING), time=2e-14, dtype=jnp.float32)
    wave = WaveCharacter(wavelength=_WL)
    source = fdtdx.UniformPlaneSource(
        name="src",
        partial_grid_shape=(4, 4, 1),
        wave_character=wave,
        temporal_profile=fdtdx.SingleFrequencyProfile(),
        direction="+",
        fixed_E_polarization_vector=(1, 0, 0),
    )
    placed = source.place_on_grid(((0, 4), (0, 4), (0, 1)), config, jax.random.PRNGKey(0))
    with pytest.raises(Exception, match="Call apply_params"):
        placed.injected_power_spectrum(_freqs())


def test_dispersive_source_uses_the_filtered_h_side_profile():
    """In a dispersive medium the H side carries its own broadband-matched profile.

    Reusing the E-side signal there would drop the impedance correction, so the two setups
    must not agree — while both remain finite, positive powers.
    """
    plain, _ = _place_plane_source()
    dispersive, _ = _place_plane_source(dispersive=True)
    assert dispersive._temporal_H_filter is not None, "expected apply() to build an H-side filter"

    p_plain = float(plain.injected_power_spectrum(_freqs())[0])
    p_disp = float(dispersive.injected_power_spectrum(_freqs())[0])
    assert np.isfinite(p_disp) and p_disp > 0
    # Ratio, not difference: both powers are ~5e-13, under pytest.approx's default 1e-12 floor.
    assert abs(p_disp / p_plain - 1.0) > 1e-3, f"dispersive={p_disp:.6e} plain={p_plain:.6e}"
