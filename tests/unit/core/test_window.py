"""Tests for the shared temporal envelope helpers and the detector apodization windows."""

import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.core.wavelength import WaveCharacter
from fdtdx.core.window import (
    GaussianWindow,
    TemporalWindow,
    TukeyWindow,
    gaussian_envelope,
    linear_rampup,
    tukey_envelope,
    windowed_dft,
)
from fdtdx.objects.sources.profile import GaussianPulseProfile, SingleFrequencyProfile, TemporalProfile


class TestWindowShapes:
    """The pure envelope helpers shared by source profiles and detector windows."""

    def test_gaussian_envelope(self):
        """Peaks at the center, falls to exp(-1/2) at one sigma, and is symmetric."""
        t = jnp.linspace(0, 10, 101)
        w = gaussian_envelope(t, center=5.0, sigma=1.0)
        assert float(w[50]) == pytest.approx(1.0, abs=1e-6)
        assert float(gaussian_envelope(jnp.array(6.0), 5.0, 1.0)) == pytest.approx(np.exp(-0.5), rel=1e-6)
        np.testing.assert_allclose(np.array(w), np.array(w)[::-1], atol=1e-6)

    def test_linear_rampup(self):
        """Ramps 0 to 1 over the duration and clamps outside it."""
        t = jnp.array([-1.0, 0.0, 0.5, 1.0, 2.0])
        w = linear_rampup(t, ramp_duration=1.0)
        np.testing.assert_allclose(np.array(w), [0.0, 0.0, 0.5, 1.0, 1.0], atol=1e-6)

    def test_tukey_endpoints_and_flat_top(self):
        """Tapers to zero at both ends with a flat unit top in between."""
        t = jnp.linspace(0, 1, 101)
        w = tukey_envelope(t, start=0.0, end=1.0, alpha=0.5)
        assert float(w[0]) == pytest.approx(0.0, abs=1e-6)
        assert float(w[-1]) == pytest.approx(0.0, abs=1e-6)
        assert float(w[50]) == pytest.approx(1.0, abs=1e-6)

    def test_tukey_alpha_limits(self):
        """alpha=0 is rectangular, alpha=1 is Hann."""
        t = jnp.linspace(0, 1, 51)
        np.testing.assert_allclose(np.array(tukey_envelope(t, 0.0, 1.0, alpha=0.0)), 1.0, atol=1e-6)
        hann = tukey_envelope(t, 0.0, 1.0, alpha=1.0)
        np.testing.assert_allclose(np.array(hann), 0.5 * (1 - np.cos(2 * np.pi * np.array(t))), atol=1e-6)

    def test_tukey_zero_outside_range(self):
        """Outside [start, end] the window is exactly zero."""
        w = tukey_envelope(jnp.array([-0.1, 0.5, 1.1]), 0.0, 1.0, alpha=0.5)
        assert float(w[0]) == 0.0 and float(w[2]) == 0.0


class TestTemporalWindows:
    """TemporalWindow wraps the envelopes with the carrier-free detector interface."""

    def test_gaussian_window_matches_envelope(self):
        t = jnp.linspace(0, 1e-12, 50)
        win = GaussianWindow(center_time=5e-13, sigma_time=1e-13)
        np.testing.assert_allclose(np.array(win.get_window(t)), np.array(gaussian_envelope(t, 5e-13, 1e-13)), atol=1e-7)

    def test_tukey_window_matches_envelope(self):
        t = jnp.linspace(0, 1e-12, 50)
        win = TukeyWindow(start_time=0.0, end_time=1e-12, alpha=0.5)
        np.testing.assert_allclose(np.array(win.get_window(t)), np.array(tukey_envelope(t, 0.0, 1e-12, 0.5)), atol=1e-7)

    def test_windows_are_non_negative(self):
        """Window weights never change sign."""
        t = jnp.linspace(-1e-12, 2e-12, 200)
        for win in (
            GaussianWindow(center_time=5e-13, sigma_time=1e-13),
            TukeyWindow(start_time=0.0, end_time=1e-12, alpha=0.5),
        ):
            assert bool(jnp.all(win.get_window(t) >= 0.0))

    def test_windows_are_not_temporal_profiles(self):
        """Windows and source profiles are separate hierarchies."""
        assert issubclass(GaussianWindow, TemporalWindow)
        assert issubclass(TukeyWindow, TemporalWindow)
        assert not issubclass(GaussianWindow, TemporalProfile)
        assert not issubclass(SingleFrequencyProfile, TemporalWindow)
        assert not issubclass(GaussianPulseProfile, TemporalWindow)


class TestProfileRefactorRegression:
    """Extracting the envelope helpers must leave the source profiles unchanged."""

    def test_gaussian_pulse_unchanged(self):
        center = WaveCharacter(wavelength=1e-6)
        width = WaveCharacter(wavelength=2e-6)
        prof = GaussianPulseProfile(center_wave=center, spectral_width=width)
        t = jnp.linspace(0, 5e-14, 60)
        sigma_t = 1.0 / (2 * np.pi * width.get_frequency())
        t0 = 6 * sigma_t
        envelope = np.exp(-((np.array(t) - t0) ** 2) / (2 * sigma_t**2))
        carrier = np.real(np.exp(-1j * (2 * np.pi * center.get_frequency() * np.array(t) + center.phase_shift)))
        np.testing.assert_allclose(
            np.array(prof.get_amplitude(t, period=1e-15)), envelope * carrier, rtol=1e-4, atol=1e-5
        )

    def test_single_frequency_unchanged(self):
        prof = SingleFrequencyProfile(num_startup_periods=4)
        period = 1e-15
        t = jnp.linspace(0, 10 * period, 80)
        time_phase = 2 * np.pi * np.array(t) / period + 0.0 + prof.phase_shift
        raw = np.real(np.exp(-1j * time_phase))
        factor = np.clip(np.array(t) / (4 * period), 0, 1)
        np.testing.assert_allclose(np.array(prof.get_amplitude(t, period=period)), factor * raw, rtol=1e-4, atol=1e-5)


class TestWindowValidation:
    """Window parameters that would produce NaN or a non-window are rejected."""

    @pytest.mark.parametrize("sigma", [0.0, -1e-13])
    def test_gaussian_rejects_non_positive_sigma(self, sigma):
        with pytest.raises(ValueError, match="sigma_time must be positive"):
            GaussianWindow(center_time=5e-13, sigma_time=sigma)

    @pytest.mark.parametrize("end", [0.0, -1e-12, 1e-12])
    def test_tukey_rejects_non_increasing_bounds(self, end):
        with pytest.raises(ValueError, match="end_time must exceed start_time"):
            TukeyWindow(start_time=1e-12, end_time=end)

    @pytest.mark.parametrize("alpha", [-0.1, 1.5])
    def test_tukey_rejects_alpha_outside_unit_interval(self, alpha):
        with pytest.raises(ValueError, match=r"alpha must lie in \[0, 1\]"):
            TukeyWindow(start_time=0.0, end_time=1e-12, alpha=alpha)

    @pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
    def test_tukey_accepts_the_closed_unit_interval(self, alpha):
        win = TukeyWindow(start_time=0.0, end_time=1e-12, alpha=alpha)
        assert bool(jnp.all(jnp.isfinite(win.get_window(jnp.linspace(0, 1e-12, 20)))))


class TestWindowedDFT:
    """``windowed_dft`` is the spectrum the analytic source power is built on."""

    @staticmethod
    def _signal(n=32, seed=0):
        return jnp.asarray(np.random.default_rng(seed).standard_normal(n), dtype=jnp.float32)

    def test_matches_the_fft_at_exact_bins_under_a_flat_window(self):
        """With w=1 at bin frequencies it is the conjugate FFT — its exponent is ``+i w t``."""
        n = 32
        dt = 1.0 / n  # so FFT bin k sits exactly at f = k
        signal = self._signal(n)
        got = np.asarray(windowed_dft(signal, jnp.ones(n), jnp.arange(6.0), dt))
        expected = np.conj(np.fft.fft(np.asarray(signal, dtype=np.float64)))[:6]
        np.testing.assert_allclose(got, expected, rtol=1e-4, atol=1e-3)

    def test_window_multiplies_the_signal(self):
        """Weighting is exactly ``w * x``, not applied to the phase or the result."""
        n, dt = 32, 1.0 / 32
        signal = self._signal(n)
        window = jnp.asarray(tukey_envelope(jnp.arange(n) * dt, 0.0, (n - 1) * dt, 0.5))
        freqs = jnp.asarray([1.0, 2.5, 7.0])
        np.testing.assert_allclose(
            np.asarray(windowed_dft(signal, window, freqs, dt)),
            np.asarray(windowed_dft(signal * window, jnp.ones(n), freqs, dt)),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_dc_term_is_the_weighted_sum(self):
        """At f=0 every phase factor is 1, so the transform reduces to ``sum(w * x)``."""
        n, dt = 32, 1.0 / 32
        signal = self._signal(n)
        window = jnp.asarray(tukey_envelope(jnp.arange(n) * dt, 0.0, (n - 1) * dt, 0.5))
        dc = windowed_dft(signal, window, jnp.asarray([0.0]), dt)[0]
        assert float(jnp.abs(jnp.imag(dc))) < 1e-4
        assert float(jnp.real(dc)) == pytest.approx(float(jnp.sum(window * signal)), rel=1e-5)

    def test_evaluates_between_fft_bins(self):
        """The explicit sum exists so arbitrary (non-bin) frequencies can be requested."""
        n, dt = 32, 1.0 / 32
        signal = self._signal(n)
        off_bin = windowed_dft(signal, jnp.ones(n), jnp.asarray([3.5]), dt)[0]
        neighbours = np.asarray(windowed_dft(signal, jnp.ones(n), jnp.asarray([3.0, 4.0]), dt))
        assert np.isfinite(complex(off_bin))
        # A genuine interpolation point, not a silent snap to either neighbouring bin.
        assert not np.isclose(complex(off_bin), neighbours[0], rtol=1e-3)
        assert not np.isclose(complex(off_bin), neighbours[1], rtol=1e-3)
