"""Unit tests for the analytical doubly-clamped beam mode factory.

Verifies the closed-form Euler-Bernoulli mode shapes / frequencies used as a
physics oracle by the elasticity eigensolver tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from fdtdx.optimization.mechanical.modal import doubly_clamped_beam_modes

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Mode geometry: vanishing at clamps, symmetric about midpoint
# ---------------------------------------------------------------------------


def test_first_mode_clamps_and_symmetry() -> None:
    """First flexural mode must vanish at both endpoints and be symmetric."""
    out = doubly_clamped_beam_modes(
        length_m=10e-6,
        width_m=500e-9,
        thickness_m=220e-9,
        young_modulus=170e9,
        density=2330.0,
        grid_shape=(64, 4, 4),
        beam_axis=0,
        deflection_axis=2,
        n_modes=1,
    )
    phi = np.asarray(out["mode_shapes"])[0, 2]  # (Nx, Ny, Nz), z-component
    profile = phi[:, 2, 2]  # take a slice through the middle of width/thickness

    # Endpoints (the analytical mode is exactly zero at ξ=0 and ξ=1).
    assert abs(float(profile[0])) < 1e-5
    assert abs(float(profile[-1])) < 1e-5

    # Symmetry about midpoint.
    reversed_profile = profile[::-1]
    assert np.allclose(profile, reversed_profile, atol=1e-5)

    # The first symmetric mode is bell-shaped: maximum at the centre.
    assert int(np.argmax(np.abs(profile))) == len(profile) // 2 or int(
        np.argmax(np.abs(profile))
    ) == len(profile) // 2 - 1


# ---------------------------------------------------------------------------
# 2. Natural frequency matches Euler-Bernoulli closed form
# ---------------------------------------------------------------------------


def test_first_natural_frequency_doubly_clamped_si_beam() -> None:
    """ω₁ = (β₁/L)² √(EI/ρA) within 0.1 % of the analytical prediction."""
    L = 10e-6
    w = 500e-9
    t = 220e-9
    E = 170e9
    rho = 2330.0
    out = doubly_clamped_beam_modes(
        length_m=L,
        width_m=w,
        thickness_m=t,
        young_modulus=E,
        density=rho,
        grid_shape=(128, 4, 4),
        beam_axis=0,
        deflection_axis=2,
        n_modes=1,
    )
    f_predicted = float(out["natural_frequencies_hz"][0])

    beta = 4.730040744862704
    I = w * t**3 / 12.0
    A = w * t
    omega = (beta / L) ** 2 * np.sqrt(E * I / (rho * A))
    f_analytical = omega / (2.0 * np.pi)

    assert abs(f_predicted - f_analytical) / f_analytical < 1e-3
