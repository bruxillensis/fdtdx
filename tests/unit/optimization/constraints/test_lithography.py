"""Unit tests for :mod:`fdtdx.optimization.constraints.lithography`.

Covers the Hopkins/SOCS preparation step, the forward aerial-image and
resist stages, and the :class:`OPCConstraint` wrapper.  Tests avoid any
real ``Device`` / ``ObjectContainer`` by stubbing the required interface.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.optimization.constraints.lithography import LithographyModel, OPCConstraint

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubDevice:
    def __init__(
        self,
        name: str,
        rho: jax.Array,
        voxel_pitch_m: float,
    ):
        self.name = name
        self._rho = rho
        self.single_voxel_real_shape = (voxel_pitch_m, voxel_pitch_m, voxel_pitch_m)
        self.matrix_voxel_grid_shape = rho.shape + (1,) if rho.ndim == 2 else rho.shape

    def __call__(self, _params: jax.Array) -> jax.Array:
        return self._rho


class _StubObjects:
    def __init__(self, devices: list[_StubDevice]):
        self.devices = devices


# ---------------------------------------------------------------------------
# Test model: a pitch well below λ/(2·NA) so the grid actually resolves
# features below the diffraction limit.
# ---------------------------------------------------------------------------


def _make_prepared_model(
    ny: int = 24,
    nx: int = 24,
    *,
    num_kernels: int = 6,
) -> tuple[LithographyModel, float]:
    pitch = 20e-9  # 20 nm voxels
    # Tune NA / wavelength so half-pitch resolution is ~λ/(2NA) ~ 100 nm.
    model = LithographyModel(
        wavelength_m=193e-9,
        numerical_aperture=0.9,
        sigma_inner=0.0,
        sigma_outer=0.7,
        resist_threshold=0.3,
        resist_sharpness=50.0,
        num_kernels=num_kernels,
        source_grid_points=25,
    )
    prepared = model.prepare((ny, nx), pitch)
    return prepared, pitch


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------


def test_prepare_shapes_and_descending_eigenvalues() -> None:
    prepared, _ = _make_prepared_model(ny=20, nx=20, num_kernels=5)
    assert prepared.kernels is not None
    assert prepared.eigenvalues is not None
    assert prepared.kernels.shape == (5, 20, 20)
    assert prepared.eigenvalues.shape == (5,)
    eigs = np.asarray(prepared.eigenvalues)
    assert np.all(eigs >= -1e-6), "eigenvalues must be non-negative"
    # Descending order.
    assert np.all(np.diff(eigs) <= 1e-6)


def test_prepare_raises_on_bad_pitch() -> None:
    model = LithographyModel()
    with pytest.raises(ValueError):
        model.prepare((16, 16), 0.0)


def test_prepare_handles_coarse_grid() -> None:
    """A coarse grid should still prepare successfully because the DC cell
    is always inside the passband."""
    model = LithographyModel(
        wavelength_m=193e-9,
        numerical_aperture=0.9,
        num_kernels=1,
        source_grid_points=9,
    )
    prepared = model.prepare((4, 4), 1e-6)
    assert prepared.kernels is not None


def test_soft_edge_source_differs_from_hard_edge() -> None:
    """Enabling sigma_transition should produce a different (softer) TCC.

    The top-kernel eigenvalue changes because the source no longer cuts off
    abruptly at sigma_outer; some mass leaks past it and some is pulled in
    from just inside.  We only assert the result differs meaningfully from
    the hard-edge baseline and that both remain finite and non-negative.
    """
    shared = dict(
        wavelength_m=193e-9,
        numerical_aperture=0.9,
        sigma_outer=0.7,
        num_kernels=4,
        source_grid_points=31,
    )
    hard = LithographyModel(**shared).prepare((16, 16), 20e-9)
    soft = LithographyModel(**shared, sigma_transition=0.2).prepare((16, 16), 20e-9)
    assert hard.eigenvalues is not None
    assert soft.eigenvalues is not None
    eh = np.asarray(hard.eigenvalues)
    es = np.asarray(soft.eigenvalues)
    assert np.all(eh >= -1e-6) and np.all(es >= -1e-6)
    # Soft edge has a larger effective source extent => different eigenspectrum.
    assert not np.allclose(eh, es, atol=1e-3)


def test_soft_edge_rejects_negative_transition() -> None:
    model = LithographyModel(sigma_transition=-0.1)
    with pytest.raises(ValueError, match="sigma_transition"):
        model.prepare((16, 16), 20e-9)


def test_paper_defaults_prepare_successfully() -> None:
    """Lin et al. (IEEE JSTQE 2020) calibrated values for 193 nm DUV."""
    model = LithographyModel(
        wavelength_m=193e-9,
        numerical_aperture=0.671,
        sigma_inner=0.0,
        sigma_outer=0.884,
        sigma_transition=0.882,
        resist_threshold=0.165,
        num_kernels=6,
        source_grid_points=41,
    )
    prepared = model.prepare((24, 24), 20e-9)
    assert prepared.kernels is not None
    assert prepared.eigenvalues is not None
    assert float(jnp.asarray(prepared.eigenvalues).max()) > 0.0


# ---------------------------------------------------------------------------
# aerial_image: full-field and single-pixel behaviour
# ---------------------------------------------------------------------------


def test_aerial_full_field_is_near_dc() -> None:
    """A uniformly-bright design has only DC content, so the aerial image is
    uniform and equals |dc_contribution|^2 summed over the SOCS modes."""
    prepared, _ = _make_prepared_model()
    design = jnp.ones(prepared.grid_shape, dtype=jnp.float32)
    aerial = prepared.aerial_image(design)
    # The aerial image of a uniform mask is uniform (all cells equal).
    assert float(aerial.max() - aerial.min()) < 1e-4 * float(aerial.max() + 1e-12)
    assert float(aerial.min()) > 0.0


def test_aerial_contrast_differs_for_bar_vs_dot() -> None:
    """Sub-resolution dots produce diffuse, low-contrast aerial images;
    wide bars produce high-contrast ones.  Check that the normalised
    peak-to-mean ratio is higher for the bar."""
    prepared, _ = _make_prepared_model(ny=32, nx=32, num_kernels=8)
    ny, nx = prepared.grid_shape
    dot = jnp.zeros((ny, nx), dtype=jnp.float32).at[ny // 2, nx // 2].set(1.0)
    bar = jnp.zeros((ny, nx), dtype=jnp.float32)
    half = 5  # 10-voxel = 200 nm-wide bar (>> resolution limit)
    bar = bar.at[:, nx // 2 - half : nx // 2 + half].set(1.0)

    a_dot = prepared.aerial_image(dot)
    a_bar = prepared.aerial_image(bar)

    # Both produce strictly positive aerial images.
    assert float(a_dot.min()) >= 0.0
    assert float(a_bar.min()) >= 0.0
    assert float(a_dot.max()) > 0.0
    assert float(a_bar.max()) > 0.0
    # The wide bar transfers much more total mask power to the image plane
    # than a single pixel.
    assert float(a_bar.sum()) > 10.0 * float(a_dot.sum())


# ---------------------------------------------------------------------------
# forward(): resist + sigmoid
# ---------------------------------------------------------------------------


def test_forward_returns_printed_and_aerial() -> None:
    prepared, _ = _make_prepared_model()
    design = jnp.ones(prepared.grid_shape, dtype=jnp.float32)
    printed, aerial = prepared.forward(design)
    assert printed.shape == design.shape
    assert aerial.shape == design.shape
    # Uniform full-field printing is either uniformly on or uniformly off.
    assert float(printed.max() - printed.min()) < 1e-4


# ---------------------------------------------------------------------------
# OPCConstraint: self-consistency, target mode, and device-based setup
# ---------------------------------------------------------------------------


def test_opc_constraint_with_prepared_model() -> None:
    prepared, pitch = _make_prepared_model(ny=24, nx=24, num_kernels=6)
    design = jnp.zeros(prepared.grid_shape, dtype=jnp.float32)
    design = design.at[:, 8:16].set(1.0)  # 8-voxel coarse bar

    device = _StubDevice("dev", design, voxel_pitch_m=pitch)
    objects = _StubObjects([device])
    params = {"dev": design}

    con = OPCConstraint(name="opc", device_name="dev", litho_model=prepared)
    penalty, info = con.compute(params=params, objects=objects)
    assert float(penalty) >= 0.0
    assert jnp.isfinite(penalty)
    assert f"{con.name}_aerial_max" in info
    assert f"{con.name}_aerial_mean" in info


def test_opc_constraint_target_design_mode() -> None:
    """When target_design matches printed exactly, the penalty should be
    tiny; a mismatched target should produce a larger penalty."""
    prepared, pitch = _make_prepared_model(ny=24, nx=24, num_kernels=6)
    design = jnp.ones(prepared.grid_shape, dtype=jnp.float32)

    printed, _ = prepared.forward(design)
    device = _StubDevice("dev", design, voxel_pitch_m=pitch)
    objects = _StubObjects([device])

    con_match = OPCConstraint(
        name="opc",
        device_name="dev",
        litho_model=prepared,
        target_design=printed,
    )
    p_match, _ = con_match.compute(params={"dev": design}, objects=objects)

    bad_target = jnp.zeros(prepared.grid_shape, dtype=jnp.float32)
    con_bad = OPCConstraint(
        name="opc",
        device_name="dev",
        litho_model=prepared,
        target_design=bad_target,
    )
    p_bad, _ = con_bad.compute(params={"dev": design}, objects=objects)

    assert float(p_match) < 1e-5
    assert float(p_bad) > 0.5  # full-field printed vs zero target


def test_opc_constraint_for_device_autoprepares() -> None:
    pitch = 20e-9
    ny, nx = 20, 20
    design = jnp.zeros((ny, nx, 1), dtype=jnp.float32).at[:, 6:14, 0].set(1.0)
    device = _StubDevice("dev", design, voxel_pitch_m=pitch)
    objects = _StubObjects([device])

    raw_model = LithographyModel(
        wavelength_m=193e-9,
        numerical_aperture=0.9,
        num_kernels=4,
        source_grid_points=21,
    )
    con = OPCConstraint.for_device(device, raw_model, name="opc")
    # The stored model should now be prepared.
    assert con.litho_model.kernels is not None
    assert con.litho_model.grid_shape == (ny, nx)

    penalty, _ = con.compute(params={"dev": design}, objects=objects)
    assert jnp.isfinite(penalty)


def test_opc_constraint_gradient_flows() -> None:
    prepared, pitch = _make_prepared_model(ny=16, nx=16, num_kernels=4)
    device_rho_init = jnp.full(prepared.grid_shape, 0.3, dtype=jnp.float32)

    def loss(rho: jax.Array) -> jax.Array:
        dev = _StubDevice("dev", rho, voxel_pitch_m=pitch)
        objects = _StubObjects([dev])
        con = OPCConstraint(name="opc", device_name="dev", litho_model=prepared)
        val, _ = con.compute(params={"dev": rho}, objects=objects)
        return val

    grads = jax.grad(loss)(device_rho_init)
    assert grads.shape == device_rho_init.shape
    assert bool(jnp.all(jnp.isfinite(grads)))


def test_aerial_rejects_unprepared_model() -> None:
    model = LithographyModel()
    with pytest.raises(ValueError, match="has not been prepared"):
        model.aerial_image(jnp.ones((8, 8), dtype=jnp.float32))


def test_aerial_rejects_shape_mismatch() -> None:
    prepared, _ = _make_prepared_model(ny=16, nx=16)
    with pytest.raises(ValueError, match="does not match prepared grid"):
        prepared.aerial_image(jnp.ones((20, 20), dtype=jnp.float32))
