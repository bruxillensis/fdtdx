"""Unit tests for :mod:`fdtdx.optimization.constraints.manufacturing`.

The constraints read a device's post-projection density and the device's XY
voxel pitch, then evaluate a filter-and-project penalty.  Tests use stub
devices whose ``__call__`` returns a user-supplied density and whose
``single_voxel_real_shape`` supplies a known pitch.  No real
``ObjectContainer`` / ``Device`` is needed.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fdtdx.optimization.constraints.manufacturing import (
    MinInclusion,
    MinLineSpace,
    NoFloatingMaterial,
)
from fdtdx.optimization.utils.morphology import smooth_erosion

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Stubs
# ---------------------------------------------------------------------------


class _StubDevice:
    def __init__(self, name: str, rho: jax.Array, voxel_pitch_m: float = 10e-9):
        self.name = name
        self._rho = rho
        self.single_voxel_real_shape = (voxel_pitch_m, voxel_pitch_m, voxel_pitch_m)

    def __call__(self, _params: jax.Array) -> jax.Array:
        return self._rho


class _StubObjects:
    def __init__(self, devices: list[_StubDevice]):
        self.devices = devices


def _wide_bar(ny: int, nx: int, bar_width: int) -> jax.Array:
    """Solid horizontal bar of the given width (in voxels) centred vertically."""
    img = np.zeros((ny, nx), dtype=np.float32)
    y0 = (ny - bar_width) // 2
    img[y0 : y0 + bar_width, :] = 1.0
    return jnp.asarray(img)


# ---------------------------------------------------------------------------
# MinLineSpace
# ---------------------------------------------------------------------------


def test_min_line_space_low_on_coarse_pattern() -> None:
    """A large, well-spaced feature on a large canvas should produce a
    small (but not necessarily zero) penalty.  We check that the penalty
    is meaningfully below the full-1 reference level."""
    pitch = 10e-9
    ny, nx = 41, 41
    # 21-voxel bar (210 nm) on a 41-voxel canvas (410 nm).
    bar = _wide_bar(ny, nx, bar_width=21)
    objects = _StubObjects([_StubDevice("dev", bar, voxel_pitch_m=pitch)])

    con = MinLineSpace(
        name="mls",
        device_name="dev",
        min_line_width_m=30e-9,  # 3 voxels
        min_space_m=30e-9,
    )
    penalty, info = con.compute(params={"dev": bar}, objects=objects)
    # Penalty is dominated by filter-response fall-off at the bar edges and
    # at the gap boundary; both scale as (thickness_of_edge / area).
    assert float(penalty) < 0.1
    assert f"{con.name}_thin_line" in info


def test_min_line_space_near_zero_on_empty_canvas() -> None:
    """A canvas with no solid features should have negligible penalty.

    The smooth-projection baseline at ``rho = 0`` is ``sigmoid(-beta*eta)``
    for the dilation term; with a sharper ``beta`` that baseline is
    negligible.
    """
    pitch = 10e-9
    ny, nx = 21, 21
    empty = jnp.zeros((ny, nx), dtype=jnp.float32)
    objects = _StubObjects([_StubDevice("dev", empty, voxel_pitch_m=pitch)])
    con = MinLineSpace(
        name="mls",
        device_name="dev",
        min_line_width_m=50e-9,
        min_space_m=50e-9,
        beta=30.0,
    )
    penalty, _ = con.compute(params={"dev": empty}, objects=objects)
    assert float(penalty) < 1e-4


def test_min_line_space_thin_line_violates_min_line_width() -> None:
    """A 1-voxel-wide solid line must produce a nonzero thin-line penalty
    component (which catches the rule violation independently of the
    gap term)."""
    pitch = 10e-9
    ny, nx = 21, 21
    thin = np.zeros((ny, nx), dtype=np.float32)
    thin[ny // 2, :] = 1.0
    thin = jnp.asarray(thin)
    objects = _StubObjects([_StubDevice("dev", thin, voxel_pitch_m=pitch)])
    con = MinLineSpace(
        name="mls",
        device_name="dev",
        min_line_width_m=50e-9,  # 5 voxels -> thin line violates line rule
        min_space_m=10e-9,  # small: the gap term is not the focus
    )
    _, info = con.compute(params={"dev": thin}, objects=objects)
    # The thin-line sub-penalty must be strictly positive.
    thin_line_component = float(info[f"{con.name}_thin_line"])
    assert thin_line_component > 0.01


def test_min_line_space_gradient_finite() -> None:
    pitch = 10e-9
    rho0 = _wide_bar(21, 21, bar_width=5)

    def loss(rho: jax.Array) -> jax.Array:
        objects = _StubObjects([_StubDevice("dev", rho, voxel_pitch_m=pitch)])
        con = MinLineSpace(
            name="mls",
            device_name="dev",
            min_line_width_m=30e-9,
            min_space_m=30e-9,
        )
        val, _ = con.compute(params={"dev": rho}, objects=objects)
        return val

    grads = jax.grad(loss)(rho0)
    assert bool(jnp.all(jnp.isfinite(grads)))
    assert bool(jnp.any(grads != 0.0))


# ---------------------------------------------------------------------------
# MinInclusion
# ---------------------------------------------------------------------------


def test_min_inclusion_zero_when_inner_inside_eroded_outer() -> None:
    """If the inner density is contained within the eroded outer, the
    penalty should be (essentially) zero."""
    pitch = 10e-9
    ny, nx = 25, 25
    outer = _wide_bar(ny, nx, bar_width=11)  # thick bar
    # Build an inner that lives strictly inside erode(outer, margin).
    from fdtdx.optimization.utils.morphology import meters_to_odd_kernel

    k = meters_to_odd_kernel(30e-9, pitch)  # 3-voxel erosion kernel
    eroded_outer = smooth_erosion(outer, k, beta=8.0, eta=0.75)
    # Threshold heavily and mask the inner strictly inside the eroded region.
    inner = (eroded_outer > 0.8).astype(jnp.float32) * 0.5

    objects = _StubObjects(
        [
            _StubDevice("inner", inner, voxel_pitch_m=pitch),
            _StubDevice("outer", outer, voxel_pitch_m=pitch),
        ]
    )
    params = {"inner": inner, "outer": outer}

    con = MinInclusion(
        name="mi",
        inner_device_name="inner",
        outer_device_name="outer",
        min_margin_m=30e-9,
    )
    penalty, _ = con.compute(params=params, objects=objects)
    assert float(penalty) < 5e-3


def test_min_inclusion_nonzero_when_inner_outside_outer() -> None:
    """Inner poking outside the eroded outer must yield a positive penalty."""
    pitch = 10e-9
    ny, nx = 21, 21
    outer = _wide_bar(ny, nx, bar_width=5)  # narrow bar
    # Inner occupies entire domain -> vastly outside eroded outer.
    inner = jnp.ones((ny, nx), dtype=jnp.float32)

    objects = _StubObjects(
        [
            _StubDevice("inner", inner, voxel_pitch_m=pitch),
            _StubDevice("outer", outer, voxel_pitch_m=pitch),
        ]
    )
    params = {"inner": inner, "outer": outer}

    con = MinInclusion(
        name="mi",
        inner_device_name="inner",
        outer_device_name="outer",
        min_margin_m=30e-9,
    )
    penalty, info = con.compute(params=params, objects=objects)
    assert float(penalty) > 0.3
    assert f"{con.name}_max_violation" in info


def test_min_inclusion_xy_shape_mismatch_raises() -> None:
    pitch = 10e-9
    inner = jnp.ones((8, 8), dtype=jnp.float32)
    outer = jnp.ones((10, 10), dtype=jnp.float32)
    objects = _StubObjects(
        [
            _StubDevice("inner", inner, voxel_pitch_m=pitch),
            _StubDevice("outer", outer, voxel_pitch_m=pitch),
        ]
    )
    con = MinInclusion(
        name="mi",
        inner_device_name="inner",
        outer_device_name="outer",
        min_margin_m=30e-9,
    )
    with pytest.raises(ValueError, match="share XY grid"):
        con.compute(params={"inner": inner, "outer": outer}, objects=objects)


# ---------------------------------------------------------------------------
# NoFloatingMaterial
# ---------------------------------------------------------------------------


def test_no_floating_material_zero_for_monotone_stack() -> None:
    """Each upper layer <= lower layer => penalty is zero."""
    ny, nx = 10, 10
    bottom = jnp.ones((ny, nx), dtype=jnp.float32)
    middle = jnp.full((ny, nx), 0.7, dtype=jnp.float32)
    top = jnp.full((ny, nx), 0.4, dtype=jnp.float32)

    objects = _StubObjects(
        [
            _StubDevice("bot", bottom),
            _StubDevice("mid", middle),
            _StubDevice("top", top),
        ]
    )
    params = {"bot": bottom, "mid": middle, "top": top}

    con = NoFloatingMaterial(
        name="nf",
        device_stack_names=("bot", "mid", "top"),
    )
    penalty, _ = con.compute(params=params, objects=objects)
    assert float(penalty) == pytest.approx(0.0, abs=1e-6)


def test_no_floating_material_positive_for_floating_pixel() -> None:
    """A single floating pixel on top of a void in the layer below gives a
    positive, analytically predictable penalty."""
    ny, nx = 10, 10
    bottom = jnp.zeros((ny, nx), dtype=jnp.float32)
    top = jnp.zeros((ny, nx), dtype=jnp.float32).at[4, 4].set(1.0)

    objects = _StubObjects(
        [
            _StubDevice("bot", bottom),
            _StubDevice("top", top),
        ]
    )
    params = {"bot": bottom, "top": top}

    con = NoFloatingMaterial(
        name="nf",
        device_stack_names=("bot", "top"),
    )
    penalty, info = con.compute(params=params, objects=objects)
    # excess = relu(1 - 0) = 1 at one cell out of ny*nx; mean(excess^2) = 1/100.
    assert float(penalty) == pytest.approx(1.0 / (ny * nx), rel=1e-6)
    assert float(info[f"{con.name}_max_excess"]) == pytest.approx(1.0, rel=1e-6)


def test_no_floating_material_rejects_short_stack() -> None:
    objects = _StubObjects([_StubDevice("a", jnp.zeros((4, 4)))])
    con = NoFloatingMaterial(name="nf", device_stack_names=("a",))
    with pytest.raises(ValueError, match="at least two"):
        con.compute(params={"a": jnp.zeros((4, 4))}, objects=objects)


def test_no_floating_material_rejects_shape_mismatch() -> None:
    a = jnp.zeros((4, 4), dtype=jnp.float32)
    b = jnp.zeros((4, 5), dtype=jnp.float32)
    objects = _StubObjects([_StubDevice("a", a), _StubDevice("b", b)])
    con = NoFloatingMaterial(name="nf", device_stack_names=("a", "b"))
    with pytest.raises(ValueError, match="share shape"):
        con.compute(params={"a": a, "b": b}, objects=objects)


def test_no_floating_material_gradient_flows_to_top() -> None:
    ny, nx = 6, 6
    bottom = jnp.full((ny, nx), 0.3, dtype=jnp.float32)
    top_init = jnp.full((ny, nx), 0.7, dtype=jnp.float32)

    def objects_for(top: jax.Array) -> _StubObjects:
        return _StubObjects([_StubDevice("bot", bottom), _StubDevice("top", top)])

    def loss(top: jax.Array) -> jax.Array:
        con = NoFloatingMaterial(name="nf", device_stack_names=("bot", "top"))
        val, _ = con.compute(
            params={"bot": bottom, "top": top},
            objects=objects_for(top),
        )
        return val

    grads = jax.grad(loss)(top_init)
    assert bool(jnp.all(jnp.isfinite(grads)))
    # Gradient wrt the top layer is positive when top > bottom (encourages
    # reducing the excess).
    assert bool(jnp.all(grads > 0.0))
