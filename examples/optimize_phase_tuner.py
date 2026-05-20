"""Inverse co-design of a GDS dual-slot MEMS phase tuner (optics + mechanics).

Geometry is imported from ``examples/phase-tuner-draft.gds`` (single flat
cell ``phase-tuner``):

  * Layer 10 (220 nm Si device layer), 3 polygons — center grounded core
    (#0) and the top / bottom movable spring bodies (#1/#2).  **The whole
    L10 pattern is the inverse-design region**: ρ co-designs the optics
    *and* the mechanical suspension.
  * Layer 12 (90 nm Si rib slab), 2 polygons — west (#4) and east (#3) rib
    waveguides; imported static (the fixed I/O).

Stack (no wafer substrate): 3 µm SiO2 below + Si device layer + 3 µm SiO2
above, but **only in the anchored I/O end regions**; the central released
phase-tuner span is **air** around the suspended Si (so the springs move
and the slot optics work).

Actuation: a voltage between the grounded center and the top/bottom bodies
pulls them laterally (slot-closing) by the 3-D fringing Maxwell stress; the
lateral shift of the inverse-designed pattern changes the guided index.
Each epoch the pull-in voltage of the *current* ρ is recomputed by
displacement-controlled limit-point continuation, and the operating
voltages are set from it:  ``V_on = 0.9 · V_pull_in``,  ``V_off = V_on + 1``.

Objectives (all minimized, ramped):
  * insertion loss        (maximize through-transmission in both states)
  * back reflection       (both states)
  * 2-D footprint         (ρ-weighted x *and* y extent → a compact tuner)

A one-page live status figure (optical field, the inverse-design shapes,
the V_on/V_off deformed geometry, the objective history and a stats table
incl. the pull-in voltage) is drawn every logged epoch via the
``Optimization.epoch_callback`` hook and overwrites ``status.png`` so it can
be watched live / re-plotted in Jupyter (see ``plot_status``).

Pull-in voltage path:  the mechanical eigensolve runs on the Si
subdomain of the current ρ (element-level mask; only Q1 elements with
all 8 corners in Si contribute strain energy, air nodes pinned) — the
same mesh-only-the-solid physics COMSOL uses, so V_pi tracks the real
folded-spring stiffness (<10 V on this draft geometry).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, cast

import gdstk
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import optax
from loguru import logger

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import fdtdx  # noqa: E402
from fdtdx import colors  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GDS_PATH = Path(__file__).with_name("phase-tuner-draft.gds")
GDS_CELL = "phase-tuner"
L10, L12 = 10, 12  # GDS layers: 220 nm device / 90 nm rib slab

PERMITTIVITY_SI_OPT = fdtdx.constants.relative_permittivity_silicon  # ~12.25
PERMITTIVITY_SIO2 = fdtdx.constants.relative_permittivity_silica     # ~2.25
EPS_SI_DC = 1.0e4   # doped-Si conductor surrogate (quasi-equipotential)
EPS_AIR_DC = 1.0

SI_YOUNG_MODULUS = 170e9
SI_DENSITY = 2330.0

V_SWING = 1.0          # V_off = V_on + V_SWING
_PULLIN_SAFETY = 0.9   # V_on = 0.9 · V_pull_in (of the *current* geometry)
_V_PI_FALLBACK = 5.0   # used only if the per-epoch pull-in solve degenerates

_BASE_DETECTORS = ("overlap_out", "back_flux", "fwd_flux", "optical_energy")


# ---------------------------------------------------------------------------
# GDS helpers
# ---------------------------------------------------------------------------
def _gds_polys(layer: int) -> list[tuple[np.ndarray, float]]:
    """Return [(vertices_um, area_um2)] for every polygon on ``layer``."""
    lib = gdstk.read_gds(str(GDS_PATH))
    cell = next(c for c in lib.cells if c.name == GDS_CELL)
    u = lib.unit * 1e6  # GDS units → µm
    out = []
    for p in cell.polygons:
        if p.layer != layer:
            continue
        v = np.asarray(p.points) * u
        x, y = v[:, 0], v[:, 1]
        area = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        out.append((v, float(area)))
    return out


def _half_poly_um(vertices_um: np.ndarray) -> np.ndarray:
    """Clip a polygon to the y ≥ 0 half-plane (y-symmetry).

    The static GDS shapes (rib slab, center core) straddle y=0; under the
    y=0 symmetry plane only the upper half is simulated.  Returns the
    largest resulting polygon's vertices (µm); these shapes stay simply-
    connected when clipped at y=0, so a single polygon is expected.
    """
    xmin, xmax = float(vertices_um[:, 0].min()) - 1.0, float(vertices_um[:, 0].max()) + 1.0
    ymax = float(vertices_um[:, 1].max()) + 1.0
    keep = gdstk.rectangle((xmin, 0.0), (xmax, ymax))
    out = gdstk.boolean(gdstk.Polygon(vertices_um), keep, "and")
    if not out:
        raise ValueError("y≥0 clip produced no polygon")
    big = max(out, key=lambda p: p.area())
    return np.asarray(big.points)


def _rasterize(polys: list[np.ndarray], xs_um: np.ndarray, ys_um: np.ndarray) -> np.ndarray:
    """Union point-in-polygon mask on the (xs × ys) µm grid → (Nx, Ny) float32."""
    pts = [(float(x), float(y)) for x in xs_um for y in ys_um]
    mask = np.zeros((xs_um.size, ys_um.size), bool)
    for v in polys:
        ins = np.asarray(gdstk.inside(pts, gdstk.Polygon(v))).reshape(xs_um.size, ys_um.size)
        mask |= ins
    return mask.astype(np.float32)


# ---------------------------------------------------------------------------
# Status page (live during optimization + Jupyter replot)
# ---------------------------------------------------------------------------
def _imshow(ax, data, title, cmap, *, symmetric=False):
    a = np.asarray(data).T
    kw: dict[str, Any] = dict(aspect="auto", origin="lower", cmap=cmap)
    if symmetric:
        m = float(np.max(np.abs(a))) or 1.0
        kw.update(vmin=-m, vmax=m)
    im = ax.imshow(a, **kw)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def render_status_figure(panels: dict, history: dict, stats: dict, out_png: Path) -> None:
    """Build the one-page status figure from plain numpy/dict inputs.

    Importable & reusable from Jupyter (see :func:`plot_status`).  ``panels``
    holds 2-D arrays; ``history`` holds 1-D objective series; ``stats`` is a
    flat str→value dict rendered as a table.
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(3, 4, height_ratios=[1.0, 1.0, 0.8])

    _imshow(fig.add_subplot(gs[0, 0]), panels["rho"], "inverse-design ρ (x–y)", "viridis")
    _imshow(fig.add_subplot(gs[0, 1]), panels["optical"], "optical |E|² (x–z)", "inferno")
    _imshow(fig.add_subplot(gs[0, 2]), panels["deps_on"], "Δε  V_on (x–y)", "RdBu_r", symmetric=True)
    _imshow(fig.add_subplot(gs[0, 3]), panels["deps_off"], "Δε  V_off (x–y)", "RdBu_r", symmetric=True)
    _imshow(fig.add_subplot(gs[1, 0]), panels["uy_on"], "lateral u_y V_on (x–y)", "RdBu_r", symmetric=True)
    _imshow(fig.add_subplot(gs[1, 1]), panels["uy_off"], "lateral u_y V_off (x–y)", "RdBu_r", symmetric=True)

    axo = fig.add_subplot(gs[1, 2:])
    for key, lab in (("loss", "loss"), ("insertion", "insertion"),
                     ("back_refl", "back-refl"), ("footprint", "footprint")):
        y = np.asarray(history.get(key, []))
        if y.size:
            axo.plot(np.arange(y.size), y, label=lab, linewidth=1.5)
    axo.set_xlabel("epoch")
    axo.set_yscale("symlog", linthresh=1e-3)
    axo.set_title("objective history", fontsize=9)
    axo.legend(fontsize=8)
    axo.grid(True, alpha=0.3)

    axt = fig.add_subplot(gs[2, :])
    axt.axis("off")
    items = list(stats.items())
    rows = [items[i : i + 4] for i in range(0, len(items), 4)]
    cells, labels = [], []
    for r in rows:
        labels.append([k for k, _ in r] + [""] * (4 - len(r)))
        cells.append([f"{v}" for _, v in r] + [""] * (4 - len(r)))
    tbl = axt.table(cellText=cells, cellLoc="left", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for j, row in enumerate(rows):
        for i in range(len(row)):
            tbl[(j, i)].get_text().set_text(f"{labels[j][i]} = {cells[j][i]}")
    axt.set_title("status / stats", fontsize=9)

    fig.suptitle("phase-tuner inverse co-design — live status", fontsize=13)
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)


def plot_status(logdir: str | Path):
    """Jupyter helper: rebuild the status page from a run directory.

    ``render_status_figure`` is fed live during a run; this re-reads the
    persisted ``status_panels.npz`` / ``status_history.npz`` / ``status.json``
    that the callback writes so a notebook can replot the latest state::

        from examples.optimize_phase_tuner import plot_status
        plot_status("outputs/.../mems_phase_tuner/<run>")
    """
    import json

    d = Path(logdir)
    panels = {k: v for k, v in np.load(d / "status_panels.npz").items()}
    history = {k: v for k, v in np.load(d / "status_history.npz").items()}
    stats = json.loads((d / "status.json").read_text())
    render_status_figure(panels, history, stats, d / "status.png")
    return d / "status.png"


def main(args):
    seed = args.seed_rng
    evaluation = args.evaluation
    seed_from = args.seed_from
    seed_iter = args.seed_iter
    resume_from = args.resume_from

    exp_logger = fdtdx.Logger(experiment_name="phase_tuner", name=None)
    key = jax.random.PRNGKey(seed=seed)

    # ------------------------------------------------------------------
    # Simulation config
    # ------------------------------------------------------------------
    wavelength = 1.55e-6
    period = fdtdx.constants.wavelength_to_period(wavelength)
    resolution = 50e-9
    # Device is ~30 µm end-to-end → light needs a few hundred fs to traverse
    # and settle (group velocity ≈ c/4 in Si).
    config = fdtdx.SimulationConfig(
        time=450e-15, resolution=resolution, dtype=jnp.float32, courant_factor=0.99
    )
    period_steps = round(period / config.time_step_duration)
    all_time_steps = list(range(config.time_steps_total))
    if not evaluation:
        config = config.aset(
            "gradient_config",
            fdtdx.GradientConfig(
                recorder=fdtdx.Recorder(modules=[fdtdx.DtypeConversion(dtype=jnp.float8_e4m3fnuz)])
            ),
        )

    # ------------------------------------------------------------------
    # GDS-derived geometry constants
    # ------------------------------------------------------------------
    l10 = _gds_polys(L10)
    l12 = _gds_polys(L12)
    all_v = np.concatenate([v for v, _ in l10 + l12], axis=0)
    x_min, x_max = float(all_v[:, 0].min()), float(all_v[:, 0].max())
    y_min, y_max = float(all_v[:, 1].min()), float(all_v[:, 1].max())
    total_x = (x_max - x_min) * 1e-6
    # --- SYMMETRY: simulate a QUARTER domain --------------------------------
    # The device is mirror-symmetric about y=0 (grounded center, identical
    # ±V bodies on independent springs → no antisymmetric branch) and about
    # the Si-layer mid-plane (symmetric SiO2/Si/SiO2 stack, no substrate).
    # The fundamental guided mode is the EVEN TE mode → a PMC (magnetic
    # wall) on the y=0 and z-mid faces selects it; the other faces are PML.
    # Domain: y ∈ [0, +y_half], z ∈ [Si-mid, +z_half]  ⇒ ~4× cheaper FDTD,
    # ~2-4× cheaper EOM (Poisson/elasticity get a free Neumann symmetry at
    # the halved-grid edge; the elasticity DOF mask pins the normal
    # component on the symmetry planes).  Parity check: the quarter-model
    # transmission must match the full model — if it is ~0, flip PMC↔PEC.
    total_y = y_max * 1e-6 + 1.0e-6        # y: 0 (symmetry) → y_max + pad
    # Released span = x-range of the L10 movable bodies (#1/#2); the rib I/O
    # (L12) sits outside it and is the anchored region.
    l10_body = sorted(l10, key=lambda t: -t[1])[:2]  # the two equal-area bodies
    rel_x0 = min(float(v[:, 0].min()) for v, _ in l10_body) * 1e-6
    rel_x1 = max(float(v[:, 0].max()) for v, _ in l10_body) * 1e-6
    rel_len = rel_x1 - rel_x0
    # Extend the inverse-design x-extent past the spring bodies so the
    # pinned x-border band lands in the anchor / I-O-core margin rather than
    # freezing the spring flexures themselves (their x-edge == rel_x0/rel_x1).
    x_margin = 1.0e-6
    dev_len_x = rel_len + 2.0 * x_margin
    l10_y0 = min(float(v[:, 1].min()) for v, _ in l10) * 1e-6
    l10_y1 = max(float(v[:, 1].max()) for v, _ in l10) * 1e-6
    # y-symmetric ⇒ design only the y≥0 half; device y ∈ [0, l10_y1].
    dev_w = l10_y1

    t_si = 220e-9          # L10 device layer (full thickness)
    t_rib = 90e-9          # L12 rib slab (centred on the Si mid for z-symmetry)
    t_clad = 3.0e-6        # SiO2 above (anchored I/O only; no substrate)
    t_si_h = t_si / 2.0    # half Si layer (z ≥ Si-mid is simulated)
    total_z = t_si_h + t_clad                 # z: Si-mid (symmetry) → top
    rib_half = 0.5 * (total_x - dev_len_x)  # x-length of each anchored end region

    mat_si = fdtdx.Material(permittivity=PERMITTIVITY_SI_OPT)
    mat_sio2 = fdtdx.Material(permittivity=PERMITTIVITY_SIO2)
    mat_air = fdtdx.Material(permittivity=1.0)

    placement: list = []
    objs: list = []

    volume = fdtdx.SimulationVolume(partial_real_shape=(total_x, total_y, total_z))
    objs.append(volume)
    # PML on the open faces; PMC (magnetic wall) on the y=0 and z=Si-mid
    # symmetry planes to select the even fundamental TE mode.
    bcfg = fdtdx.BoundaryConfig(
        boundary_type_minx="pml", boundary_type_maxx="pml",
        boundary_type_miny="pmc", boundary_type_maxy="pml",
        boundary_type_minz="pmc", boundary_type_maxz="pml",
    )
    bdict, clist = fdtdx.boundary_objects_from_config(bcfg, volume)
    placement.extend(clist)
    objs.extend(list(bdict.values()))

    air_bg = fdtdx.UniformMaterialObject(
        name="air_bg", partial_real_shape=(None, None, None),
        material=mat_air, color=colors.XKCD_WHITE,
    )
    placement.extend([*air_bg.same_position_and_size(volume)])
    objs.append(air_bg)

    # SiO2 cladding ABOVE the device layer (the z<Si-mid half + its bottom
    # clad are mirrored away by the z-symmetry), ONLY in the two anchored
    # rib end regions — the central released span stays air.
    for tag, side in (("w", -1), ("e", 1)):
        clad = fdtdx.UniformMaterialObject(
            name=f"clad_{tag}_top",
            partial_real_shape=(rib_half, None, t_clad),
            material=mat_sio2, color=colors.XKCD_ORANGE,
        )
        placement.extend([
            clad.place_relative_to(volume, axes=0, own_positions=side, other_positions=side),
            clad.place_relative_to(volume, axes=2, own_positions=1, other_positions=1),
        ])
        objs.append(clad)

    # Static GDS parts, clipped to the y≥0 half and z-min-aligned (Si-mid on
    # the PMC plane).  ``ExtrudedPolygon`` is built directly from the
    # clipped vertices (centred on their own bbox); placed at the true
    # x-centre, y-min-aligned (the y=0 cut edge → symmetry wall), z-min-
    # aligned (Si-mid → symmetry wall), with half the physical thickness.
    def _add_static_poly(vtx_um, t_full, name):
        half = _half_poly_um(np.asarray(vtx_um))                 # (n,2) µm, y≥0
        cx_um = 0.5 * (float(half[:, 0].min()) + float(half[:, 0].max()))
        v_m = half * 1e-6
        centre = 0.5 * (v_m.min(axis=0) + v_m.max(axis=0))
        ep = fdtdx.ExtrudedPolygon(
            vertices=v_m - centre, axis=2, material_name="Si",
            materials={"air": mat_air, "Si": mat_si},
            partial_real_shape=(None, None, t_full / 2.0), name=name,
        )
        placement.extend([
            ep.place_relative_to(volume, axes=0, own_positions=0, other_positions=0,
                                 grid_margins=int(round(cx_um * 1e-6 / resolution))),
            ep.place_relative_to(volume, axes=1, own_positions=-1, other_positions=-1),
            ep.place_relative_to(volume, axes=2, own_positions=-1, other_positions=-1),
        ])
        objs.append(ep)

    # L12 rib slab (90 nm, centred on Si-mid) at both ends — the fixed I/O.
    for i, (vtx, _a) in enumerate(l12):
        _add_static_poly(vtx, t_rib, f"rib_{i}")
    # L10 #0 centre core over the FULL length (220 nm): with the 90 nm slab
    # this forms the real rib waveguide that launches/collects the guided
    # mode in the anchored I/O.  The Device is placed AFTER it and overrides
    # it inside the released span.
    _add_static_poly(l10[0][0], t_si, "rib_core")

    # Inverse-design Device: the whole L10 pattern over the released span
    # (center core + top/bottom spring bodies) co-designing optics+mechanics.
    bvs = resolution
    # Device matrix-voxel grid is deterministic from the sizing — compute it
    # up front so the "always-Si" mask can be built before construction.
    _nx = int(round(dev_len_x / bvs))
    _ny = int(round(dev_w / bvs))
    _xs = np.linspace((rel_x0 - x_margin) * 1e6, (rel_x1 + x_margin) * 1e6, _nx) \
        + 0.5 * (dev_len_x / _nx) * 1e6
    _ys = np.linspace(0.0, l10_y1 * 1e6, _ny) + 0.5 * (dev_w / _ny) * 1e6
    l10_mask = _rasterize([v for v, _ in l10], _xs, _ys)              # (nx, ny)
    # "Always Si": the structural anchors that must stay solid for the whole
    # optimization — the I/O-waveguide / core stubs at the x-ends and the
    # spring supports at the y-ends — i.e. the GDS-solid L10 inside a border
    # band (the same edges the mechanical clamp pins).  Without this the
    # smoothing+projection bleeds air into the north support.
    _bw = max(4, int(round(0.5e-6 / bvs)))                            # band ≈0.5µm
    _ix = np.arange(_nx)[:, None]
    _iy = np.arange(_ny)[None, :]
    # y=0 is the SYMMETRY plane (the grounded core, kept free/optimizable),
    # NOT an anchor — so the y-side pin is only the OUTER edge (the spring
    # N support); plus the x-end I/O-core stubs.
    _border = (_ix < _bw) | (_ix >= _nx - _bw) | (_iy >= _ny - _bw)
    always_si = jnp.asarray((l10_mask > 0.5) & _border)              # (nx, ny) bool
    logger.info(f"[always-Si mask] {int(np.asarray(always_si).sum())} of "
                f"{_nx * _ny} cells pinned solid (I/O + spring supports)")

    bridge = fdtdx.Device(
        name="phase_tuner",
        partial_real_shape=(dev_len_x, dev_w, t_si_h),  # y,z = the half-extents
        materials={"air": mat_air, "Si": mat_si},
        param_transforms=[
            fdtdx.GaussianSmoothing2D(std_discrete=2),
            fdtdx.SubpixelSmoothedProjection(),
            # Pinned LAST so it overrides the projected density.
            fdtdx.FixedMaterialMask(mask=always_si, value=1.0),
        ],
        partial_voxel_real_shape=(bvs, bvs, t_si_h),
    )
    placement.extend([
        bridge.place_at_center(volume, axes=(0,)),                 # x: centred
        bridge.place_relative_to(volume, axes=1, own_positions=-1, other_positions=-1),
        bridge.place_relative_to(volume, axes=2, own_positions=-1, other_positions=-1),
    ])
    objs.append(bridge)

    # ------------------------------------------------------------------
    # Source / detectors (rib mode in west, mode-overlap out east)
    # ------------------------------------------------------------------
    src = fdtdx.ModePlaneSource(
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, None, total_z),
        wave_character=fdtdx.WaveCharacter(wavelength=wavelength),
        direction="+", mode_index=0, filter_pol="te",
    )
    placement.extend([
        src.place_relative_to(volume, axes=0, own_positions=-1, other_positions=-1,
                              grid_margins=bcfg.thickness_grid_minx + 4),
        src.place_at_center(volume, axes=(1, 2)),
    ])
    objs.append(src)

    overlap = fdtdx.ModeOverlapDetector(
        name="overlap_out",
        partial_grid_shape=(1, None, None),
        partial_real_shape=(None, total_y, total_z),
        wave_characters=(fdtdx.WaveCharacter(wavelength=wavelength),),
        direction="+", mode_index=0, filter_pol="te",
        switch=fdtdx.OnOffSwitch(period=period, start_time=0.7 * config.time, on_for_periods=3),
    )
    placement.extend([
        overlap.place_relative_to(volume, axes=0, own_positions=1, other_positions=1,
                                  grid_margins=-(bcfg.thickness_grid_maxx + 4)),
        overlap.place_at_center(volume, axes=(1, 2)),
    ])
    objs.append(overlap)

    back_flux = fdtdx.PoyntingFluxDetector(
        name="back_flux", partial_grid_shape=(1, None, None), direction="-",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-2 * period_steps:]),
    )
    placement.append(back_flux.place_relative_to(
        volume, axes=0, own_positions=-1, other_positions=-1,
        grid_margins=bcfg.thickness_grid_minx + 2))
    objs.append(back_flux)
    fwd_flux = fdtdx.PoyntingFluxDetector(
        name="fwd_flux", partial_grid_shape=(1, None, None), direction="+",
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=all_time_steps[-2 * period_steps:]),
    )
    placement.append(fwd_flux.place_relative_to(
        volume, axes=0, own_positions=-1, other_positions=-1,
        grid_margins=bcfg.thickness_grid_minx + 6))
    objs.append(fwd_flux)

    optical_energy = fdtdx.EnergyDetector(
        name="optical_energy", as_slices=True,
        switch=fdtdx.OnOffSwitch(fixed_on_time_steps=[-1]),
    )
    placement.extend([*optical_energy.same_position_and_size(volume)])
    objs.append(optical_energy)

    # ------------------------------------------------------------------
    # Place on grid
    # ------------------------------------------------------------------
    key, sk = jax.random.split(key)
    objects, arrays, params, config, _ = fdtdx.place_objects(
        object_list=objs, config=config, constraints=placement, key=sk
    )

    dev = next(d for d in objects.devices if d.name == "phase_tuner")
    dev_grid = (int(dev.matrix_voxel_grid_shape[0]),
                int(dev.matrix_voxel_grid_shape[1]),
                int(dev.matrix_voxel_grid_shape[2]))
    dev_slice = dev.grid_slice
    dev_gpv = (int(dev.single_voxel_grid_shape[0]),
               int(dev.single_voxel_grid_shape[1]),
               int(dev.single_voxel_grid_shape[2]))
    logger.info(f"[device] grid={dev_grid} gpv={dev_gpv} "
                f"slice={[(s.start, s.stop) for s in dev_slice]}")

    # Seed the Device ρ from the rasterized GDS L10 pattern (precomputed
    # above; the placed matrix grid must match the sizing-derived shape).
    assert dev_grid[:2] == (_nx, _ny), f"grid {dev_grid[:2]} != ({_nx},{_ny})"
    seed_rho = jnp.asarray(l10_mask[:, :, None], dtype=jnp.float32)
    if isinstance(params, dict):
        params = {**params, "phase_tuner": seed_rho}
    else:
        raise RuntimeError("expected dict params from place_objects")

    # ------------------------------------------------------------------
    # Through-thickness refinement for the (Q1) elasticity element + the
    # 3-D fringe electrostatics (the optical Device is 1 voxel thick in z).
    # ------------------------------------------------------------------
    # Mech model: FULL y+z symmetric quarter (the user-requested config;
    # validated — its fundamental eig matches the mirrored-full model
    # within ~5 %).  Half Si thickness (z ≥ Si-mid), ≥3 cells for the Q1
    # bending element.  NB: the resulting pull-in voltage of the *un-
    # optimized* GDS draft is legitimately high (stiff folded Si springs,
    # narrow slot) — the optimization reduces it; V_on tracks 0.9·V_pi
    # every epoch.
    nz_th = max(3, int(round(t_si_h / bvs)))
    mech_grid = (dev_grid[0], dev_grid[1], nz_th)
    mech_vox = (dev_len_x / mech_grid[0], dev_w / mech_grid[1], t_si_h / nz_th)

    def _thicken(rho2d):
        r = rho2d.reshape(dev_grid[0], dev_grid[1])
        return jnp.broadcast_to(r[:, :, None], (dev_grid[0], dev_grid[1], nz_th)).astype(jnp.float32)

    # The device is THREE mechanically independent bodies: the grounded
    # centre core (anchored E/W into the ribs — a rigid reference, it does
    # NOT vibrate) and the top/bottom slot bodies on folded springs.
    # Solving them as one continuum coupled through the SIMP air-floor
    # returns polluted GLOBAL modes → wrong effective stiffness → V_pi
    # ~100× too high.  Fix: CLAMP the centre-core band so it is rigid; the
    # free elastic domain is then exactly the top body + its springs,
    # anchored at the N support (y-max).  With y-symmetry this is "the top
    # body's mode with the N edge fixed" — the user's prescription; the
    # bottom is its mirror image (handled by the y=0 symmetry plane).
    _core_band = max(1, int(round(0.32e-6 / (dev_w / dev_grid[1]))))  # ≈ core y-extent
    mech_mask = fdtdx.build_dof_mask(
        mech_grid,
        clamped_regions=[
            (slice(0, 1), slice(None), slice(None)),       # x-min anchor
            (slice(-1, None), slice(None), slice(None)),   # x-max anchor
            (slice(None), slice(-1, None), slice(None)),    # y-max spring support
            (slice(None), slice(0, _core_band + 1), slice(None)),  # rigid centre core
        ],
        symmetry_planes=[
            (2, slice(None), slice(None), slice(0, 1)),      # z=0 Si-mid (u_z)
        ],
    )
    # Mech eigensolve on the Si subdomain of ρ (element-level mask: only
    # Q1 elements with all 8 corners satisfying ρ>τ contribute strain
    # energy, air nodes pinned).  rho_mask_threshold=0.5 is the natural
    # cut for post-β-projection densities; FixedMaterialMask-pinned I/O
    # and spring-support bands at ρ=1 are correctly classified Si.
    mech = fdtdx.ElasticityEigenmodes(
        young_modulus=SI_YOUNG_MODULUS, poisson_ratio=0.27, density=SI_DENSITY,
        voxel_size_m=mech_vox, free_dof_mask=mech_mask,
        rho_mask_threshold=0.5,
        n_modes=2, n_subspace=6, n_subspace_iter=14,
        cg_iterations=150, cg_tol=1e-6,  # cold (no analytic warm-start)
    )

    # ------------------------------------------------------------------
    # 3-D fringe electrostatics on the y-MIRRORED full ρ, FULL-z (the
    # proven config that gave V_pi≈15 V): core-centred ground band,
    # body/electrode outer bands, device z centred with air both sides.
    # Keeping this auxiliary Poisson at the full y+z geometry makes its
    # force / the f0 calibration consistent with the full-thickness mech
    # `peak0` and the full-thickness pull-in cross-section.  The y≥0 half
    # of the force feeds the (validated) y-symmetric half mech model.
    # ------------------------------------------------------------------
    z_air = max(3, int(round(250e-9 / bvs)))
    _ny_h = dev_grid[1]
    _ny_f = 2 * _ny_h
    eom_shape = (dev_grid[0], _ny_f, nz_th + z_air)   # z: Si-mid at z=0 (sym)
    _z0, _z1 = 0, nz_th
    _yc = _ny_f // 2                                # core centre of full-y grid
    core_half = max(1, int(round(0.30e-6 / (dev_w / _ny_h))))
    gmask = np.zeros(eom_shape, bool)
    emask = np.zeros(eom_shape, bool)
    gmask[:, _yc - core_half: _yc + core_half + 1, :] = True   # core/ground
    emask[:, : _yc - core_half, :] = True                      # body/electrode
    emask[:, _yc + core_half + 1:, :] = True
    poisson = fdtdx.PoissonSolver(
        electrode_mask=jnp.asarray(emask), ground_mask=jnp.asarray(gmask),
        eps_min=EPS_AIR_DC, eps_max=EPS_SI_DC,
        cg_iterations=400, cg_tol=1e-6,
        voxel_size_m=(bvs, bvs, bvs),
    )

    def _assemble_eom(rho2d):
        r2 = rho2d.reshape(dev_grid[0], _ny_h)
        rho_full = jnp.concatenate([r2[:, ::-1], r2], axis=1)   # mirror y
        rt = jnp.broadcast_to(rho_full[:, :, None],
                              (dev_grid[0], _ny_f, nz_th)).astype(jnp.float32)
        r = jnp.zeros(eom_shape, jnp.float32)
        return r.at[:, :, _z0:_z1].set(rt)

    def _force_to_mech(F):
        # Full-y Poisson force → the y≥0 half (matches the half mech grid),
        # device z-cells.  F: (3, nx, _ny_f, eom_nz) → (3, nx, _ny_h, nz_th).
        return F[:, :, _yc:, _z0:_z1]

    voxel_size_m = (resolution, resolution, resolution)
    _sim = arrays.inv_permittivities.shape[-3:]
    sim_shape = (int(_sim[0]), int(_sim[1]), int(_sim[2]))

    # ------------------------------------------------------------------
    # Per-epoch pull-in voltage (jittable): displacement-controlled
    # limit-point continuation on a refined y–z cross-section, finite-gap
    # force from the capacitance/co-energy (∫|∇φ|²), moving body advected
    # with ``fdtdx.advect_density`` (differentiable).  Calibrated to the
    # validated 3-D modal stiffness via peak0.  Recomputed every epoch from
    # the current ρ ⇒ V_on = 0.9·V_pi(ρ),  V_off = V_on + 1.
    # ------------------------------------------------------------------
    # NB: the pull-in cross-section is a geometry-DECOUPLED reduced y–z
    # capacitance model (a separate small auxiliary solve, not the FDTD or
    # the main mech eigensolve).  It must use the FULL device width /
    # thickness — the symmetry halving of `dev_w`/`t_si` for the main solve
    # must NOT shrink this reference, or k_gen = f0/peak0 is mis-calibrated
    # (observed: V_pi → ~940 V when this used the halved dev_w).
    dev_w_full = 2.0 * dev_w     # full lateral extent (dev_w is the y-half)
    g_near_m = 0.30e-6           # nominal center↔body slot (from the GDS)
    xs_pitch = g_near_m / 16.0
    nn = lambda L: max(1, int(round(L / xs_pitch)))  # noqa: E731
    Ny = 1 + nn(0.4e-6) + nn(dev_w_full) + nn(g_near_m) + 1
    Nz = nn(250e-9) + nn(t_si) + nn(250e-9)
    yb0 = 1 + nn(0.4e-6)
    yb1 = yb0 + nn(dev_w_full)
    zb0 = nn(250e-9)
    zb1 = zb0 + nn(t_si)
    _xs_elec = np.zeros((Ny, Nz), bool); _xs_elec[-1, :] = True
    _xs_gnd = np.zeros((Ny, Nz), bool); _xs_gnd[0, :] = True
    poisson_xs = fdtdx.PoissonSolver(
        electrode_mask=jnp.asarray(_xs_elec), ground_mask=jnp.asarray(_xs_gnd),
        eps_min=EPS_AIR_DC, eps_max=EPS_SI_DC, cg_iterations=400, cg_tol=1e-7,
        voxel_size_m=(xs_pitch, xs_pitch, xs_pitch),
    )
    _xs_block = jnp.zeros((Ny, Nz), jnp.float32).at[yb0:yb1, zb0:zb1].set(1.0)
    _n_pi = 14
    _s_pi = jnp.linspace(0.9 * g_near_m / _n_pi, 0.9 * g_near_m, _n_pi, dtype=jnp.float32)

    def _eom_branch(rho2d):
        """Static branch of the reduced actuator for the current ρ.

        Returns ``(v_pi, s_grid, v_branch, lin_dir, peak0)`` where
        ``lin_dir`` is the unit-peak modal deflection *shape* (3, nx, ny,
        nz_th) and ``v_branch = √(k·s/f(s))`` is the displacement-controlled
        V(s) curve from the finite-gap capacitance continuation.
        """
        rho2d = jnp.clip(rho2d, 0.0, 1.0).astype(jnp.float32)
        phi0 = poisson.solve(_assemble_eom(rho2d), jnp.asarray(1.0, jnp.float32))
        u0 = mech.equilibrium_displacement(
            _thicken(rho2d), _force_to_mech(poisson.force_field(phi0, _assemble_eom(rho2d)))
        )
        peak0 = jnp.maximum(jnp.max(jnp.abs(u0[1])), 1e-18)
        lin_dir = u0 / peak0                                 # unit-peak shape

        def cap(s):
            disp = jnp.stack(
                [(s / xs_pitch) * _xs_block, jnp.zeros((Ny, Nz), jnp.float32)], axis=0
            )
            rs = fdtdx.advect_density(
                _xs_block * float(EPS_SI_DC), disp, order=1
            ) / float(EPS_SI_DC)
            phi = poisson_xs.solve(jnp.clip(rs, 0.0, 1.0), jnp.asarray(1.0, jnp.float32))
            gy, gz = jnp.gradient(phi)
            return jnp.sum(gy ** 2 + gz ** 2)                # ∝ C(s) at V=1
        C = jax.vmap(cap)(_s_pi)
        f = jnp.where(0.5 * jnp.gradient(C) > 0, 0.5 * jnp.gradient(C), 1e-30)
        k_gen = f[0] / peak0
        v_pi, v_branch = fdtdx.pull_in_voltage(_s_pi, k_gen, f)
        v_pi = jnp.where(jnp.isfinite(v_pi) & (v_pi > 0), v_pi, _V_PI_FALLBACK)
        return v_pi, _s_pi, v_branch, lin_dir, peak0

    def _pull_in_voltage(rho2d):
        return _eom_branch(rho2d)[0]

    def _saturated_disp(rho2d, voltage):
        """Physically-saturated deflection field at ``voltage``.

        The LINEAR modal model (u ∝ V²) has no gap-narrowing saturation and,
        at V_on = 0.9·V_pi (near pull-in *by design*), overshoots the true
        deflection ~5× → a >voxel displacement that makes the first-order
        ε-perturbation diverge.  Instead read the equilibrium amplitude off
        the continuation's stable branch:  s_eq solves V(s)=voltage on the
        rising part (clamped at the limit point), and the displacement field
        is ``s_eq · lin_dir`` (the modal shape scaled to the *saturated*
        amplitude).
        """
        _, s_grid, v_branch, lin_dir, _ = _eom_branch(rho2d)
        i_max = jnp.argmax(v_branch)                          # limit point
        idx = jnp.arange(v_branch.shape[0])
        stable = idx <= i_max                                 # rising branch
        # monotone-increasing V on the stable branch → invert by interp.
        vb = jnp.where(stable, v_branch, v_branch[i_max] + idx * 1e-9)
        s_eq = jnp.interp(jnp.asarray(voltage, jnp.float32), vb, s_grid)
        s_eq = jnp.clip(s_eq, 0.0, s_grid[i_max])             # cap at pull-in
        return s_eq * lin_dir                                 # (3, nx, ny, nz_th)

    # ------------------------------------------------------------------
    # EOM forward at one voltage (saturated deflection → ε-perturb → FDTD).
    # ------------------------------------------------------------------
    def _run_at_voltage(params, arrays, objects, key, beta, voltage):
        arrays_b, new_objs, _ = fdtdx.apply_params(arrays, objects, params, key, beta=beta)
        nd = next(d for d in new_objs.devices if d.name == "phase_tuner")
        rho = jnp.clip(nd(params["phase_tuner"], beta=beta), 0.0, 1.0).astype(jnp.float32)
        u = _saturated_disp(rho, voltage)                  # bounded (≤ pull-in)
        u_dev = jnp.mean(u, axis=3, keepdims=True)         # (3, nx, ny, 1)
        inv_eps = fdtdx.apply_finite_displacement_to_permittivity(
            inv_permittivities_sim=arrays_b.inv_permittivities,
            displacement_design=u_dev, grid_points_per_voxel=dev_gpv,
            sim_grid_shape=sim_shape, sim_slice=dev_slice, voxel_size_m=voxel_size_m,
        )
        _, af = fdtdx.run_fdtd(
            arrays=arrays_b.aset("inv_permittivities", inv_eps),
            objects=new_objs, config=config, key=key,
        )
        return af

    def _beta(epoch):
        return 0.1 + (50.0 - 0.1) * jnp.clip(epoch / float(epochs * 0.9), 0.0, 1.0)

    def simulate_fn(params, arrays, objects, config, key, epoch):
        del config
        beta = _beta(epoch)
        nd = next(d for d in objects.devices if d.name == "phase_tuner")
        rho = jnp.clip(nd(params["phase_tuner"], beta=beta), 0.0, 1.0).astype(jnp.float32)
        v_pi = jax.lax.stop_gradient(_pull_in_voltage(rho))
        v_on = _PULLIN_SAFETY * v_pi
        v_off = v_on + V_SWING
        k1, k2 = jax.random.split(key)
        a_on = _run_at_voltage(params, arrays, objects, k1, beta, v_on)
        a_off = _run_at_voltage(params, arrays, objects, k2, beta, v_off)
        merged = {}
        for k in _BASE_DETECTORS:
            merged[k] = a_on.detector_states[k]
            merged[k + "__off"] = a_off.detector_states[k]
        out = a_on.aset("detector_states", merged)
        return out

    # Pre-register __off detector keys so the threaded pytree is epoch-stable.
    _ds = dict(arrays.detector_states)
    for _k in _BASE_DETECTORS:
        _ds[_k + "__off"] = jax.tree_util.tree_map(jnp.zeros_like, _ds[_k])
    arrays = arrays.aset("detector_states", _ds)

    # Reference output mode (geometry-fixed rib cross-section → solve once).
    _ov = next(d for d in objects.detectors if d.name == "overlap_out")
    assert isinstance(_ov, fdtdx.ModeOverlapDetector)
    overlap_p: fdtdx.ModeOverlapDetector = _ov.apply(
        key=key, inv_permittivities=arrays.inv_permittivities,
        inv_permeabilities=arrays.inv_permeabilities,
    )

    # ------------------------------------------------------------------
    # Objective metrics
    # ------------------------------------------------------------------
    def _insertion_loss(*, arrays, **_):
        a1 = overlap_p.compute_overlap(arrays.detector_states["overlap_out"])
        a2 = overlap_p.compute_overlap(arrays.detector_states["overlap_out__off"])
        t = 0.5 * (jnp.abs(a1) ** 2 + jnp.abs(a2) ** 2)     # mean transmission
        return jnp.sqrt(jnp.clip(t, 1e-12, None)), {"abs_on": jnp.abs(a1), "abs_off": jnp.abs(a2)}

    def _back_reflection(*, arrays, **_):
        def frac(suf):
            b = jnp.maximum(arrays.detector_states[f"back_flux{suf}"]["poynting_flux"].sum(), 0.0)
            fwd = jnp.maximum(arrays.detector_states[f"fwd_flux{suf}"]["poynting_flux"].sum(), 0.0)
            return b / (b + fwd + 1e-12)
        return 0.5 * (frac("") + frac("__off")), {}

    def _footprint(*, params, objects, epoch, **_):
        nd = next(d for d in objects.devices if d.name == "phase_tuner")
        rho = jnp.clip(nd(params["phase_tuner"], beta=_beta(epoch)), 0.0, 1.0)
        rho2 = rho.reshape(dev_grid[0], dev_grid[1])
        m = jnp.clip(jnp.sum(rho2), 1e-6, None)
        ix = jnp.arange(dev_grid[0], dtype=jnp.float32)[:, None]
        iy = jnp.arange(dev_grid[1], dtype=jnp.float32)[None, :]
        xb = jnp.sum(rho2 * ix) / m
        yb = jnp.sum(rho2 * iy) / m
        # Effective extent = √(12·Var) (= full width for a uniform block),
        # normalized by the region size → minimize the x AND y footprint.
        ex = jnp.sqrt(12.0 * jnp.sum(rho2 * (ix - xb) ** 2) / m) / dev_grid[0]
        ey = jnp.sqrt(12.0 * jnp.sum(rho2 * (iy - yb) ** 2) / m) / dev_grid[1]
        return ex * ey, {"ext_x": ex, "ext_y": ey}

    epochs = int(os.environ.get("PT_EPOCHS", "200"))
    _fab0, _fab1 = round(0.4 * epochs), round(0.95 * epochs)
    objectives = (
        fdtdx.FunctionObjective(name="insertion", schedule=fdtdx.ConstantSchedule(value=1.0),
                                fn=_insertion_loss),
    )
    constraints: list = [
        fdtdx.FunctionConstraint(name="back_refl", schedule=fdtdx.ConstantSchedule(value=0.3),
                                 fn=_back_reflection),
        fdtdx.FunctionConstraint(
            name="footprint",
            schedule=fdtdx.LinearSchedule(epoch_start=_fab0, epoch_end=_fab1,
                                          start_value=0.0, end_value=0.5),
            fn=_footprint),
        fdtdx.MinLineSpace(
            name="line_space", device_name="phase_tuner",
            min_line_width_m=140e-9, min_space_m=140e-9,
            schedule=fdtdx.LinearSchedule(epoch_start=_fab0, epoch_end=_fab1,
                                          start_value=0.0, end_value=1.0)),
    ]

    # ------------------------------------------------------------------
    # Setup figure + device snapshot (both branches)
    # ------------------------------------------------------------------
    exp_logger.savefig(exp_logger.cwd, "setup.png",
                       fdtdx.plot_setup(config=config, objects=objects,
                                        exclude_object_list=[air_bg, optical_energy]))
    exp_logger.log_params(iter_idx=-1, params=params, objects=objects,
                          export_stl=True, export_figure=True,
                          beta=_beta(jnp.asarray(0.0, jnp.float32)))

    # Fast geometry-only check: PT_GEOM_ONLY=1 writes setup.png + the seeded
    # device figure then exits before any FDTD/EOM compile (seconds, not
    # minutes) — used to verify GDS placement without the heavy solve.
    if os.environ.get("PT_GEOM_ONLY") == "1":
        logger.info(f"[geom-only] setup + device figures written to {exp_logger.cwd}")
        return

    # ------------------------------------------------------------------
    # Field maps for the status page (no FDTD): φ/Δε/u at a given V.
    # ------------------------------------------------------------------
    def _field_maps(params, arrays, objects, key, voltage):
        ab, no, _ = fdtdx.apply_params(arrays, objects, params, key, beta=50.0)
        nd = next(d for d in no.devices if d.name == "phase_tuner")
        rho = jnp.clip(nd(params["phase_tuner"], beta=50.0), 0.0, 1.0).astype(jnp.float32)
        u = _saturated_disp(rho, voltage)
        u_dev = jnp.mean(u, axis=3, keepdims=True)
        inv_eps = fdtdx.apply_finite_displacement_to_permittivity(
            inv_permittivities_sim=ab.inv_permittivities, displacement_design=u_dev,
            grid_points_per_voxel=dev_gpv, sim_grid_shape=sim_shape,
            sim_slice=dev_slice, voxel_size_m=voxel_size_m)
        eb = 1.0 / jnp.clip(ab.inv_permittivities[0], 1e-30, None)
        ed = 1.0 / jnp.clip(inv_eps[0], 1e-30, None)
        deps = jnp.mean((ed - eb)[dev_slice], axis=2)
        return {"deps": deps, "uy": u_dev[1, :, :, 0],
                "rho": rho.reshape(dev_grid[0], dev_grid[1])}

    # ------------------------------------------------------------------
    # Live status callback (Optimization.epoch_callback)
    # ------------------------------------------------------------------
    _hist: dict[str, list] = {k: [] for k in ("loss", "insertion", "back_refl", "footprint")}

    def _status_cb(*, epoch, params, objects, arrays, info, optimization):
        for k, ik in (("loss", "loss"), ("insertion", "insertion_raw"),
                      ("back_refl", "back_refl_raw"), ("footprint", "footprint_raw")):
            if ik in info:
                _hist[k].append(float(info[ik]))
        kk = jax.random.PRNGKey(epoch)
        v_pi = float(_pull_in_voltage(
            jnp.clip(next(d for d in objects.devices if d.name == "phase_tuner")(
                params["phase_tuner"], beta=float(_beta(jnp.asarray(epoch, jnp.float32)))),
                0.0, 1.0)))
        v_on = _PULLIN_SAFETY * v_pi
        fj = jax.jit(lambda p, a, k, v: _field_maps(p, a, objects, k, v))
        m_on = fj(params, arrays, kk, jnp.asarray(v_on, jnp.float32))
        m_off = fj(params, arrays, kk, jnp.asarray(v_on + V_SWING, jnp.float32))
        st = arrays.detector_states.get("optical_energy", {})
        kx = next((q for q in st if "XZ" in q), None)
        opt = np.asarray(st[kx])[-1] if kx is not None else np.zeros((4, 4))

        # Only the y≥0 half is simulated (y-symmetry); mirror the x–y panels
        # about y=0 so the full symmetric device is shown.  u_y flips sign
        # across the mirror (odd component).
        def _mir(a, odd=False):
            a = np.asarray(a)
            other = -a[:, ::-1] if odd else a[:, ::-1]
            return np.concatenate([other, a], axis=1)

        panels = {
            "rho": _mir(m_on["rho"]),
            "optical": opt,
            "deps_on": _mir(m_on["deps"]), "deps_off": _mir(m_off["deps"]),
            "uy_on": _mir(m_on["uy"], odd=True), "uy_off": _mir(m_off["uy"], odd=True),
        }
        stats = {
            "epoch": int(epoch), "V_pull_in": f"{v_pi:.3g} V",
            "V_on": f"{v_on:.3g} V", "V_off": f"{v_on + V_SWING:.3g} V",
            "loss": f"{float(info.get('loss', np.nan)):.4g}",
            "insertion_raw": f"{float(info.get('insertion_raw', np.nan)):.4g}",
            "back_refl_raw": f"{float(info.get('back_refl_raw', np.nan)):.4g}",
            "footprint_raw": f"{float(info.get('footprint_raw', np.nan)):.4g}",
            "grad_norm": f"{float(info.get('grad_norm', np.nan)):.3g}",
            "abs_on": f"{float(info.get('abs_on', np.nan)):.3g}",
            "abs_off": f"{float(info.get('abs_off', np.nan)):.3g}",
            "runtime_s": f"{float(info.get('runtime', np.nan)):.1f}",
        }
        d = exp_logger.cwd
        np.savez(d / "status_panels.npz", **panels)
        np.savez(d / "status_history.npz", **{k: np.asarray(v) for k, v in _hist.items()})
        import json
        (d / "status.json").write_text(json.dumps(stats))
        render_status_figure(panels, _hist, stats, d / "status.png")

    # ------------------------------------------------------------------
    # Fast NaN-localization (PT_DIAG=1): no FDTD — checks each EOM stage and
    # the reference mode for finiteness so the optical NaN can be pinned in
    # ~30 s instead of a 13 min FDTD cycle.
    # ------------------------------------------------------------------
    if os.environ.get("PT_DIAG") == "1":
        nd = next(d for d in objects.devices if d.name == "phase_tuner")
        rho = jnp.clip(nd(params["phase_tuner"], beta=50.0), 0.0, 1.0).astype(jnp.float32)
        chk = {}
        chk["rho_finite"] = bool(jnp.all(jnp.isfinite(rho)))
        chk["mode_E_finite"] = bool(jnp.all(jnp.isfinite(overlap_p._mode_E)))
        chk["mode_H_finite"] = bool(jnp.all(jnp.isfinite(overlap_p._mode_H)))
        chk["mode_E_absmax"] = float(jnp.max(jnp.abs(overlap_p._mode_E)))
        vpi = _pull_in_voltage(rho)
        chk["v_pi"] = float(vpi)
        chk["v_pi_finite"] = bool(jnp.isfinite(vpi))
        v_on = _PULLIN_SAFETY * float(vpi)
        eom = _assemble_eom(rho)
        phi = poisson.solve(eom, jnp.asarray(v_on, jnp.float32))
        chk["phi_finite"] = bool(jnp.all(jnp.isfinite(phi)))
        chk["phi_absmax"] = float(jnp.max(jnp.abs(phi)))
        F = _force_to_mech(poisson.force_field(phi, eom))
        chk["force_finite"] = bool(jnp.all(jnp.isfinite(F)))
        chk["force_absmax"] = float(jnp.max(jnp.abs(F)))

        # Mech fundamental of the (core-clamped) body-on-springs mode.
        ev_h, _mh = mech.compute_modes(_thicken(rho))
        chk["mech_eig0"] = float(ev_h[0])
        chk["mech_f0_MHz"] = float(jnp.sqrt(jnp.maximum(ev_h[0], 0.0)) / (2.0 * jnp.pi) / 1e6)

        u_lin = mech.equilibrium_displacement(_thicken(rho), F)
        chk["u_LINEAR_absmax_m"] = float(jnp.max(jnp.abs(u_lin)))
        u = _saturated_disp(rho, jnp.asarray(v_on, jnp.float32))
        chk["u_finite"] = bool(jnp.all(jnp.isfinite(u)))
        chk["u_saturated_absmax_m"] = float(jnp.max(jnp.abs(u)))
        u_dev = jnp.mean(u, axis=3, keepdims=True)
        ab, no, _ = fdtdx.apply_params(arrays, objects, params,
                                       jax.random.PRNGKey(0), beta=50.0)
        inv_eps = fdtdx.apply_finite_displacement_to_permittivity(
            inv_permittivities_sim=ab.inv_permittivities, displacement_design=u_dev,
            grid_points_per_voxel=dev_gpv, sim_grid_shape=sim_shape,
            sim_slice=dev_slice, voxel_size_m=voxel_size_m)
        chk["inv_eps_base_finite"] = bool(jnp.all(jnp.isfinite(ab.inv_permittivities)))
        chk["inv_eps_def_finite"] = bool(jnp.all(jnp.isfinite(inv_eps)))
        chk["inv_eps_def_min"] = float(jnp.min(inv_eps))
        chk["inv_eps_def_absmax"] = float(jnp.max(jnp.abs(inv_eps)))
        logger.info("[PT_DIAG] " + ", ".join(f"{k}={v}" for k, v in chk.items()))
        return

    # ------------------------------------------------------------------
    # Eval branch
    # ------------------------------------------------------------------
    if evaluation:
        if seed_from is not None:
            params = cast(dict, fdtdx.load_seed_params(
                Path(seed_from), params,
                iter_idx=(None if seed_iter in (None, "latest", "auto", "") else int(seed_iter))))
        t0 = time.time()
        jitted = jax.jit(lambda p, a, k, e: simulate_fn(p, a, objects, config, k, e)).lower(
            params, arrays, key, jnp.asarray(0.0, jnp.float32)).compile()
        logger.info(f"compile {time.time() - t0:.1f}s")
        key, sk = jax.random.split(key)
        t0 = time.time()
        af = jitted(params, arrays, sk, jnp.asarray(0.0))
        logger.info(f"run {time.time() - t0:.1f}s")
        nd = next(d for d in objects.devices if d.name == "phase_tuner")
        rho0 = jnp.clip(nd(params["phase_tuner"], beta=50.0), 0.0, 1.0)
        v_pi = float(_pull_in_voltage(rho0))
        il, ili = _insertion_loss(arrays=af)
        br, _ = _back_reflection(arrays=af)
        fp, fpi = _footprint(params=params, objects=objects, epoch=jnp.asarray(0.0))
        logger.info(
            "[eval] " + ", ".join(f"{k}={v:.5g}" for k, v in {
                "V_pull_in": v_pi, "V_on": _PULLIN_SAFETY * v_pi,
                "V_off": _PULLIN_SAFETY * v_pi + V_SWING,
                "insertion": float(il), "abs_on": float(ili["abs_on"]),
                "abs_off": float(ili["abs_off"]), "back_refl": float(br),
                "footprint": float(fp), "ext_x": float(fpi["ext_x"]),
                "ext_y": float(fpi["ext_y"]),
            }.items()))
        _status_cb(epoch=0, params=params, objects=objects, arrays=af,
                   info={"loss": float(il), "insertion_raw": float(il),
                         "back_refl_raw": float(br), "footprint_raw": float(fp),
                         "grad_norm": 0.0, "abs_on": float(ili["abs_on"]),
                         "abs_off": float(ili["abs_off"]), "runtime": 0.0},
                   optimization=None)
        logger.info(f"status.png + setup.png written to {exp_logger.cwd}")
        return

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    sched = optax.warmup_cosine_decay_schedule(
        init_value=1e-5, peak_value=2e-3, end_value=2e-4,
        warmup_steps=10, decay_steps=round(0.9 * epochs))
    optimizer = optax.inject_hyperparams(optax.nadam)(learning_rate=sched)
    opt = fdtdx.Optimization(
        objects=objects, arrays=arrays, params=params, config=config,
        simulate_fn=simulate_fn, optimizer=optimizer,
        objectives=objectives, constraints=tuple(constraints),
        total_epochs=epochs, param_clip=(0.0, 1.0), logger=exp_logger,
        log_every=1, checkpoint_every=25, epoch_callback=_status_cb,
    )
    rsi = None if seed_iter in (None, "latest", "auto", "") else int(seed_iter)
    key, rk = jax.random.split(key)
    opt.run(key=rk, seed_from=seed_from, seed_iter=rsi, resume_from=resume_from)


if __name__ == "__main__":
    parser = fdtdx.build_arg_parser(description="GDS dual-slot MEMS phase-tuner co-design")
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        argv = ["--seed-rng", sys.argv[1]]
        if len(sys.argv) > 2 and sys.argv[2].lower() in ("true", "1", "eval"):
            argv.append("--evaluation")
        a = parser.parse_args(argv)
    else:
        a = parser.parse_args()
    main(a)
