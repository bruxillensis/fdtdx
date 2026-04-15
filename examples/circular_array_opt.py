"""
circular_array_opt.py — Circular OPA placement and routing optimizer

Two-phase approach
------------------
Phase 1 — Emitter placement:
  Optimise per-ring angular offsets (and optionally per-emitter nudges) to
  maximise the minimum clearance between straight-radial routing channels and
  emitter bodies.  Uses differential evolution.

Phase 2 — Passage-window routing:
  For each emitter waveguide, compute waypoints at every outer ring boundary.
  At each boundary the waveguide must fit through a "passage window" (gap
  between emitters wide enough for ww + 2·clr).  If the straight-radial angle
  already clears the emitters, no bend occurs.  Otherwise the waveguide bends
  the minimum amount to reach the nearest window edge.
  Waypoints are fed to gf.path.smooth (Euler bends) for smooth GDS geometry.

Clearance model
---------------
  channel-emitter: centre_sep ≥ ww/2 + ed/2 + clr  (at each outer ring radius)
  channel-channel: centre_sep ≥ ww + clr

The theoretical maximum achievable clearance with any offset for ring pair (i,j)
is π/lcm(n_i,n_j) × r_j − required.  Pairs where this is negative are flagged
as infeasible for straight routing; Phase 2 bends compensate for them.

Hard constraint (independent of offsets):
    arc gap = dr − ed  ≥  ww + 2·clr
    ⇒  dr  ≥  ed + ww + 2·clr   (= 0.9 μm for defaults)
"""

from __future__ import annotations

import math
from math import gcd
import numpy as np
from scipy.optimize import differential_evolution
import gdsfactory as gf

gf.gpdk.PDK.activate()

# ═══════════════════════════════════════════════════════════════
# ❶  User parameters
# ═══════════════════════════════════════════════════════════════

DR      = 5      # radial + arc pitch (μm)
R0      = DR / 2   # first-ring radius (μm)  — DR/2 puts ring 0 at half-pitch
N_RINGS = 10        # number of concentric rings

WW  = 0.20   # routing waveguide width (μm)
ED  = 0.50   # emitter disk diameter (μm)
CLR = 0.25   # minimum edge-to-edge clearance (μm)

# Round n_per_ring to multiples of N_BASE.
# Larger N_BASE → smaller lcm(n_i, n_j) → wider routing gaps → easier to route.
# 1 = disabled; 4 or 6 are useful values that preserve rotational symmetry.
N_BASE = 1

# ── Optional second-stage: nudge each emitter ±DELTA_MAX degrees ──
OPTIMIZE_POSITIONS = True
DELTA_MAX_DEG      = 3.0

# ── Output ────────────────────────────────────────────────────────
ROUTE_EXTRA = 5.0   # routing waveguide extends this far beyond the outermost ring (μm)
GDS_OUT     = "circular_array_opt.gds"
GRID        = 0.001  # snap grid (μm)

# Minimum bend radius for Euler-bend routes (μm).
# Smaller = tighter bends allowed; increase for lower insertion loss.
BEND_RADIUS = 10.0

# ── DE solver settings ────────────────────────────────────────────
DE_SEED     = 42
DE_MAXITER  = 1000
DE_POPSIZE  = 20

# ═══════════════════════════════════════════════════════════════
# ❷  Geometry helpers
# ═══════════════════════════════════════════════════════════════

def snap(v: np.ndarray | float) -> np.ndarray:
    return np.round(np.asarray(v, float) / GRID) * GRID


def ring_geometry(r0: float, dr: float, n_rings: int,
                  n_base: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (radii, n_per_ring).

    n_base > 1 rounds n_per_ring to the nearest multiple of n_base so that
    gcd(n_i, n_j) ≥ n_base, reducing lcm and widening routing gaps.
    """
    radii = r0 + np.arange(n_rings, dtype=float) * dr
    n_raw = np.round(2 * np.pi * radii / dr)
    if n_base > 1:
        n_raw = np.round(n_raw / n_base) * n_base
    n_per_ring = np.maximum(max(n_base, 3), n_raw).astype(int)
    return radii, n_per_ring


def ring_angles(n: int, offset: float) -> np.ndarray:
    """Uniformly spaced angles on one ring: 2π k/n + offset, k = 0..n−1."""
    return (2 * np.pi / n) * np.arange(int(n)) + offset


# ═══════════════════════════════════════════════════════════════
# ❸  Clearance computation
# ═══════════════════════════════════════════════════════════════

def min_circ_sep(a: np.ndarray, b: np.ndarray) -> float:
    """
    Minimum circular angular separation (rad) between any element of `a`
    and any element of `b`.  Used for non-uniform (nudged) rings.
    """
    d = np.abs(a[:, None] - b[None, :]) % (2 * np.pi)
    return float(np.minimum(d, 2 * np.pi - d).min())


def min_circ_sep_uniform(n_i: int, n_j: int,
                          offset_i: float, offset_j: float) -> float:
    """
    O(1) minimum circular separation for two *uniform* rings.

    The fractions {k/n_i − m/n_j mod 1 : k,m ∈ Z} are multiples of
    1/lcm(n_i,n_j), so the minimum is determined analytically:

        step = 2π / lcm(n_i, n_j)
        φ    = (offset_i − offset_j) mod step
        sep  = min(φ, step − φ)
    """
    lcm_ij = (n_i * n_j) // gcd(n_i, n_j)
    step   = 2 * math.pi / lcm_ij
    phi    = (float(offset_i) - float(offset_j)) % step
    return min(phi, step - phi)


# Required physical clearance: tighter of channel–emitter and channel–channel.
def _required_clr(ww: float, ed: float, clr: float) -> float:
    return max(ww / 2 + ed / 2, ww) + clr


def signed_clearances_uniform(
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    offsets: np.ndarray,
    ww: float,
    ed: float,
    clr: float,
) -> np.ndarray:
    """
    Fast O(N²) clearances using the O(1) uniform-ring formula.
    Used for Phase 1 offset optimisation (rings are uniformly spaced).

    One value per ring pair (i < j):
        min_circ_sep_uniform(n_i, n_j, off_i, off_j) × r_j  −  required
    """
    req = _required_clr(ww, ed, clr)
    result = []
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            sep = min_circ_sep_uniform(
                int(n_per_ring[i]), int(n_per_ring[j]),
                float(offsets[i]),  float(offsets[j]))
            result.append(sep * float(radii[j]) - req)
    return np.asarray(result)


def _n_cands(n_j: int, r_j: float, req: float, delta_max: float) -> int:
    """
    Minimum candidates per emitter to guarantee an exact local minimum.

    After nudging by ±delta_max, only emitters on ring j within angular
    range  req/r_j + 2·delta_max  of a ring-i emitter can be binding.
    Returns enough neighbours to cover that range plus one for safety.
    """
    arc = 2 * math.pi / n_j
    return max(2, int(math.ceil((req / r_j + 2 * delta_max) / arc)) + 1)


def signed_clearances_local(
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    angles_list: list[np.ndarray],
    ww: float,
    ed: float,
    clr: float,
    delta_max: float,
) -> np.ndarray:
    """
    Signed clearances for nudged (non-uniform) rings using angular-local
    binary search.

    Key ideas
    ---------
    1. Each ring is sorted ONCE per evaluation (not per ring pair), avoiding
       O(N²) sort calls.
    2. For each ring-i emitter only the n_cands nearest ring-j emitters are
       checked (binary search + ±n_cands window).  n_cands is computed so
       the window covers req/r_j + 2·delta_max — large enough that nudges
       cannot move the true minimum outside the window.

    Complexity: O(N · n log n) sort  +  O(N² · n · n_cands) distance.
    n_cands ≈ 2-4, giving 5-20× speed-up over O(n_i · n_j) on large arrays.
    """
    req = _required_clr(ww, ed, clr)

    # Sort every ring once; keep mod-2π values for a, sorted for b
    sorted_rings = [np.sort(a % (2 * np.pi)) for a in angles_list]
    mod_rings    = [a % (2 * np.pi) for a in angles_list]

    result = []
    offs_cache: dict[int, np.ndarray] = {}

    for i in range(len(radii)):
        a_m = mod_rings[i]
        for j in range(i + 1, len(radii)):
            b_s = sorted_rings[j]
            n_b = len(b_s)
            nc  = _n_cands(int(n_per_ring[j]), float(radii[j]), req, delta_max)

            # Cache the ±nc offset range (reuse across evaluations within a call)
            if nc not in offs_cache:
                offs_cache[nc] = np.arange(-nc, nc + 1)
            offs = offs_cache[nc]

            # Binary search: O(n_i log n_j)
            idxs  = np.searchsorted(b_s, a_m)                   # (n_i,)
            cidxs = (idxs[:, None] + offs[None, :]) % n_b       # (n_i, 2nc+1)

            d = np.abs(a_m[:, None] - b_s[cidxs]) % (2 * np.pi)
            d = np.minimum(d, 2 * np.pi - d)

            result.append(float(d.min()) * float(radii[j]) - req)

    return np.asarray(result)


def signed_clearances(
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    angles_list: list[np.ndarray],
    ww: float,
    ed: float,
    clr: float,
) -> np.ndarray:
    """
    General signed clearances using the full O(n_i·n_j) matrix.
    Kept for reference / verification.
    """
    req = _required_clr(ww, ed, clr)
    result = []
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            sep = min_circ_sep(angles_list[i], angles_list[j])
            result.append(sep * float(radii[j]) - req)
    return np.asarray(result)


def theoretical_max_clearance(
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    ww: float,
    ed: float,
    clr: float,
) -> list[tuple[int, int, float]]:
    """
    For each ring pair (i, j), compute the best clearance achievable by any
    choice of offsets: (π / lcm(n_i, n_j)) × r_j − required.
    Returns list of (i, j, max_achievable_clearance) for pairs where
    this is negative (fundamentally infeasible).
    """
    req = _required_clr(ww, ed, clr)
    infeasible = []
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            lcm_ij = (n_per_ring[i] * n_per_ring[j]) // gcd(int(n_per_ring[i]),
                                                              int(n_per_ring[j]))
            best_sep = math.pi / lcm_ij          # best achievable angular sep
            best_clr = best_sep * float(radii[j]) - req
            if best_clr < 0:
                infeasible.append((i, j, best_clr))
    return infeasible


# ═══════════════════════════════════════════════════════════════
# ❹  Cost functions
# ═══════════════════════════════════════════════════════════════

def cost_offsets(
    free_offsets: np.ndarray,
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    ww: float,
    ed: float,
    clr: float,
) -> float:
    """
    Negative minimum clearance for uniform rings — O(N²) per evaluation.
    Uses the O(1) lcm formula so no n_i×n_j matrix is ever built.
    """
    offsets = np.concatenate([[0.0], free_offsets])
    return -float(signed_clearances_uniform(
        radii, n_per_ring, offsets, ww, ed, clr).min())


def cost_positions(
    nudges: np.ndarray,
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    base_offsets: np.ndarray,
    ww: float,
    ed: float,
    clr: float,
    delta_max: float,
) -> float:
    """
    Negative minimum clearance for nudged rings.
    Uses angular-local search: O(n_i · n_cands) per ring pair.
    nudges: flat array of length sum(n_per_ring).
    """
    angles = []
    idx = 0
    for n, off in zip(n_per_ring, base_offsets):
        base = ring_angles(int(n), float(off))
        angles.append(base + nudges[idx: idx + int(n)])
        idx += int(n)
    return -float(signed_clearances_local(
        radii, n_per_ring, angles, ww, ed, clr, delta_max).min())


# ═══════════════════════════════════════════════════════════════
# ❺  Optimizers
# ═══════════════════════════════════════════════════════════════

def optimize_offsets(
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    ww: float,
    ed: float,
    clr: float,
) -> tuple[np.ndarray, float]:
    """
    Global search for per-ring offsets via differential evolution.
    Ring 0 is fixed at 0 (rotational symmetry).
    Each free offset is searched in [0, 2π/n_i] (one symmetry period).
    Returns (offsets, min_clearance_μm).
    """
    # One search period per ring, ring 0 excluded
    bounds = [(0.0, 2.0 * np.pi / float(n)) for n in n_per_ring[1:]]

    result = differential_evolution(
        cost_offsets,
        bounds,
        args=(radii, n_per_ring, ww, ed, clr),
        seed=DE_SEED,
        maxiter=DE_MAXITER,
        tol=1e-8,
        popsize=DE_POPSIZE,
        polish=True,
        workers=-1,
        updating='deferred',
    )
    offsets = np.concatenate([[0.0], result.x])
    return offsets, -result.fun


def refine_positions(
    radii: np.ndarray,
    n_per_ring: np.ndarray,
    base_offsets: np.ndarray,
    ww: float,
    ed: float,
    clr: float,
    delta_max_rad: float,
) -> tuple[list[np.ndarray], float]:
    """
    Second-stage refinement: allow each emitter to shift ±delta_max_rad.
    Returns (angles_list, min_clearance_μm).
    """
    total   = int(n_per_ring.sum())
    bounds  = [(-delta_max_rad, delta_max_rad)] * total

    result = differential_evolution(
        cost_positions,
        bounds,
        args=(radii, n_per_ring, base_offsets, ww, ed, clr, delta_max_rad),
        seed=0,
        maxiter=500,
        tol=1e-8,
        popsize=15,
        polish=True,
        workers=-1,
        updating='deferred',
    )
    angles, idx = [], 0
    for n, off in zip(n_per_ring, base_offsets):
        base = ring_angles(int(n), float(off))
        angles.append(base + result.x[idx: idx + int(n)])
        idx += int(n)
    return angles, -result.fun


# ═══════════════════════════════════════════════════════════════
# ❻  Phase 2 — passage-window routing
# ═══════════════════════════════════════════════════════════════

def passage_windows(r_j: float, emitter_angles: np.ndarray,
                    ww: float, ed: float, clr: float) -> list[tuple[float, float]]:
    """
    Angular gaps at radius r_j wide enough for a waveguide (centre separation
    ≥ ww/2 + ed/2 + clr from any emitter centre).

    Returns list of (gap_start, gap_end) CCW arc pairs on [0, 2π).
    """
    alpha  = (ww / 2 + ed / 2 + clr) / r_j   # required angular half-clearance
    thetas = np.sort(emitter_angles % (2 * np.pi))
    n = len(thetas)
    wins: list[tuple[float, float]] = []
    for m in range(n):
        gap_arc = (thetas[(m + 1) % n] - thetas[m]) % (2 * np.pi) - 2 * alpha
        if gap_arc > 1e-12:
            w_s = (thetas[m] + alpha) % (2 * np.pi)
            w_e = (thetas[(m + 1) % n] - alpha) % (2 * np.pi)
            wins.append((float(w_s), float(w_e)))
    return wins


def _in_arc(theta: float, start: float, end: float) -> bool:
    t, s, e = theta % (2 * np.pi), start % (2 * np.pi), end % (2 * np.pi)
    return (s <= t <= e) if s <= e else (t >= s or t <= e)


def _cdist(a: float, b: float) -> float:
    d = abs(a - b) % (2 * np.pi)
    return min(d, 2 * np.pi - d)


def _window_center(w_s: float, w_e: float) -> float:
    """Centre of the CCW arc from w_s to w_e."""
    arc = (float(w_e) - float(w_s)) % (2 * math.pi)
    return (float(w_s) + arc / 2) % (2 * math.pi)


def nearest_passage_angle(theta: float,
                           windows: list[tuple[float, float]]) -> float:
    """
    If theta is inside any window, return it unchanged (no bend needed).
    Otherwise return the centre of the nearest window (by centre distance).

    Routing to window centres rather than edges prevents the left/right
    oscillation that occurs when consecutive rings have window boundaries
    on alternating sides of the current trajectory.  That oscillation is
    the main cause of zig-zag polygon artefacts in the GDS output.
    """
    theta = float(theta) % (2 * np.pi)
    for w_s, w_e in windows:
        if _in_arc(theta, w_s, w_e):
            return theta
    best, best_d = theta, math.inf
    for w_s, w_e in windows:
        center = _window_center(w_s, w_e)
        d = _cdist(theta, center)
        if d < best_d:
            best_d, best = d, center
    return best


def compute_route(r_emitter: float, theta_emitter: float,
                  radii: np.ndarray, angles_list: list[np.ndarray],
                  ww: float, ed: float, clr: float,
                  route_extra: float) -> list[tuple[float, float]]:
    """
    Return (r, theta) waypoints from the emitter to the die boundary.

    A waypoint is only inserted at ring j when the waveguide actually needs
    to bend (passage angle differs from the current trajectory).  Rings where
    the straight-radial angle already fits in a window generate no waypoint,
    keeping the path clean and preventing degenerate near-zero-angle bends.
    """
    r_outer = float(radii[-1])
    wpts: list[tuple[float, float]] = [(r_emitter, float(theta_emitter) % (2 * np.pi))]
    cur = float(theta_emitter) % (2 * np.pi)

    for r_j, ang_j in zip(radii, angles_list):
        if float(r_j) <= r_emitter + 1e-9:
            continue
        wins    = passage_windows(float(r_j), ang_j, ww, ed, clr)
        passage = nearest_passage_angle(cur, wins)
        # Only add waypoint when a real bend occurs
        if _cdist(passage, cur) > 1e-6:
            wpts.append((float(r_j), passage))
            cur = passage

    wpts.append((r_outer + route_extra, cur))
    return wpts


# ═══════════════════════════════════════════════════════════════
# ❼  GDS generation
# ═══════════════════════════════════════════════════════════════

@gf.cell
def emitter_disk(ed: float):
    """Emitter disk, layer 2."""
    c = gf.Component()
    φ   = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pts = np.column_stack([(ed / 2) * np.cos(φ), (ed / 2) * np.sin(φ)])
    c.add_polygon(snap(pts).tolist(), layer=(2, 0))
    return c


def _polyline_polygon(pts_xy: np.ndarray, ww: float) -> np.ndarray:
    """Expand a polyline to a closed polygon of constant width ww."""
    hw = ww / 2.0
    pts = np.asarray(pts_xy, float)
    left: list[np.ndarray] = []
    right: list[np.ndarray] = []
    for i in range(len(pts)):
        tang = (pts[1] - pts[0] if i == 0
                else pts[-1] - pts[-2] if i == len(pts) - 1
                else pts[i + 1] - pts[i - 1])
        t = float(np.hypot(*tang))
        if t < 1e-12:
            continue
        norm = np.array([-tang[1], tang[0]]) / t
        left.append(pts[i] + hw * norm)
        right.append(pts[i] - hw * norm)
    return np.array(left + right[::-1])


def _rt_to_xy(wpts: list[tuple[float, float]]) -> np.ndarray:
    """Convert (r, θ) waypoints to (x, y), removing near-duplicate points."""
    pts = np.array([[r * np.cos(th), r * np.sin(th)] for r, th in wpts])
    keep = np.concatenate([[True],
                           np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])) > GRID])
    return pts[keep]


def _prune_shallow_bends(pts: np.ndarray, min_angle_deg: float = 2.0) -> np.ndarray:
    """
    Iteratively remove interior waypoints whose bend angle is below
    min_angle_deg.  gf.path.smooth inserts Euler curves at every interior
    vertex; sub-degree bends produce degenerate S-curves that appear as
    zig-zag polygon artifacts.
    """
    if len(pts) <= 2:
        return pts
    min_angle = np.radians(min_angle_deg)
    changed = True
    while changed:
        changed = False
        keep = [True]
        for i in range(1, len(pts) - 1):
            v1 = pts[i] - pts[i - 1]
            v2 = pts[i + 1] - pts[i]
            l1, l2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
            if l1 < 1e-12 or l2 < 1e-12:
                keep.append(False)
                changed = True
                continue
            cos_a = float(np.clip(np.dot(v1 / l1, v2 / l2), -1.0, 1.0))
            angle = np.arccos(cos_a)
            if angle < min_angle:
                keep.append(False)
                changed = True
            else:
                keep.append(True)
        keep.append(True)
        pts = pts[np.array(keep, dtype=bool)]
    return pts


def add_route(c: gf.Component,
              wpts_rt: list[tuple[float, float]],
              ww: float, bend_radius: float,
              layer: tuple[int, int] = (1, 0)) -> None:
    """
    Add a smooth Euler-bend waveguide to component c.
    Falls back to a straight-segment polygon if gf.path.smooth fails.
    """
    pts = _rt_to_xy(wpts_rt)
    pts = _prune_shallow_bends(pts)
    if len(pts) < 2:
        return
    try:
        path = gf.path.smooth(points=pts, radius=float(bend_radius),
                              bend=gf.path.euler)
        xs  = gf.cross_section.strip(width=float(ww), layer=layer)
        c.add_ref(gf.path.extrude(path, xs))
    except Exception as e:
        import warnings
        warnings.warn(f"gf.path.smooth failed ({e}); falling back to polyline polygon. pts={pts.tolist()}")
        poly = snap(_polyline_polygon(pts, ww))
        c.add_polygon(poly.tolist(), layer=layer)


def build_gds(radii: np.ndarray,
              angles_list: list[np.ndarray],
              ww: float, ed: float, clr: float,
              route_extra: float, bend_radius: float) -> gf.Component:
    """Assemble layout: emitter disks + passage-window bent routing."""
    c = gf.Component("circular_opa")
    for r, angles in zip(radii, angles_list):
        for θ in angles:
            x = float(snap(r * np.cos(θ)))
            y = float(snap(r * np.sin(θ)))
            ref = c.add_ref(emitter_disk(float(ed)))
            ref.dmove((x, y))
            wpts = compute_route(float(r), float(θ),
                                 radii, angles_list, ww, ed, clr, route_extra)
            add_route(c, wpts, ww, bend_radius)
    return c


# ═══════════════════════════════════════════════════════════════
# ❽  Main
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    arc_gap     = DR - ED
    gap_needed  = WW + 2.0 * CLR
    dr_min_hard = ED + WW + 2.0 * CLR

    print("=" * 66)
    print("Circular OPA — placement + passage-window routing optimizer")
    print("=" * 66)
    print(f"  DR={DR} μm  R0={R0:.3f} μm  N_RINGS={N_RINGS}  N_BASE={N_BASE}")
    print(f"  WW={WW} μm  ED={ED} μm  CLR={CLR} μm  BEND_RADIUS={BEND_RADIUS} μm")
    print(f"  Arc gap = DR − ED = {arc_gap:.3f} μm"
          f"  (min for 1 waveguide/gap: {gap_needed:.3f} μm)")
    if arc_gap < gap_needed:
        print(f"\n  *** increase DR to ≥ {dr_min_hard:.3f} μm ***\n")
    print()

    radii, n_per_ring = ring_geometry(R0, DR, N_RINGS, N_BASE)
    print(f"  Total emitters : {int(n_per_ring.sum())}")
    print(f"  n_per_ring     : {n_per_ring.tolist()}")
    print()

    # Pairs infeasible for straight routing (Phase 2 bends compensate)
    infeasible = theoretical_max_clearance(radii, n_per_ring, WW, ED, CLR)
    if infeasible:
        print("  Ring pairs requiring bent routing:")
        for i, j, mc in infeasible[:12]:
            lcm_ij = (n_per_ring[i] * n_per_ring[j]) // gcd(
                int(n_per_ring[i]), int(n_per_ring[j]))
            print(f"    ({i},{j})  n=({n_per_ring[i]},{n_per_ring[j]})"
                  f"  lcm={lcm_ij}  best_straight_clr={mc:.4f} μm")
        if len(infeasible) > 12:
            print(f"    … and {len(infeasible) - 12} more")
        print()

    # ── Phase 1: emitter placement ────────────────────────────
    print("Phase 1: optimising ring offsets …")
    offsets, min_clr = optimize_offsets(radii, n_per_ring, WW, ED, CLR)
    print(f"  Min clearance (straight model): {min_clr:.4f} μm"
          f"  {'[OK]' if min_clr >= 0 else '[violations → Phase 2 bends compensate]'}")
    print("  Ring offsets:")
    for i, (n, off) in enumerate(zip(n_per_ring, offsets)):
        print(f"    ring {i:2d}  n={n:4d}  {math.degrees(off):9.4f}°")

    if OPTIMIZE_POSITIONS:
        delta_rad = math.radians(DELTA_MAX_DEG)
        print(f"\nPhase 1b: per-emitter nudge ±{DELTA_MAX_DEG}° …")
        angles_list, min_clr2 = refine_positions(
            radii, n_per_ring, offsets, WW, ED, CLR, delta_rad)
        print(f"  Min clearance after nudge: {min_clr2:.4f} μm")
    else:
        angles_list = [ring_angles(int(n), float(off))
                       for n, off in zip(n_per_ring, offsets)]

    # ── Phase 2: passage-window routing + GDS ─────────────────
    print(f"\nPhase 2: passage-window routing + GDS → {GDS_OUT}")
    c = build_gds(radii, angles_list, WW, ED, CLR, ROUTE_EXTRA, BEND_RADIUS)
    c.write_gds(GDS_OUT)
    print("Done.")
