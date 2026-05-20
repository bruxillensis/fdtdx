"""Static pull-in (limit-point) voltage via displacement-controlled continuation.

Pull-in is a saddle-node bifurcation: on the voltage-controlled static branch
the tangent stiffness goes singular, so a voltage-stepped Newton solve (or a
transient ramp) is both ill-conditioned and expensive right where the answer
is.  The robust, singularity-free method is **displacement-controlled
continuation**: parametrize the equilibrium branch by the deflection
amplitude ``s`` (a generalized coordinate along the static deflection shape)
instead of by ``V``.  For an electrostatic actuator reduced to that one
coordinate the static balance is

.. math::

    k \\, s \\;=\\; V^2 \\, f(s),

where ``k`` is the generalized elastic stiffness along the deflection shape
(for mass-normalized modes ``Φ_i`` with eigenfrequencies ``ω_i`` and a unit
shape ``ŝ = Σ c_i Φ_i``, ``k = Σ c_i² ω_i²``) and ``f(s)`` is the generalized
electrostatic force **per V²** evaluated at the *finite* deflection ``s`` —
i.e. with the moving conductor actually translated to that position and the
electrostatics re-solved (a finite-gap force, **not** the first-order
``ε(x − u·∇ε)`` perturbation, which has no gap-narrowing softening and hence
no pull-in).

Then ``V²(s) = k s / f(s)`` and the pull-in voltage is the maximum of ``V``
over the stable branch:

.. math::

    V_\\mathrm{pi} \\;=\\; \\max_{s} \\sqrt{\\,k\\,s / f(s)\\,}.

Beyond that maximum no statically-stable equilibrium exists.  ``V_pi`` is a
*smooth* function of the design even though the deflection at ``V_pi`` is
not: by the envelope theorem ``dV_pi/dθ = ∂V/∂θ|_{s*}`` (the ``s``-derivative
vanishes at the maximum), so a hard ``max`` over a dense ``s`` grid yields
the correct design sensitivity to sampling resolution — no need to
differentiate through the bifurcation.

:func:`advect_density` is the companion finite-displacement helper: a
differentiable Eulerian resample that rigidly shifts a (sub-voxel) density
field, used by the caller to place the moving conductor at amplitude ``s``
before each electrostatic solve.
"""

from __future__ import annotations

from typing import Callable

import jax
import jax.numpy as jnp

__all__ = [
    "advect_density",
    "pull_in_voltage",
    "pull_in_voltage_from_force_fn",
]


def advect_density(
    field: jax.Array,
    displacement_voxels: jax.Array,
    order: int = 1,
) -> jax.Array:
    """Resample ``field`` at ``x − displacement`` (rigid Eulerian shift).

    Differentiable finite translation of a density / indicator field — the
    correct tool for finite-gap electrostatics, where the moving conductor
    travels a sizeable fraction of the gap (the first-order shape-derivative
    ``ε(x − u·∇ε)`` is only valid for ``|u| ≪`` one voxel and carries no
    gap-narrowing term).

    Parameters
    ----------
    field:
        Density / indicator array of arbitrary spatial rank ``d``.
    displacement_voxels:
        Shift in **voxels** (not metres), shape ``(d, *field.shape)`` or
        broadcastable to it.  Positive component ``j`` moves material toward
        increasing index along axis ``j``.
    order:
        Interpolation order for :func:`jax.scipy.ndimage.map_coordinates`.
        Must be 1 (linear) for a usable gradient; 0 (nearest) is
        non-differentiable.

    Returns
    -------
    jax.Array
        ``field`` shifted by ``displacement_voxels``, same shape, out-of-
        bounds sampled with ``mode="nearest"`` (the surrounding medium).
    """
    coords = jnp.indices(field.shape, dtype=jnp.float32)
    src = coords - jnp.asarray(displacement_voxels, dtype=jnp.float32)
    return jax.scipy.ndimage.map_coordinates(
        field, list(src), order=order, mode="nearest"
    )


def pull_in_voltage(
    amplitudes_m: jax.Array,
    generalized_stiffness: jax.Array,
    generalized_force_per_v2: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Pull-in voltage from a precomputed displacement-controlled branch.

    Parameters
    ----------
    amplitudes_m:
        Strictly-increasing deflection samples ``s`` (m), shape ``(N,)``,
        starting above 0 (s=0 is the trivial undeflected point).
    generalized_stiffness:
        Scalar ``k`` (N/m) — elastic stiffness along the deflection shape.
    generalized_force_per_v2:
        Shape ``(N,)``; ``f(s_j)`` = electrostatic generalized force per V²
        at the *finite* deflection ``s_j`` (caller computes this by actually
        translating the moving conductor to ``s_j`` and solving the
        electrostatics at 1 V, then projecting the force onto the shape).

    Returns
    -------
    v_pi:
        Scalar pull-in voltage ``max_s √(k s / f(s))``.
    v_branch:
        Shape ``(N,)`` — ``V(s_j)`` along the branch (diagnostics / plotting).
        The maximum is the limit point; samples past it are the unstable
        branch.
    """
    s = jnp.asarray(amplitudes_m, dtype=jnp.float32)
    f = jnp.asarray(generalized_force_per_v2, dtype=jnp.float32)
    k = jnp.asarray(generalized_stiffness, dtype=jnp.float32)
    v2 = k * s / jnp.clip(f, 1e-30, None)
    v_branch = jnp.sqrt(jnp.clip(v2, 0.0, None))
    v_pi = jnp.max(v_branch)
    return v_pi, v_branch


def pull_in_voltage_from_force_fn(
    generalized_stiffness: jax.Array,
    force_per_v2_fn: Callable[[jax.Array], jax.Array],
    max_amplitude_m: float,
    n_samples: int = 48,
) -> tuple[jax.Array, jax.Array]:
    """Convenience wrapper: sample the branch via ``force_per_v2_fn`` then
    :func:`pull_in_voltage`.

    ``force_per_v2_fn(s) -> f(s)`` is the geometry-specific closure that, for
    a scalar deflection amplitude ``s`` (m), translates the moving conductor
    to that position (e.g. with :func:`advect_density`), solves the
    electrostatics at 1 V and returns the generalized force per V² projected
    onto the deflection shape.  It is ``vmap``-ed over a uniform grid of
    ``n_samples`` amplitudes in ``(0, max_amplitude_m]``.

    ``max_amplitude_m`` should bracket the limit point — a safe choice is a
    little under the physical gap (pull-in occurs near ``g/3`` for an ideal
    parallel plate; sample out to ~0.7·g so the maximum is interior).
    """
    s = jnp.linspace(
        max_amplitude_m / n_samples, max_amplitude_m, n_samples, dtype=jnp.float32
    )
    f = jax.vmap(force_per_v2_fn)(s)
    return pull_in_voltage(s, generalized_stiffness, f)
