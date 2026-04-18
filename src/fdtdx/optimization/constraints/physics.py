"""Generic base classes for physics-based constraints.

A physics-based constraint reads the post-projection density :math:`\\rho` from
a device, runs a physics model on it (a PDE, a fixed-point iteration, an ODE
integrator, ...), and reduces the result to a scalar penalty.  Because the
forward physics is JAX-traced, gradients flow back to the device parameters
via automatic differentiation — or, for linear systems solved by
:func:`jax.scipy.sparse.linalg.cg`, via implicit differentiation for free.

Two layers:

- :class:`PhysicsConstraint` - handles the common boilerplate of looking up
  a device by name, evaluating its parameter-transform pipeline to obtain
  :math:`\\rho`, squeezing trailing singleton axes, and clipping.  Subclasses
  only need to implement :meth:`build_penalty` which maps a rank-2 or rank-3
  density to a ``(scalar, info_dict)`` pair.

- :class:`LinearSteadyStatePDEConstraint` - specializes
  :class:`PhysicsConstraint` for symmetric-positive-definite linear systems
  ``A(rho) u = b(rho)`` solved with conjugate gradient.  Subclasses implement
  :meth:`operator`, :meth:`rhs`, :meth:`penalty`, and optionally
  :meth:`preconditioner_diag` (for Jacobi preconditioning).

:class:`~fdtdx.optimization.constraints.connectivity.VirtualTemperatureConnectivity`
is the canonical concrete subclass of the latter: it supplies a harmonic-mean
variable-coefficient diffusion operator, a source-cell RHS, a diagonal
preconditioner, and a source-mean penalty.
"""

from abc import ABC, abstractmethod
from typing import Any

import jax
import jax.numpy as jnp

from fdtdx.core.jax.pytrees import autoinit, frozen_field
from fdtdx.optimization.terms import Constraint


def _get_device(objects: Any, device_name: str) -> Any:
    """Look up a :class:`Device` by ``name`` in an ``ObjectContainer``."""
    for d in objects.devices:
        if d.name == device_name:
            return d
    raise ValueError(f"No device named {device_name!r} in ObjectContainer")


def _squeeze_to_match(arr: jax.Array, target_ndim: int) -> jax.Array:
    """Drop size-1 axes from ``arr`` until ``arr.ndim == target_ndim``."""
    while arr.ndim > target_ndim:
        squeezable = [i for i, s in enumerate(arr.shape) if s == 1]
        if not squeezable:
            raise ValueError(f"Cannot reduce density of shape {arr.shape} to ndim={target_ndim}")
        arr = jnp.squeeze(arr, axis=squeezable[0])
    return arr


def _squeeze_trailing_singletons(arr: jax.Array) -> jax.Array:
    """Drop all trailing size-1 axes (common when Z=1 for a single-layer device)."""
    while arr.ndim > 1 and arr.shape[-1] == 1:
        arr = jnp.squeeze(arr, axis=-1)
    return arr


@autoinit
class PhysicsConstraint(Constraint, ABC):
    """Constraint backed by a physics model evaluated on device density.

    Handles the common pipeline ``params -> rho`` so subclasses only have to
    build a penalty from the clipped, normalized density field.  The density
    returned by the device is clipped into ``[0, 1]`` and cast to float32
    before :meth:`build_penalty` is called.
    """

    device_name: str = frozen_field()

    @abstractmethod
    def build_penalty(self, rho: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Given the device density ``rho``, return ``(penalty, info_dict)``."""

    def compute(
        self,
        *,
        params: Any,
        objects: Any,
        **_unused: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        device = _get_device(objects, self.device_name)
        rho = device(params[self.device_name])
        rho = jnp.clip(rho, 0.0, 1.0).astype(jnp.float32)
        return self.build_penalty(rho)


@autoinit
class LinearSteadyStatePDEConstraint(PhysicsConstraint, ABC):
    """Physics constraint that solves ``A(rho) u = b(rho)`` with CG.

    The linear operator ``A`` must be symmetric positive definite so that
    conjugate gradient converges.  Subclasses implement the matrix-free action
    :meth:`operator`, the right-hand side :meth:`rhs`, the scalar extraction
    :meth:`penalty`, and optionally :meth:`preconditioner_diag` for Jacobi
    preconditioning.

    Gradients with respect to ``rho`` are supplied automatically by
    :func:`jax.scipy.sparse.linalg.cg`, which defines a VJP via the implicit
    function theorem.
    """

    cg_iterations: int = frozen_field(default=200)
    cg_tol: float = frozen_field(default=1e-6)

    @abstractmethod
    def operator(self, u: jax.Array, rho: jax.Array) -> jax.Array:
        """Matrix-free action ``y = A(rho) u``.  Must be symmetric in ``u``."""

    @abstractmethod
    def rhs(self, rho: jax.Array) -> jax.Array:
        """Right-hand side ``b(rho)``."""

    @abstractmethod
    def penalty(
        self,
        u: jax.Array,
        rho: jax.Array,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        """Reduce the solved state ``u`` (given ``rho``) to a scalar penalty."""

    def preconditioner_diag(self, rho: jax.Array) -> jax.Array | None:
        """Diagonal of ``A(rho)`` for Jacobi preconditioning.  ``None`` = identity."""
        del rho
        return None

    def build_penalty(self, rho: jax.Array) -> tuple[jax.Array, dict[str, jax.Array]]:
        b = self.rhs(rho)

        def A(u: jax.Array) -> jax.Array:
            return self.operator(u, rho)

        diag = self.preconditioner_diag(rho)
        if diag is not None:

            def M(u: jax.Array) -> jax.Array:
                return u / diag
        else:
            M = None

        u, _info = jax.scipy.sparse.linalg.cg(
            A,
            b,
            x0=jnp.zeros_like(b),
            M=M,
            tol=self.cg_tol,
            maxiter=self.cg_iterations,
        )
        return self.penalty(u, rho)
