"""Abstract base for mechanical-model plug-ins used by the EOM co-simulation.

A ``MechanicalModel`` consumes a device density ``rho`` (the optical topology
post-projection) and a body-force field ``force`` on the same design-voxel
grid, and returns the equilibrium displacement field ``u``.  ``rho`` is passed
in case a model needs the geometry (e.g. modal stiffness depending on filled
voxels); minimal implementations can ignore it.

The production mechanical model is
:class:`fdtdx.optimization.mechanical.elasticity.ElasticityEigenmodes`.
Adding a new model — squeeze-film damping, nonlinear elasticity — is a
matter of implementing this interface; nothing else in the EOM pipeline
needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import jax

from fdtdx.core.jax.pytrees import TreeClass, autoinit


@autoinit
class MechanicalModel(TreeClass, ABC):
    """Map (geometry, force) → equilibrium displacement field.

    The interface is intentionally narrow: a single ``equilibrium_displacement``
    call returns a vector field on the design-voxel grid of the device the
    model is attached to.  Time-dependent / dynamic models can implement the
    same interface by returning the quasi-static displacement at a chosen
    operating point.
    """

    @abstractmethod
    def equilibrium_displacement(
        self,
        rho: jax.Array,
        force: jax.Array,
    ) -> jax.Array:
        """Return ``u`` of shape ``(3, *design_grid_shape)``.

        Parameters
        ----------
        rho:
            Post-projection density on the design grid, shape
            ``(Nx, Ny, Nz)`` or ``(Nx, Ny)`` for a single-layer device.
        force:
            Body-force vector field, shape ``(3, *rho.shape)``.

        Notes
        -----
        Differentiability with respect to ``rho`` and ``force`` is the
        responsibility of the subclass.  Linear modal projections are
        trivially differentiable; nonlinear fixed-point solves should wrap
        their iteration in ``jax.lax.custom_root`` or
        ``jaxopt.implicit_diff.custom_root`` to keep gradients sane.
        """
