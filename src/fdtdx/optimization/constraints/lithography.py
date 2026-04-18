"""Hopkins partially-coherent lithography model and OPC constraint.

**Hopkins formulation.** The aerial image of a mask :math:`m(x)` with
spectrum :math:`M(f)` through a partially coherent optical system is

.. math::

    I(x) = \\iint \\mathrm{TCC}(f_1, f_2) \\, M(f_1) \\, M^*(f_2)
           \\, e^{2\\pi i (f_1 - f_2) \\cdot x} \\, df_1 \\, df_2,

where the **Transmission Cross Coefficient**

.. math::

    \\mathrm{TCC}(f_1, f_2) = \\int J(s) \\, H(s + f_1) \\, H^*(s + f_2) \\, ds

absorbs the source intensity :math:`J(s)` and the pupil function :math:`H(f)`.

**SOCS compression.** TCC is a Hermitian positive-semidefinite operator on
the frequency grid, so it admits the eigendecomposition

.. math::

    \\mathrm{TCC}(f_1, f_2) = \\sum_k \\lambda_k \\, \\varphi_k(f_1) \\varphi_k^*(f_2)

with :math:`\\lambda_k \\ge 0`.  Truncating to the top :math:`K` eigenpairs
turns the 4D integral into :math:`K` coherent imaging systems:

.. math::

    I(x) \\approx \\sum_k \\lambda_k \\, | \\varphi_k * m |^2 (x).

All computation is JAX-native (``jnp.fft.fft2``, ``jnp.abs`` squared,
reduce-sum).  The eigendecomposition runs once at construction via
:meth:`LithographyModel.prepare`, which returns a new (prepared) model
holding the truncated kernel spectra and eigenvalues as frozen pytree
leaves.  The forward pass is then :math:`K` complex FFT/IFFT pairs and
sees gradients flow end-to-end.

**Source and pupil model.** The pupil ``H(f) = 1`` for
``|f| <= NA/wavelength`` (binary, aberration-free).  The source ``J(s)``
is a conventional disc (``sigma_inner=0``) or an annulus
(``sigma_inner, sigma_outer``) in normalized pupil-fraction coordinates.
The optional ``sigma_transition`` (δσ) replaces the hard edges with
sigmoid roll-offs of that radial width, matching real scanner illuminators
(Lin et al., IEEE JSTQE 2020).

**Usage.** Either construct the model, call ``prepare(grid_shape, pitch)``
explicitly, and pass the prepared model into ``OPCConstraint``; or use the
convenience constructor ``OPCConstraint.for_device(device, model, ...)``
which does the preparation automatically from the device's XY design-voxel
grid.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from fdtdx.core.jax.pytrees import TreeClass, autoinit, frozen_field
from fdtdx.optimization.constraints.physics import _get_device, _squeeze_trailing_singletons
from fdtdx.optimization.terms import Constraint

__all__ = [
    "LithographyModel",
    "OPCConstraint",
]


@autoinit
class LithographyModel(TreeClass):
    """Partially-coherent Hopkins imaging + sigmoid resist.

    Instances are lightweight configuration objects until :meth:`prepare` is
    called; after preparation they carry the precomputed SOCS kernels and
    eigenvalues on a specific frequency grid.  A prepared model only works on
    designs matching that grid (``prepare(grid_shape, voxel_pitch_m)``).
    """

    wavelength_m: float = frozen_field(default=193e-9)
    numerical_aperture: float = frozen_field(default=0.85)
    sigma_inner: float = frozen_field(default=0.0)
    sigma_outer: float = frozen_field(default=0.7)
    sigma_transition: float = frozen_field(default=0.0)
    resist_threshold: float = frozen_field(default=0.3)
    resist_sharpness: float = frozen_field(default=50.0)
    num_kernels: int = frozen_field(default=20)
    source_grid_points: int = frozen_field(default=41)

    # Populated by prepare(); None on a fresh model.
    kernels: Any = frozen_field(default=None)
    eigenvalues: Any = frozen_field(default=None)
    grid_shape: Any = frozen_field(default=None)
    voxel_pitch_m: Any = frozen_field(default=None)

    # ------------------------------------------------------------------
    # Setup: TCC -> eigh -> top-K kernel spectra (runs once, on numpy)
    # ------------------------------------------------------------------

    def prepare(
        self,
        grid_shape: tuple[int, int],
        voxel_pitch_m: float,
    ) -> "LithographyModel":
        """Build the TCC on ``grid_shape`` for the given voxel pitch, run
        :func:`numpy.linalg.eigh`, keep the top ``num_kernels`` kernels, and
        return a new :class:`LithographyModel` with the cache populated.

        The setup uses numpy (not jax.numpy) so the eigendecomposition does
        not inflate JIT traces; the cached kernels are then copied to JAX
        arrays as frozen pytree leaves.

        Parameters
        ----------
        grid_shape:
            ``(Ny, Nx)`` of the design-voxel grid this model will be applied
            to.  ``Ny`` and ``Nx`` are the last two spatial axes of the
            device density.
        voxel_pitch_m:
            Physical size of a design voxel in meters (isotropic XY).
        """
        ny, nx = int(grid_shape[0]), int(grid_shape[1])
        pitch = float(voxel_pitch_m)
        if pitch <= 0.0:
            raise ValueError(f"voxel_pitch_m must be > 0, got {pitch}")

        fy = np.fft.fftfreq(ny, d=pitch)
        fx = np.fft.fftfreq(nx, d=pitch)
        fy2d, fx2d = np.meshgrid(fy, fx, indexing="ij")
        f_radial = np.sqrt(fy2d**2 + fx2d**2)

        cutoff = float(self.numerical_aperture) / float(self.wavelength_m)

        # TCC has support only where either f1 or f2 lies in the (shifted)
        # pupil; |f| <= 2*cutoff is the conservative outer envelope.
        active_mask = f_radial <= 2.0 * cutoff
        iy, ix = np.where(active_mask)
        n_active = iy.size
        if n_active == 0:
            raise ValueError(
                "No frequencies fall inside the pupil passband - "
                f"voxel_pitch_m={pitch} is too coarse for "
                f"NA={self.numerical_aperture}, wavelength={self.wavelength_m}"
            )

        # Source grid J(s) on a square in pupil-fraction units.  With a soft
        # edge (sigma_transition > 0), extend the integration square past
        # sigma_outer by ~3 delta-sigma so the sigmoid tail is resolved.
        ns = int(self.source_grid_points)
        dsigma = float(self.sigma_transition)
        if dsigma < 0.0:
            raise ValueError(f"sigma_transition must be >= 0, got {dsigma}")
        sigma_extent_frac = float(self.sigma_outer) + (3.0 * dsigma if dsigma > 0.0 else 0.0)
        s_extent = sigma_extent_frac * cutoff
        s_axis = np.linspace(-s_extent, s_extent, ns)
        sy, sx = np.meshgrid(s_axis, s_axis, indexing="ij")
        s_frac = np.sqrt(sy**2 + sx**2) / max(cutoff, 1e-30)
        if dsigma == 0.0:
            w = ((s_frac >= float(self.sigma_inner)) & (s_frac <= float(self.sigma_outer))).astype(np.float32)
        else:
            # Outer sigmoid falls from 1 to 0 around sigma_outer over width dsigma;
            # inner sigmoid rises from 0 to 1 around sigma_inner.  Product is the
            # full soft annulus (reduces to a soft disc when sigma_inner == 0).
            w_out = 1.0 / (1.0 + np.exp((s_frac - float(self.sigma_outer)) / dsigma))
            w_in = 1.0 / (1.0 + np.exp((float(self.sigma_inner) - s_frac) / dsigma))
            w = (w_out * w_in).astype(np.float32)
        w_sum = float(w.sum())
        if w_sum == 0.0:
            raise ValueError("Source has zero integrated weight; check sigma_inner/outer and source_grid_points.")
        w = w / w_sum

        # Prune negligible contributions.  Hard edge: any nonzero cell.  Soft
        # edge: drop cells below 1e-4 of peak, which keeps the TCC build
        # tractable while preserving the tail shape to <0.01% accuracy.
        if dsigma == 0.0:
            mask = w > 0
        else:
            mask = w > 1e-4 * float(w.max())
        sy_flat = sy[mask]
        sx_flat = sx[mask]
        w_flat = w[mask]
        # Renormalize post-pruning so the kept mass still integrates to 1.
        w_flat = w_flat / float(w_flat.sum())

        fy_active = fy2d[iy, ix]
        fx_active = fx2d[iy, ix]

        # h_shifted[s, i] = H(s + f_i) = (|s+f_i| <= cutoff).
        # Shape (n_src, n_active).  For a moderate grid this is the biggest
        # intermediate; e.g. 64x64 grid inside a pupil of radius cutoff gives
        # n_active ~ 10k for tight pupils, n_src up to ~ns^2/2.
        fshift_y = sy_flat[:, None] + fy_active[None, :]
        fshift_x = sx_flat[:, None] + fx_active[None, :]
        h_shifted = (np.sqrt(fshift_y**2 + fshift_x**2) <= cutoff).astype(np.float32)

        # TCC[i, j] = sum_s w_s * h_shifted[s, i] * h_shifted[s, j]
        # h is real, so TCC is real symmetric; store as complex64 for eigh.
        weighted = h_shifted * w_flat[:, None]
        tcc = weighted.T @ h_shifted
        tcc = tcc.astype(np.complex64)
        # Symmetrize against tiny roundoff (eigh requires exact Hermitian).
        tcc = 0.5 * (tcc + tcc.conj().T)

        eigvals_all, eigvecs_all = np.linalg.eigh(tcc)  # ascending
        k = min(int(self.num_kernels), n_active)
        # Top-k (largest eigenvalues are at the end); flip to descending.
        eigvals = eigvals_all[-k:][::-1].real.astype(np.float32)
        eigvecs = eigvecs_all[:, -k:][:, ::-1]  # (n_active, k) complex64

        # Scatter kernel spectra back onto the full (k, Ny, Nx) grid.
        kernels_spec = np.zeros((k, ny, nx), dtype=np.complex64)
        for kk in range(k):
            kernels_spec[kk, iy, ix] = eigvecs[:, kk]

        kernels_jax = jnp.asarray(kernels_spec)
        eigvals_jax = jnp.asarray(eigvals)

        new = self.aset("kernels", kernels_jax)
        new = new.aset("eigenvalues", eigvals_jax)
        new = new.aset("grid_shape", (ny, nx))
        new = new.aset("voxel_pitch_m", pitch)
        return new

    # ------------------------------------------------------------------
    # Forward: aerial image + resist model
    # ------------------------------------------------------------------

    def _check_prepared(self, design_shape: tuple[int, ...]) -> tuple[int, int]:
        if self.kernels is None or self.eigenvalues is None or self.grid_shape is None:
            raise ValueError(
                "LithographyModel has not been prepared. "
                "Call `.prepare((Ny, Nx), voxel_pitch_m)` first, "
                "or use `OPCConstraint.for_device(device, model)`."
            )
        ny, nx = self.grid_shape
        if design_shape[-2:] != (ny, nx):
            raise ValueError(f"design spatial shape {design_shape[-2:]} does not match prepared grid {(ny, nx)}")
        return int(ny), int(nx)

    def aerial_image(self, design: jax.Array) -> jax.Array:
        """Compute the continuous partially-coherent aerial intensity.

        Operates on the last two axes; any leading axes broadcast as batch.
        Returns a real, non-negative array with the same shape as ``design``.
        """
        ny, nx = self._check_prepared(design.shape)

        leading = design.shape[:-2]
        m = design.reshape((-1, ny, nx)).astype(jnp.complex64)
        m_spec = jnp.fft.fft2(m)  # (B, ny, nx)

        # (B, 1, ny, nx) * (1, K, ny, nx) -> (B, K, ny, nx)
        product = m_spec[:, None, :, :] * self.kernels[None, :, :, :]
        coherent = jnp.fft.ifft2(product)  # (B, K, ny, nx) complex
        intensities = jnp.real(coherent * jnp.conj(coherent))  # (B, K, ny, nx)
        weighted = self.eigenvalues[None, :, None, None] * intensities
        aerial = jnp.sum(weighted, axis=1)  # (B, ny, nx)
        return aerial.reshape(leading + (ny, nx))

    def forward(self, design: jax.Array) -> tuple[jax.Array, jax.Array]:
        """Return ``(printed, aerial)``.

        The printed pattern is a sigmoid-thresholded version of the aerial
        image::

            printed = sigmoid(resist_sharpness * (aerial - resist_threshold))

        Both outputs have the same shape as ``design``.
        """
        aerial = self.aerial_image(design)
        printed = jax.nn.sigmoid(self.resist_sharpness * (aerial - self.resist_threshold))
        return printed, aerial


@autoinit
class OPCConstraint(Constraint):
    """Optical-proximity-correction penalty: predicted-print vs. target.

    Given a device density ``rho`` and a prepared :class:`LithographyModel`,
    computes the predicted resist pattern and penalizes its deviation from
    the design (or, if ``target_design`` is set, from that fixed reference).

    Default mode (self-consistency)::

        penalty = mean( (printed - rho)^2 )

    With ``target_design`` set::

        penalty = mean( (printed - target_design)^2 )

    The first form pushes the optimizer toward litho-robust designs - ones
    whose post-fabrication topology coincides with the design intent.  The
    second form is classical inverse lithography: drive ``rho`` so that
    ``printed`` reproduces a fixed target spec.

    The ``LithographyModel`` must be prepared on the device's XY grid before
    the optimization loop starts.  Use :meth:`for_device` for the common
    case.
    """

    device_name: str = frozen_field()
    litho_model: LithographyModel = frozen_field()
    target_design: jax.Array | None = frozen_field(default=None)

    @classmethod
    def for_device(
        cls,
        device: Any,
        litho_model: "LithographyModel",
        *,
        name: str,
        target_design: jax.Array | None = None,
        **kwargs: Any,
    ) -> "OPCConstraint":
        """Prepare ``litho_model`` on the device's XY design-voxel grid.

        Infers ``grid_shape`` from ``device.matrix_voxel_grid_shape[:2]`` and
        ``voxel_pitch_m`` from ``device.single_voxel_real_shape[0]``.  Both
        require the device to have been placed on the simulation grid
        already (i.e. ``place_objects`` has run).
        """
        grid_shape = tuple(device.matrix_voxel_grid_shape[:2])
        voxel_pitch_m = float(device.single_voxel_real_shape[0])
        prepared = litho_model.prepare(grid_shape, voxel_pitch_m)
        return cls(
            name=name,
            device_name=device.name,
            litho_model=prepared,
            target_design=target_design,
            **kwargs,
        )

    def compute(
        self,
        *,
        params: Any,
        objects: Any,
        **_unused: Any,
    ) -> tuple[jax.Array, dict[str, jax.Array]]:
        device = _get_device(objects, self.device_name)
        rho = device(params[self.device_name])
        rho = _squeeze_trailing_singletons(rho)
        rho = jnp.clip(rho, 0.0, 1.0).astype(jnp.float32)

        printed, aerial = self.litho_model.forward(rho)
        target = rho if self.target_design is None else self.target_design
        mismatch = printed - target
        penalty = jnp.mean(mismatch**2)
        info = {
            f"{self.name}_aerial_max": aerial.max(),
            f"{self.name}_aerial_mean": aerial.mean(),
            f"{self.name}_mismatch": penalty,
        }
        return penalty, info
