"""Top-level ``Optimization`` class - the generic inverse-design driver.

The class collects the pieces (objects, initial arrays/params, config, a user-
supplied ``simulate_fn``, a set of ``Objective`` s, a set of ``Constraint`` s,
and an optax optimizer) and exposes ``run()`` which:

  1. Optionally loads seed params or resumes a full checkpoint.
  2. JIT-compiles a single-step ``value_and_grad(loss_fn)``.
  3. Iterates for ``total_epochs`` epochs, applying optax updates + param clip.
  4. Logs via an optional ``fdtdx.Logger`` and writes resumable checkpoints
     every ``checkpoint_every`` epochs.

``simulate_fn`` is the user's hook into the FDTD side. Typical signature::

    def simulate(params, arrays, objects, config, key, epoch):
        arrays, objects, _ = fdtdx.apply_params(
            arrays, objects, params, key, beta=beta_schedule(epoch)
        )
        _, arrays = fdtdx.run_fdtd(
            arrays=arrays, objects=objects, config=config, key=key
        )
        return arrays

The suite doesn't hard-code ``apply_params`` + ``run_fdtd`` here - exposing it
as a callable lets users control beta-scheduling, custom backward passes, or
anything else they want to thread through.
"""

import time
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import jax
import jax.numpy as jnp
import optax
from loguru import logger as _log

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.fdtd.container import ArrayContainer, ObjectContainer, ParameterContainer
from fdtdx.optimization.terms import Constraint, Objective
from fdtdx.optimization.utils.checkpoint import (
    load_checkpoint,
    load_seed_params,
    save_checkpoint,
)
from fdtdx.utils.logger import Logger

SimulateFn = Callable[
    [ParameterContainer, ArrayContainer, ObjectContainer, SimulationConfig, jax.Array, jax.Array],
    ArrayContainer,
]


@autoinit
class Optimization(TreeClass):
    """Generic inverse-design optimization driver.

    Attributes:
        objects: Output of ``fdtdx.place_objects`` (or any ``ObjectContainer``-
            shaped stand-in). Not traced through the pytree - objects are
            structural and their fields are updated via ``simulate_fn``.
        arrays: Output of ``fdtdx.place_objects`` (or any ``ArrayContainer``-
            shaped stand-in). Traced, since its leaves (E/H fields, detector
            states) are the live simulation state.
        params: Initial parameter container. Traced; the primary gradient
            target.
        config: Simulation configuration.
        simulate_fn: Callable
            ``(params, arrays, objects, config, key, epoch) -> ArrayContainer``.
            Typically wraps ``apply_params`` + ``run_fdtd`` (and any beta
            schedule).
        optimizer: Any ``optax.GradientTransformation``.
        objectives: Sequence of ``Objective`` instances summed into the loss
            with sign ``-1`` (higher raw value -> lower loss).
        constraints: Sequence of ``Constraint`` instances summed with sign
            ``+1`` (higher raw penalty -> higher loss).
        total_epochs: Number of epochs to run in a fresh optimisation. Ignored
            when resuming if the resumed epoch already exceeds it.
        param_clip: ``(low, high)`` to clamp params after every optax update,
            or ``None`` to disable. Default ``(0.0, 1.0)`` matches the
            standard unit-box latent.
        logger: Optional ``fdtdx.Logger``. If ``None``, logging is skipped.
        log_every: Log cadence in epochs. Ignored when ``logger is None``.
        checkpoint_every: Checkpoint cadence in epochs.
        checkpoint_dir: Where to write resumable checkpoints. Defaults to
            ``logger.cwd / 'checkpoints'`` when a logger is attached, otherwise
            ``None`` (no checkpoints written).
    """

    objects: ObjectContainer = frozen_field()
    # `arrays` is frozen_field (not a pytree leaf of Optimization) so that the
    # initial reference stored on `self` doesn't block JIT-donation of the
    # buffer inside `run()` (donated buffers are deleted, and pytreeclass's
    # copy-based `.aset` on the final return would otherwise choke).
    arrays: ArrayContainer = frozen_field()
    params: ParameterContainer = field()
    config: SimulationConfig = frozen_field()
    simulate_fn: SimulateFn = frozen_field()
    optimizer: Any = frozen_field()  # optax.GradientTransformation - typed as Any to avoid pytree-izing

    # Sequences of terms are frozen_field so their internal schedule/mask
    # leaves are NOT traced as optax-updatable parameters. (Individual
    # sub-leaves are already frozen_field internally, but excluding at the
    # Optimization level keeps the pytree boundary clean.)
    objectives: Sequence[Objective] = frozen_field(default=())
    constraints: Sequence[Constraint] = frozen_field(default=())

    total_epochs: int = frozen_field(default=500)
    param_clip: tuple[float, float] | None = frozen_field(default=(0.0, 1.0))
    logger: Logger | None = frozen_field(default=None)
    log_every: int = frozen_field(default=1)
    checkpoint_every: int = frozen_field(default=50)
    checkpoint_dir: Path | str | None = frozen_field(default=None)

    # ------------------------------------------------------------------
    # Loss construction (pure, JIT-compatible)
    # ------------------------------------------------------------------
    def loss_fn(
        self,
        params: ParameterContainer,
        arrays: ArrayContainer,
        key: jax.Array,
        epoch: jax.Array,
    ) -> tuple[jax.Array, tuple[ArrayContainer, dict[str, jax.Array]]]:
        """Compute the scalar loss + updated arrays + info dict.

        The loss is the signed, weighted sum of every objective (``sign=-1``)
        and every constraint (``sign=+1``). See :class:`LossTerm` for the full
        contract.
        """
        arrays = self.simulate_fn(params, arrays, self.objects, self.config, key, epoch)
        total = jnp.asarray(0.0, dtype=jnp.float32)
        info: dict[str, jax.Array] = {}
        for obj in self.objectives:
            contrib, obj_info = obj(
                sign=-1.0,
                params=params,
                objects=self.objects,
                arrays=arrays,
                config=self.config,
                epoch=epoch,
            )
            total = total + contrib
            info.update(obj_info)
        for con in self.constraints:
            contrib, con_info = con(
                sign=+1.0,
                params=params,
                objects=self.objects,
                arrays=arrays,
                config=self.config,
                epoch=epoch,
            )
            total = total + contrib
            info.update(con_info)
        info["loss"] = total
        return total, (arrays, info)

    # ------------------------------------------------------------------
    # Main driver
    # ------------------------------------------------------------------
    def run(
        self,
        *,
        key: jax.Array,
        seed_from: str | Path | None = None,
        seed_iter: int | None = None,
        resume_from: str | Path | None = None,
    ) -> "Optimization":
        """Run the optimization loop.

        Args:
            key: Initial RNG key.
            seed_from: Optional seed-param directory (see
                :func:`load_seed_params`). Mutually exclusive with
                ``resume_from``.
            seed_iter: Optional iteration index inside ``seed_from`` to load.
            resume_from: Optional checkpoint path (directory or file) to
                restore params + optax state + epoch + RNG. Mutually exclusive
                with ``seed_from``.

        Returns:
            A new :class:`Optimization` with ``params`` and ``arrays`` updated
            to the final state of the loop.
        """
        if seed_from is not None and resume_from is not None:
            raise ValueError("seed_from and resume_from are mutually exclusive.")

        params = self.params
        arrays = self.arrays
        opt_state = self.optimizer.init(params)
        start_epoch = 0

        if seed_from is not None:
            params = load_seed_params(seed_from, params, iter_idx=seed_iter)
            _log.info(f"Seeded params from {seed_from}")

        if resume_from is not None:
            restored_epoch, params, opt_state, key = load_checkpoint(
                resume_from,
                params_template=params,
                opt_state_template=opt_state,
                rng_key_template=key,
            )
            start_epoch = restored_epoch + 1
            _log.info(f"Resumed from epoch {restored_epoch}; starting at {start_epoch}")

        ckpt_dir = self._resolve_checkpoint_dir()

        value_and_grad = jax.value_and_grad(self.loss_fn, has_aux=True)
        jit_step = jax.jit(value_and_grad, donate_argnames=["arrays"])
        # Lower + compile once up front so the first epoch isn't penalised.
        _log.info("Compiling loss_fn...")
        compile_start = time.time()
        compiled = jit_step.lower(
            params,
            arrays,
            key,
            jnp.asarray(start_epoch, dtype=jnp.float32),
        ).compile()
        _log.info(f"Compilation finished in {time.time() - compile_start:.1f}s")

        progress_task = None
        if self.logger is not None:
            progress_task = self.logger.progress.add_task(
                "Optimization",
                total=max(self.total_epochs - start_epoch, 0),
            )

        for epoch in range(start_epoch, self.total_epochs):
            run_start = time.time()
            key, subkey = jax.random.split(key)
            epoch_arr = jnp.asarray(epoch, dtype=jnp.float32)
            (loss, (arrays, info)), grads = compiled(params, arrays, subkey, epoch_arr)

            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = cast(ParameterContainer, optax.apply_updates(params, updates))
            if self.param_clip is not None:
                low, high = self.param_clip
                params = cast(
                    ParameterContainer,
                    jax.tree_util.tree_map(lambda p: jnp.clip(p, low, high), params),
                )

            info["epoch"] = jnp.asarray(epoch)
            info["loss"] = loss
            info["grad_norm"] = optax.global_norm(grads)
            info["update_norm"] = optax.global_norm(updates)
            info["runtime"] = jnp.asarray(time.time() - run_start)

            if self.logger is not None and epoch % self.log_every == 0:
                changed = self.logger.log_params(
                    iter_idx=epoch,
                    params=params,
                    objects=self.objects,
                )
                info["changed_voxels"] = changed
                self.logger.log_detectors(
                    iter_idx=epoch,
                    objects=self.objects,
                    detector_states=arrays.detector_states,
                )
                self.logger.write(info)
                if progress_task is not None:
                    self.logger.progress.update(progress_task, advance=1)

            if ckpt_dir is not None and (epoch % self.checkpoint_every == 0 or epoch == self.total_epochs - 1):
                save_checkpoint(
                    ckpt_dir,
                    epoch=epoch,
                    params=params,
                    opt_state=opt_state,
                    rng_key=key,
                )

        return self.aset("params", params).aset("arrays", arrays)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _resolve_checkpoint_dir(self) -> Path | None:
        if self.checkpoint_dir is not None:
            p = Path(self.checkpoint_dir)
            p.mkdir(parents=True, exist_ok=True)
            return p
        if self.logger is not None:
            p = Path(self.logger.cwd) / "checkpoints"
            p.mkdir(parents=True, exist_ok=True)
            return p
        return None
