"""Multi-state inverse design: memory-bounded gradient accumulation.

Some losses depend on several *independent* simulations that do not fit in memory
together — a MEMS phase tuner's ``V_on`` / ``V_off`` states, hot/cold thermal pairs,
multi-wavelength, multi-angle, fabrication-corner robustness. Putting them all in a
single :func:`jax.value_and_grad` holds every gradient tape at once (≈ Nx memory).

Because the loss is a function of small *per-state metrics*,

.. math::

    \\mathcal{L} = \\text{combine}(\\{m_i\\}), \\qquad
    \\frac{d\\mathcal{L}}{d\\rho} = \\sum_i \\frac{\\partial\\,\\text{combine}}{\\partial m_i}
        \\cdot \\frac{d m_i}{d\\rho},

the states can be differentiated **one at a time** and their ρ-cotangents summed —
peak memory ≈ one state's tape, runtime ≈ Nx. This is just gradient accumulation
across states.

This module provides the bare primitive :func:`accumulated_value_and_grad` and a
driver :class:`MultiStateOptimization` that mirrors
:class:`~fdtdx.optimization.optimization.Optimization`'s loop. Nothing here is
FDTD-specific: a "state" is any ``params -> metric_pytree`` map, and ``combine`` is
any ``{name: metric} -> (scalar_loss, info)``. The FDTD/FEM machinery lives in the
caller's ``state_fns`` (e.g. ``fdtdx_multiphysics.Scene.state_fn``).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Mapping, cast

import jax
import jax.numpy as jnp
import optax
from loguru import logger as _log

from fdtdx.config import SimulationConfig
from fdtdx.core.jax.pytrees import TreeClass, autoinit, field, frozen_field
from fdtdx.fdtd.container import ObjectContainer, ParameterContainer
from fdtdx.optimization.constraints.physics import _HARDEN_BETA
from fdtdx.optimization.utils.checkpoint import (
    load_checkpoint,
    load_seed_params,
    save_checkpoint,
)
from fdtdx.utils.logger import Logger

#: A state maps params -> a small metric pytree (scalar / complex / dict of reads).
StateFn = Callable[[ParameterContainer], Any]
#: combine maps {name: metric} -> (scalar loss, info dict).
CombineFn = Callable[[Mapping[str, Any]], tuple[jax.Array, dict[str, jax.Array]]]


def _tree_add(a: Any, b: Any) -> Any:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def accumulated_value_and_grad(
    params: Any,
    state_fns: Mapping[str, StateFn],
    combine: CombineFn,
) -> tuple[jax.Array, Any, dict[str, jax.Array], dict[str, Any]]:
    """Memory-bounded ``value_and_grad`` over independently-simulated states.

    Args:
        params: the design pytree differentiated against.
        state_fns: ordered mapping ``{name: f}`` where ``f(params) -> metric_pytree``.
            Each ``f`` runs ONE simulation and returns a *small* metric. They are
            evaluated/vjp'd one at a time, so only one state's gradient tape is
            resident — **pass ``jax.jit``-wrapped ``f`` to guarantee the tape is
            freed between states** (the driver below does this).
        combine: ``{name: metric} -> (scalar_loss, info_dict)`` — the (cheap)
            coupling of the per-state metrics into the scalar objective.

    Returns:
        ``(loss, grad, info, metrics)``. ``grad`` matches what a single
        ``value_and_grad`` over ``combine({n: f_n(params)})`` would give, but is
        assembled state-by-state. ``metrics`` are the (stop-gradient'd) forward
        values, handy for logging.

    Note:
        Each state's forward runs twice (once for its value, once inside its vjp) —
        the deliberate price of never holding two tapes. Complex metrics (e.g. a
        phase via ``jnp.angle``) flow through the standard JAX cotangent
        conventions and compose correctly with a real-valued ``combine``.
    """
    names = list(state_fns)
    # Phase A — forward-only metric values (one sim resident at a time).
    metrics = {n: jax.lax.stop_gradient(state_fns[n](params)) for n in names}
    # Phase B — differentiate the cheap coupling w.r.t. the per-state metrics.
    (loss, info), dmetrics = jax.value_and_grad(combine, has_aux=True)(metrics)
    # Phase C — one vjp per state, weighted by ∂combine/∂metric, summed.
    grad = jax.tree_util.tree_map(jnp.zeros_like, params)
    for n in names:
        _, vjp = jax.vjp(state_fns[n], params)
        (g,) = vjp(dmetrics[n])
        grad = _tree_add(grad, g)
    return loss, grad, info, metrics


@autoinit
class MultiStateOptimization(TreeClass):
    """Driver for losses coupling several independently-simulated states.

    Analogous to :class:`~fdtdx.optimization.optimization.Optimization`, but instead
    of one ``simulate_fn`` wrapped in one ``value_and_grad``, it runs each state's
    forward + vjp as its own compiled call (one gradient tape resident at a time —
    ≈ one sim of peak memory, ≈ Nx runtime). See :func:`accumulated_value_and_grad`.

    Args:
        state_fns: ordered mapping ``{name: f}`` with
            ``f(params, key, epoch) -> (metric_pytree, aux)``. ``metric_pytree`` is
            the small quantity ``combine`` couples; ``aux`` is anything extra (e.g.
            the ``ArrayContainer``) — the last state's ``aux`` is offered to the
            logger / ``epoch_callback`` as ``arrays``. Use ``aux=None`` if unused.
        combine: ``{name: metric} -> (scalar_loss, info_dict)``.
        optimizer: any ``optax.GradientTransformation``.
        params: the traced design pytree.
        objects: optional ``ObjectContainer`` — only used for logging
            (``log_params`` / ``log_detectors``); pass ``None`` for non-FDTD states.
        config: optional ``SimulationConfig`` (kept for parity; unused by the loop).
        total_epochs, param_clip, logger, log_every, checkpoint_every,
        checkpoint_dir, epoch_callback: as in ``Optimization``.
    """

    state_fns: Mapping[str, Callable] = frozen_field()
    combine: Callable = frozen_field()
    optimizer: Any = frozen_field()
    params: ParameterContainer = field()
    objects: ObjectContainer | None = frozen_field(default=None)
    config: SimulationConfig | None = frozen_field(default=None)
    total_epochs: int = frozen_field(default=500)
    param_clip: tuple[float, float] | None = frozen_field(default=(0.0, 1.0))
    logger: Logger | None = frozen_field(default=None)
    log_every: int = frozen_field(default=1)
    checkpoint_every: int = frozen_field(default=50)
    checkpoint_dir: Path | str | None = frozen_field(default=None)
    epoch_callback: Any = frozen_field(default=None)

    def run(
        self,
        *,
        key: jax.Array,
        seed_from: str | Path | None = None,
        seed_iter: int | None = None,
        resume_from: str | Path | None = None,
    ) -> "MultiStateOptimization":
        """Run the multi-state optimization loop. Returns a new instance with
        updated ``params``. Semantics of ``seed_from`` / ``resume_from`` match
        :meth:`Optimization.run`."""
        if seed_from is not None and resume_from is not None:
            raise ValueError("seed_from and resume_from are mutually exclusive.")

        params = self.params
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
        names = list(self.state_fns)

        # Pre-jit, ONCE, each state's forward (-> (metric, aux)) and its
        # cotangent-vjp (-> ρ-grad). Stable arg shapes ⇒ one compile each; each is
        # its own compiled call so the tape is freed on return (the 1x guarantee).
        # Recreating jax.jit inside the loop would recompile every epoch.
        def _make_vjp(fn: Callable) -> Callable:
            def _vjp(p: Any, k: jax.Array, e: jax.Array, ct: Any) -> Any:
                _, vjp = jax.vjp(lambda q: fn(q, k, e)[0], p)
                return vjp(ct)[0]

            return jax.jit(_vjp)

        jit_value = {n: jax.jit(self.state_fns[n]) for n in names}
        jit_vjp = {n: _make_vjp(self.state_fns[n]) for n in names}
        combine_vag = jax.jit(jax.value_and_grad(self.combine, has_aux=True))

        progress_task = None
        if self.logger is not None:
            progress_task = self.logger.progress.add_task(
                "MultiStateOptimization",
                total=max(self.total_epochs - start_epoch, 0),
            )

        for epoch in range(start_epoch, self.total_epochs):
            run_start = time.time()
            key, *subkeys = jax.random.split(key, len(names) + 1)
            state_keys = dict(zip(names, subkeys))
            epoch_arr = jnp.asarray(epoch, dtype=jnp.float32)

            # Phase A: per-state forward values (same key reused in the vjp pass).
            metrics: dict[str, Any] = {}
            last_aux: Any = None
            for n in names:
                metric, aux = jit_value[n](params, state_keys[n], epoch_arr)
                metrics[n] = jax.lax.stop_gradient(metric)
                last_aux = aux
            # Phase B: differentiate the cheap coupling.
            (loss, info), dmetrics = combine_vag(metrics)
            # Phase C: one tape at a time, summed.
            grads = jax.tree_util.tree_map(jnp.zeros_like, params)
            for n in names:
                grads = _tree_add(grads, jit_vjp[n](params, state_keys[n], epoch_arr, dmetrics[n]))

            updates, opt_state = self.optimizer.update(grads, opt_state, params)
            params = cast(ParameterContainer, optax.apply_updates(params, updates))
            if self.param_clip is not None:
                low, high = self.param_clip
                params = cast(
                    ParameterContainer,
                    jax.tree_util.tree_map(lambda p: jnp.clip(p, low, high), params),
                )

            info = dict(info)
            info["epoch"] = jnp.asarray(epoch)
            info["loss"] = loss
            info["grad_norm"] = optax.global_norm(grads)
            info["update_norm"] = optax.global_norm(updates)
            info["runtime"] = jnp.asarray(time.time() - run_start)

            if self.logger is not None and epoch % self.log_every == 0:
                if self.objects is not None:
                    try:
                        changed = self.logger.log_params(
                            iter_idx=epoch, params=params, objects=self.objects, beta=_HARDEN_BETA
                        )
                    except TypeError:
                        changed = self.logger.log_params(iter_idx=epoch, params=params, objects=self.objects)
                    info["changed_voxels"] = changed
                    if last_aux is not None and hasattr(last_aux, "detector_states"):
                        self.logger.log_detectors(
                            iter_idx=epoch, objects=self.objects, detector_states=last_aux.detector_states
                        )
                self.logger.write(info)
                if progress_task is not None:
                    self.logger.progress.update(progress_task, advance=1)

            if self.epoch_callback is not None and epoch % self.log_every == 0:
                try:
                    self.epoch_callback(
                        epoch=epoch,
                        params=params,
                        objects=self.objects,
                        arrays=last_aux,
                        info=info,
                        optimization=self,
                    )
                except Exception as exc:
                    _log.warning(f"epoch_callback failed at epoch {epoch}: {exc!r}")

            if ckpt_dir is not None and (epoch % self.checkpoint_every == 0 or epoch == self.total_epochs - 1):
                save_checkpoint(ckpt_dir, epoch=epoch, params=params, opt_state=opt_state, rng_key=key)

        return self.aset("params", params)

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
