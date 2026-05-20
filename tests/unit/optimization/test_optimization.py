"""Unit tests for the Optimization orchestrator.

These tests use a stubbed ``simulate_fn`` - they DO NOT run FDTD. FDTD-
integrated coverage lives in ``tests/simulation/``.

Design choice on stand-ins:
    The ``Optimization`` class declares ``arrays: field()`` (traced) and
    ``objects: frozen_field()`` (not traced). A plain ``object()`` sentinel
    works for frozen fields, but traced fields are pytree-flattened at
    ``TreeClass`` construction time, so a ``jnp.zeros(1)`` leaf is used
    instead. The stub ``simulate_fn`` passes ``arrays`` through unchanged.
"""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fdtdx.optimization.objectives.function import FunctionObjective
from fdtdx.optimization.optimization import Optimization
from fdtdx.optimization.schedules import ConstantSchedule, LinearSchedule
from fdtdx.optimization.utils.cli import build_arg_parser


def _make_stub_setup(total_epochs: int = 3, **overrides):
    """A tiny stand-in for the real FDTD setup.

    ``params = {"dev": jnp.array(...)}`` is the only thing that changes during
    optimisation. ``simulate_fn`` returns ``arrays`` unchanged; the objective
    function reads ``params["dev"]`` directly.
    """
    # `objects` is frozen_field in Optimization, so any sentinel is fine.
    # `arrays` is field() (traced); use a tiny jax-leaf stand-in so pytree
    # flattening at construction time succeeds.
    objects = object()  # any sentinel - frozen_field, never traversed
    arrays = jnp.zeros(1, dtype=jnp.float32)
    config = object()
    initial_params = {"dev": jnp.zeros((4,), dtype=jnp.float32)}

    def simulate(params, arrays, objects, config, key, epoch):
        del params, objects, config, key, epoch
        return arrays

    # Objective: want params["dev"] as close to 1.0 as possible.
    # raw = -||p - 1||^2 (positive-is-better convention).
    # With sign=-1 from Objective, contribution = +||p-1||^2, so loss decreases
    # as p approaches 1.
    def metric_fn(*, params, objects, arrays, config, epoch):
        del objects, arrays, config, epoch
        p = params["dev"]
        val = -jnp.sum((p - 1.0) ** 2)
        return val, {}

    obj = FunctionObjective(
        name="target",
        schedule=ConstantSchedule(value=1.0),
        fn=metric_fn,
    )

    optimizer = optax.adam(learning_rate=0.1)

    opt = Optimization(
        objects=objects,
        arrays=arrays,
        params=initial_params,
        config=config,
        simulate_fn=simulate,
        optimizer=optimizer,
        objectives=(obj,),
        total_epochs=total_epochs,
        logger=None,
        param_clip=None,
        **overrides,
    )
    return opt


class TestOptimizationLoop:
    def test_params_approach_target(self, tmp_path):
        """Over a few epochs of adam, params should move toward 1.0."""
        opt = _make_stub_setup(total_epochs=30, checkpoint_dir=tmp_path)
        key = jax.random.PRNGKey(0)
        final = opt.run(key=key)
        # Final params should be significantly closer to 1.0 than 0.0 start.
        p = final.params["dev"]
        assert jnp.all(p > 0.5)
        # Adam at lr=0.1 overshoots the minimum — bound loose enough to
        # accommodate the typical late-epoch oscillation.
        assert jnp.all(p <= 1.1)

    def test_seed_and_resume_mutually_exclusive(self):
        opt = _make_stub_setup()
        with pytest.raises(ValueError, match="mutually exclusive"):
            opt.run(key=jax.random.PRNGKey(0), seed_from="a", resume_from="b")

    def test_checkpoint_then_resume(self, tmp_path):
        """Run 3 epochs, checkpoint, resume from the checkpoint, run 3 more."""
        opt1 = _make_stub_setup(total_epochs=3, checkpoint_dir=tmp_path, checkpoint_every=1)
        final1 = opt1.run(key=jax.random.PRNGKey(0))
        p_after_first = final1.params["dev"]

        # Fresh Optimization (simulating a new process), then resume.
        opt2 = _make_stub_setup(total_epochs=6, checkpoint_dir=tmp_path, checkpoint_every=1)
        final2 = opt2.run(key=jax.random.PRNGKey(0), resume_from=tmp_path)
        p_after_resume = final2.params["dev"]

        # After 6 epochs total, params should be closer to target than after 3.
        assert jnp.sum((p_after_resume - 1.0) ** 2) < jnp.sum((p_after_first - 1.0) ** 2)

    def test_seed_loads_params_not_opt_state(self, tmp_path):
        """Seed path should load params but NOT restore epoch or opt_state."""
        seed_dir = tmp_path / "seed"
        seed_dir.mkdir()
        seed_values = np.array([0.9, 0.9, 0.9, 0.9], dtype=np.float32)
        np.save(seed_dir / "params_10_dev.npy", seed_values)

        opt = _make_stub_setup(total_epochs=2, checkpoint_dir=None)
        final = opt.run(key=jax.random.PRNGKey(0), seed_from=seed_dir)
        # After 2 more adam steps starting from 0.9, we should have moved
        # further. If seed wasn't applied we'd have started from 0.0 and ended
        # up near ~0.2.
        p = final.params["dev"]
        assert jnp.all(p > 0.85), f"params should reflect the seed (~0.9), got {p}"

    def test_scheduled_weight_gates_loss(self):
        """When a term's schedule is 0 for epochs < window, loss should stay 0."""

        def nonzero_fn(*, params, objects, arrays, config, epoch):
            del params, objects, arrays, config, epoch
            return jnp.asarray(42.0), {}

        term = FunctionObjective(
            name="gated",
            schedule=LinearSchedule(
                epoch_start=100,
                epoch_end=200,
                start_value=0.0,
                end_value=1.0,
            ),
            fn=nonzero_fn,
        )
        opt = _make_stub_setup(total_epochs=3)
        opt = opt.aset("objectives", (term,))

        # Run loss_fn directly at epoch=0 - should be 0 since schedule gates
        # the term off entirely outside its window.
        loss, (_, info) = opt.loss_fn(
            opt.params,
            opt.arrays,
            jax.random.PRNGKey(0),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        assert jnp.isclose(loss, 0.0), f"gated term should contribute 0, got {loss}"

    def test_param_clip_enforced(self):
        """With param_clip=(0, 1), params should always stay in that range."""
        opt = _make_stub_setup(total_epochs=20)

        # Objective that would push params toward 5.0 without a clip.
        def overshooter(*, params, objects, arrays, config, epoch):
            del objects, arrays, config, epoch
            return -jnp.sum((params["dev"] - 5.0) ** 2), {}

        term = FunctionObjective(
            name="over",
            schedule=ConstantSchedule(value=1.0),
            fn=overshooter,
        )
        opt = opt.aset("objectives", (term,)).aset("param_clip", (0.0, 1.0))
        final = opt.run(key=jax.random.PRNGKey(0))
        p = final.params["dev"]
        assert jnp.all(p >= 0.0) and jnp.all(p <= 1.0)


class TestCliParser:
    def test_default_values(self):
        parser = build_arg_parser("test")
        args = parser.parse_args([])
        assert args.seed_rng == 0
        assert args.evaluation is False
        assert args.backward is False
        assert args.seed_from is None
        assert args.resume_from is None
        assert args.seed_iter is None

    def test_parses_flags(self):
        parser = build_arg_parser("test")
        args = parser.parse_args(["--seed-rng", "42", "--evaluation", "--seed-from", "/tmp/x", "--seed-iter", "7"])
        assert args.seed_rng == 42
        assert args.evaluation is True
        assert args.seed_from == "/tmp/x"
        assert args.seed_iter == "7"

    def test_extensible(self):
        """User scripts should be able to add their own flags."""
        parser = build_arg_parser("test")
        parser.add_argument("--custom", type=float, default=1.5)
        args = parser.parse_args(["--custom", "2.5"])
        assert args.custom == 2.5
