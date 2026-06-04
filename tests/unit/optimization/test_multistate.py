"""Unit tests for :mod:`fdtdx.optimization.multistate`.

No FDTD here — "states" are toy ``params -> metric`` maps, so we can check the
split-pass gradient against a single ``jax.value_and_grad`` reference and exercise
the :class:`MultiStateOptimization` loop cheaply.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import optax

from fdtdx.optimization.multistate import (
    MultiStateOptimization,
    accumulated_value_and_grad,
)


def test_primitive_matches_single_pass():
    """accumulated_value_and_grad == one value_and_grad over the combined loss."""
    params = {"x": jnp.array([1.0, -2.0, 0.5])}
    fa = lambda p: jnp.sum(p["x"] ** 2)  # noqa: E731
    fb = lambda p: jnp.sum(jnp.cos(p["x"]))  # noqa: E731

    def combine(ms):
        d = ms["a"] - ms["b"] - 0.3
        return d * d, {"diff": d}

    loss, grad, info, metrics = accumulated_value_and_grad(params, {"a": fa, "b": fb}, combine)

    def ref(p):
        return combine({"a": fa(p), "b": fb(p)})[0]

    rloss, rgrad = jax.value_and_grad(ref)(params)
    assert np.isfinite(float(loss)) and abs(float(loss) - float(rloss)) < 1e-5
    assert np.allclose(np.asarray(grad["x"]), np.asarray(rgrad["x"]), atol=1e-5)
    assert "diff" in info
    assert set(metrics) == {"a", "b"}


def test_primitive_complex_metric_phase():
    """Complex per-state metric (a phase via jnp.angle) composes correctly."""
    params = {"x": jnp.array([1.0, 2.0])}
    fc = lambda p: p["x"][0] + 1j * p["x"][1]  # noqa: E731

    def combine(ms):
        return (jnp.angle(ms["c"]) - 0.5) ** 2, {}

    loss, grad, _, _ = accumulated_value_and_grad(params, {"c": fc}, combine)

    def ref(p):
        return combine({"c": fc(p)})[0]

    rloss, rgrad = jax.value_and_grad(ref)(params)
    assert abs(float(loss) - float(rloss)) < 1e-5
    assert np.allclose(np.asarray(grad["x"]), np.asarray(rgrad["x"]), atol=1e-5)
    assert np.all(np.isfinite(np.asarray(grad["x"])))


def _toy_states():
    """Two states; combined loss sum(x^2)+sum((x-2)^2) is minimised at x=1."""
    fa = lambda p, k, e: (jnp.sum(p["x"] ** 2), None)  # noqa: E731
    fb = lambda p, k, e: (jnp.sum((p["x"] - 2.0) ** 2), None)  # noqa: E731

    def combine(ms):
        return ms["a"] + ms["b"], {"m_a": ms["a"], "m_b": ms["b"]}

    return {"a": fa, "b": fb}, combine


def test_driver_one_step_matches_sgd():
    """One epoch of the driver == one analytic SGD step (grad = 4x - 4)."""
    states, combine = _toy_states()
    x0 = jnp.array([0.0, 1.0, 3.0])
    opt = MultiStateOptimization(
        state_fns=states,
        combine=combine,
        optimizer=optax.sgd(0.1),
        params={"x": x0},
        total_epochs=1,
        param_clip=None,
        logger=None,
    )
    final = opt.run(key=jax.random.PRNGKey(0))
    expected = x0 - 0.1 * (4.0 * x0 - 4.0)
    assert np.allclose(np.asarray(final.params["x"]), np.asarray(expected), atol=1e-5)


def test_driver_reduces_loss():
    """Adam drives the design toward the analytic optimum x=1."""
    states, combine = _toy_states()
    opt = MultiStateOptimization(
        state_fns=states,
        combine=combine,
        optimizer=optax.adam(0.2),
        params={"x": jnp.array([0.0, 0.0, 0.0])},
        total_epochs=60,
        param_clip=None,
        logger=None,
    )
    final = opt.run(key=jax.random.PRNGKey(0))
    assert np.allclose(np.asarray(final.params["x"]), 1.0, atol=0.1)


def test_driver_resume(tmp_path):
    """Checkpoint round-trip: save during one run, resume in another."""
    states, combine = _toy_states()
    kw = dict(
        state_fns=states,
        combine=combine,
        optimizer=optax.adam(0.1),
        params={"x": jnp.array([0.0, 0.0, 0.0])},
        param_clip=None,
        logger=None,
        checkpoint_dir=str(tmp_path),
        checkpoint_every=1,
    )
    MultiStateOptimization(total_epochs=3, **kw).run(key=jax.random.PRNGKey(0))
    final = MultiStateOptimization(total_epochs=5, **kw).run(
        key=jax.random.PRNGKey(0), resume_from=str(tmp_path)
    )
    assert np.all(np.isfinite(np.asarray(final.params["x"])))
