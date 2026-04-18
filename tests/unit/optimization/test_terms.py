"""Unit tests for `fdtdx.optimization.terms` and the Function* wrappers."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.optimization.constraints.function import FunctionConstraint
from fdtdx.optimization.objectives.function import FunctionObjective
from fdtdx.optimization.schedules import ConstantSchedule, LinearSchedule


@pytest.mark.unit
def test_objective_signed_contribution():
    """Objective with sign=-1 should contribute -weight * raw."""

    def fn(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(3.0), {}

    obj = FunctionObjective(
        name="obj1",
        schedule=ConstantSchedule(value=2.0),
        fn=fn,
    )
    contribution, info = obj(
        sign=-1.0,
        params=None,
        objects=None,
        arrays=None,
        config=None,
        epoch=jnp.asarray(0),
    )
    assert jnp.allclose(contribution, -6.0)
    assert "obj1_raw" in info
    assert "obj1_weight" in info
    assert "obj1_contrib" in info
    assert jnp.allclose(info["obj1_raw"], 3.0)
    assert jnp.allclose(info["obj1_weight"], 2.0)
    assert jnp.allclose(info["obj1_contrib"], -6.0)


@pytest.mark.unit
def test_constraint_signed_contribution():
    """Constraint with sign=+1 should contribute +weight * raw."""

    def fn(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(4.0), {}

    con = FunctionConstraint(
        name="con1",
        schedule=ConstantSchedule(value=0.5),
        fn=fn,
    )
    contribution, info = con(
        sign=1.0,
        params=None,
        objects=None,
        arrays=None,
        config=None,
        epoch=jnp.asarray(0),
    )
    assert jnp.allclose(contribution, 2.0)
    assert jnp.allclose(info["con1_contrib"], 2.0)


@pytest.mark.unit
def test_inactive_term_contributes_zero():
    """A term with a schedule returning 0 outside its window must contribute 0."""

    def fn(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(100.0), {}

    obj = FunctionObjective(
        name="ramp",
        schedule=LinearSchedule(
            epoch_start=10,
            epoch_end=20,
            start_value=0.0,
            end_value=1.0,
        ),
        fn=fn,
    )

    # Epoch before active window -> zero
    contrib_before, _ = obj(
        sign=-1.0,
        params=None,
        objects=None,
        arrays=None,
        config=None,
        epoch=jnp.asarray(5),
    )
    assert jnp.allclose(contrib_before, 0.0)

    # Epoch at end of active window -> weight == end_value == 1.0
    contrib_end, _ = obj(
        sign=-1.0,
        params=None,
        objects=None,
        arrays=None,
        config=None,
        epoch=jnp.asarray(20),
    )
    assert jnp.allclose(contrib_end, -100.0)


@pytest.mark.unit
def test_info_dict_merge():
    """User info dict entries must survive alongside the auto-added keys."""

    def fn(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(1.0), {"extra_stat": jnp.asarray(42.0)}

    obj = FunctionObjective(
        name="merged",
        schedule=ConstantSchedule(value=1.0),
        fn=fn,
    )
    _, info = obj(
        sign=-1.0,
        params=None,
        objects=None,
        arrays=None,
        config=None,
        epoch=jnp.asarray(0),
    )
    assert "extra_stat" in info
    assert jnp.allclose(info["extra_stat"], 42.0)
    assert "merged_raw" in info
    assert "merged_weight" in info
    assert "merged_contrib" in info


@pytest.mark.unit
def test_jit_compile():
    """Terms must be usable under jax.jit with a traced epoch."""

    def fn_a(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(1.5), {}

    def fn_b(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(2.5), {}

    obj_a = FunctionObjective(name="a", schedule=ConstantSchedule(value=1.0), fn=fn_a)
    obj_b = FunctionObjective(name="b", schedule=ConstantSchedule(value=1.0), fn=fn_b)

    @jax.jit
    def total(epoch):
        ca, _ = obj_a(
            sign=-1.0,
            params=None,
            objects=None,
            arrays=None,
            config=None,
            epoch=epoch,
        )
        cb, _ = obj_b(
            sign=-1.0,
            params=None,
            objects=None,
            arrays=None,
            config=None,
            epoch=epoch,
        )
        return ca + cb

    result = total(jnp.asarray(0))
    assert isinstance(result, jax.Array)
    assert jnp.allclose(result, -4.0)


@pytest.mark.unit
def test_multi_term_loss():
    """Two objectives and one constraint should sum to the expected scalar."""

    def fn_one(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(1.0), {}

    def fn_two(*, params, objects, arrays, config, epoch):
        del params, objects, arrays, config, epoch
        return jnp.asarray(2.0), {}

    obj_a = FunctionObjective(name="oa", schedule=ConstantSchedule(value=0.5), fn=fn_one)
    obj_b = FunctionObjective(name="ob", schedule=ConstantSchedule(value=0.5), fn=fn_one)
    con = FunctionConstraint(name="cc", schedule=ConstantSchedule(value=1.0), fn=fn_two)

    ctx = dict(
        params=None,
        objects=None,
        arrays=None,
        config=None,
        epoch=jnp.asarray(0),
    )
    ca, _ = obj_a(sign=-1.0, **ctx)
    cb, _ = obj_b(sign=-1.0, **ctx)
    cc, _ = con(sign=1.0, **ctx)

    total = ca + cb + cc
    # -0.5 + -0.5 + 2.0 = 1.0
    assert jnp.allclose(total, 1.0)
