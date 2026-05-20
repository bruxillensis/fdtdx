"""Tests for epoch-gated weight schedules."""

import jax
import jax.numpy as jnp
import pytest

from fdtdx.optimization.schedules import (
    ConstantSchedule,
    CosineSchedule,
    ExponentialSchedule,
    LinearSchedule,
    OnOffSchedule,
)


class TestConstantSchedule:
    """Test ConstantSchedule behaviour."""

    def test_inside_window_returns_value(self):
        """Inside the active window the constant value is emitted."""
        schedule = ConstantSchedule(value=0.5, epoch_start=10, epoch_end=20)
        result = schedule(15)
        assert isinstance(result, jax.Array)
        assert jnp.isclose(result, 0.5)

    def test_outside_window_returns_zero(self):
        """Outside the active window the schedule returns 0."""
        schedule = ConstantSchedule(value=0.5, epoch_start=10, epoch_end=20)
        assert jnp.isclose(schedule(5), 0.0)
        assert jnp.isclose(schedule(25), 0.0)

    def test_endpoints_inclusive(self):
        """Both epoch_start and epoch_end are inclusive."""
        schedule = ConstantSchedule(value=0.5, epoch_start=10, epoch_end=20)
        assert jnp.isclose(schedule(10), 0.5)
        assert jnp.isclose(schedule(20), 0.5)


class TestLinearSchedule:
    """Test LinearSchedule interpolation and gating."""

    def test_start_endpoint(self):
        """At exactly epoch_start the value equals start_value."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        assert jnp.isclose(schedule(10), 0.0)

    def test_end_endpoint(self):
        """At exactly epoch_end the value equals end_value."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        assert jnp.isclose(schedule(20), 1.0)

    def test_midpoint(self):
        """Midpoint of the window gives the midpoint of the value range."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        assert jnp.isclose(schedule(15), 0.5)

    def test_before_window(self):
        """Before epoch_start the schedule returns 0."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        assert jnp.isclose(schedule(5), 0.0)

    def test_after_window(self):
        """After epoch_end the schedule returns 0."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        assert jnp.isclose(schedule(25), 0.0)


class TestExponentialSchedule:
    """Test ExponentialSchedule log-linear interpolation."""

    def test_start_value_at_start(self):
        """At epoch_start the value is approximately start_value."""
        schedule = ExponentialSchedule(start_value=1e-3, end_value=1.0, epoch_start=0, epoch_end=100)
        assert jnp.isclose(schedule(0), 1e-3, atol=1e-6)

    def test_end_value_at_end(self):
        """At epoch_end the value is approximately end_value."""
        schedule = ExponentialSchedule(start_value=1e-3, end_value=1.0, epoch_start=0, epoch_end=100)
        assert jnp.isclose(schedule(100), 1.0, atol=1e-5)

    def test_monotonically_increasing(self):
        """Values across the window are monotonically increasing."""
        schedule = ExponentialSchedule(start_value=1e-3, end_value=1.0, epoch_start=0, epoch_end=100)
        samples = jnp.stack([schedule(e) for e in range(0, 101, 10)])
        diffs = jnp.diff(samples)
        assert jnp.all(diffs > 0)

    def test_log_linear_midpoint(self):
        """Midpoint equals sqrt(start * end) for log-linear interpolation."""
        schedule = ExponentialSchedule(start_value=1e-3, end_value=1.0, epoch_start=0, epoch_end=100)
        expected = jnp.sqrt(1e-3 * 1.0)
        assert jnp.isclose(schedule(50), expected, atol=1e-5)


class TestCosineSchedule:
    """Test CosineSchedule easing."""

    def test_endpoints_exact(self):
        """Cosine schedule hits start_value and end_value exactly at endpoints."""
        schedule = CosineSchedule(start_value=0.0, end_value=2.0, epoch_start=0, epoch_end=10)
        assert jnp.isclose(schedule(0), 0.0, atol=1e-6)
        assert jnp.isclose(schedule(10), 2.0, atol=1e-6)

    def test_midpoint_is_halfway(self):
        """At the midpoint cosine easing gives start + 0.5 * (end - start)."""
        schedule = CosineSchedule(start_value=0.0, end_value=2.0, epoch_start=0, epoch_end=10)
        expected = 0.0 + 0.5 * (2.0 - 0.0)
        assert jnp.isclose(schedule(5), expected, atol=1e-6)

    def test_monotonic(self):
        """Cosine easing is monotonic across the window."""
        schedule = CosineSchedule(start_value=0.0, end_value=2.0, epoch_start=0, epoch_end=10)
        samples = jnp.stack([schedule(e) for e in range(11)])
        diffs = jnp.diff(samples)
        # All increments non-negative (monotonic non-decreasing). Strictly
        # positive in the interior.
        assert jnp.all(diffs >= -1e-6)
        assert jnp.all(diffs[1:-1] > 0)

    def test_outside_window(self):
        """Returns 0 outside the window."""
        schedule = CosineSchedule(start_value=0.0, end_value=2.0, epoch_start=5, epoch_end=10)
        assert jnp.isclose(schedule(0), 0.0)
        assert jnp.isclose(schedule(15), 0.0)


class TestOnOffSchedule:
    """Test OnOffSchedule binary behaviour."""

    def test_before_window(self):
        """Returns 0 at epoch=4 (just before epoch_start=5)."""
        schedule = OnOffSchedule(value=3.0, epoch_start=5, epoch_end=15)
        assert jnp.isclose(schedule(4), 0.0)

    def test_at_start(self):
        """Returns value at epoch_start."""
        schedule = OnOffSchedule(value=3.0, epoch_start=5, epoch_end=15)
        assert jnp.isclose(schedule(5), 3.0)

    def test_mid_window(self):
        """Returns value inside the window."""
        schedule = OnOffSchedule(value=3.0, epoch_start=5, epoch_end=15)
        assert jnp.isclose(schedule(10), 3.0)

    def test_at_end(self):
        """Returns value at epoch_end (inclusive)."""
        schedule = OnOffSchedule(value=3.0, epoch_start=5, epoch_end=15)
        assert jnp.isclose(schedule(15), 3.0)

    def test_after_window(self):
        """Returns 0 at epoch=16 (just past epoch_end=15)."""
        schedule = OnOffSchedule(value=3.0, epoch_start=5, epoch_end=15)
        assert jnp.isclose(schedule(16), 0.0)


class TestJITCompatibility:
    """Test that schedules are JIT-compilable with traced epoch arguments."""

    def test_linear_jit_python_int(self):
        """JIT-wrapped schedule works with a Python int argument."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        jitted = jax.jit(lambda e: schedule(e))
        result = jitted(15)
        assert isinstance(result, jax.Array)
        assert jnp.isclose(result, 0.5)

    def test_linear_jit_jnp_scalar(self):
        """JIT-wrapped schedule works with a jnp scalar argument."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        jitted = jax.jit(lambda e: schedule(e))
        result = jitted(jnp.asarray(15))
        assert isinstance(result, jax.Array)
        assert jnp.isclose(result, 0.5)

    def test_linear_jit_outside_window(self):
        """JIT-wrapped schedule gates correctly for out-of-window epochs."""
        schedule = LinearSchedule(start_value=0.0, end_value=1.0, epoch_start=10, epoch_end=20)
        jitted = jax.jit(lambda e: schedule(e))
        assert jnp.isclose(jitted(jnp.asarray(5)), 0.0)
        assert jnp.isclose(jitted(jnp.asarray(25)), 0.0)


class TestEpochEndNone:
    """Test that epoch_end=None leaves the schedule active indefinitely."""

    def test_at_start(self):
        """Unbounded schedule immediately saturates at end_value once active."""
        schedule = LinearSchedule(start_value=0.25, end_value=1.0, epoch_start=10, epoch_end=None)
        # An unbounded window has no defined ramp duration, so the
        # schedule jumps straight to end_value at epoch_start. (See
        # WeightSchedule.__call__ for the saturation rule.)
        assert jnp.isclose(schedule(10), 1.0)

    def test_far_in_future(self):
        """Unbounded schedule at a large epoch still emits a nonzero weight."""
        schedule = LinearSchedule(start_value=0.25, end_value=1.0, epoch_start=10, epoch_end=None)
        # With end=inf, normalized time saturates at 1 via the clip; value
        # approaches end_value. Most importantly it must be nonzero (window
        # never turns off).
        far = schedule(10 + 1000)
        assert far > 0.0
        # t is clipped to [0, 1] so the ramp saturates at end_value.
        assert jnp.isclose(far, 1.0)

    def test_before_start_is_still_zero(self):
        """Even with epoch_end=None, before epoch_start the weight is 0."""
        schedule = LinearSchedule(start_value=0.25, end_value=1.0, epoch_start=10, epoch_end=None)
        assert jnp.isclose(schedule(5), 0.0)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
