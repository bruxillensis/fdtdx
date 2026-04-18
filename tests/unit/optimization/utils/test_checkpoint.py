"""Tests for fdtdx.optimization.utils.checkpoint."""

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from fdtdx.optimization.utils.checkpoint import (
    load_checkpoint,
    load_seed_params,
    save_checkpoint,
)

pytestmark = pytest.mark.unit


def _make_params_and_opt_state():
    params = {"mydevice": jnp.arange(9, dtype=jnp.float32).reshape(3, 3)}
    optimizer = optax.adam(1e-3)
    opt_state = optimizer.init(params)
    return params, opt_state, optimizer


def test_save_load_roundtrip(tmp_path):
    params, opt_state, _ = _make_params_and_opt_state()
    rng_key = jax.random.PRNGKey(42)

    save_checkpoint(
        tmp_path,
        epoch=7,
        params=params,
        opt_state=opt_state,
        rng_key=rng_key,
    )

    # Build templates with different but same-shape/dtype content to prove values
    # were actually loaded (not left from the template).
    params_template = {"mydevice": jnp.zeros((3, 3), dtype=jnp.float32)}
    _, opt_state_template, _ = _make_params_and_opt_state()
    rng_key_template = jax.random.PRNGKey(0)

    loaded_epoch, loaded_params, loaded_opt_state, loaded_rng = load_checkpoint(
        tmp_path,
        params_template=params_template,
        opt_state_template=opt_state_template,
        rng_key_template=rng_key_template,
    )

    assert loaded_epoch == 7
    assert jnp.array_equal(loaded_params["mydevice"], params["mydevice"])
    assert jnp.array_equal(loaded_rng, rng_key)

    # Compare the full opt_state pytree structure and all leaves.
    orig_leaves, orig_tree = jax.tree_util.tree_flatten(opt_state)
    loaded_leaves, loaded_tree = jax.tree_util.tree_flatten(loaded_opt_state)
    assert orig_tree == loaded_tree
    assert len(orig_leaves) == len(loaded_leaves)
    for a, b in zip(orig_leaves, loaded_leaves):
        a_arr = jnp.asarray(a)
        b_arr = jnp.asarray(b)
        assert a_arr.shape == b_arr.shape
        assert jnp.array_equal(a_arr, b_arr)


def test_load_latest_in_dir(tmp_path):
    params, opt_state, _ = _make_params_and_opt_state()
    rng_key = jax.random.PRNGKey(0)
    for epoch in (1, 5, 12):
        save_checkpoint(
            tmp_path,
            epoch=epoch,
            params=params,
            opt_state=opt_state,
            rng_key=rng_key,
        )

    params_template = {"mydevice": jnp.zeros((3, 3), dtype=jnp.float32)}
    _, opt_state_template, _ = _make_params_and_opt_state()

    loaded_epoch, _, _, _ = load_checkpoint(
        tmp_path,
        params_template=params_template,
        opt_state_template=opt_state_template,
    )
    assert loaded_epoch == 12


def test_load_missing_raises(tmp_path):
    nonexistent = tmp_path / "does_not_exist"
    params_template = {"mydevice": jnp.zeros((3, 3), dtype=jnp.float32)}
    _, opt_state_template, _ = _make_params_and_opt_state()

    with pytest.raises(FileNotFoundError):
        load_checkpoint(
            nonexistent,
            params_template=params_template,
            opt_state_template=opt_state_template,
        )


def test_load_seed_params_single_array(tmp_path):
    expected = np.arange(9, dtype=np.float32).reshape(3, 3)
    np.save(tmp_path / "params_5_mydevice.npy", expected)

    template = {"mydevice": jnp.zeros((3, 3), dtype=jnp.float32)}
    loaded = load_seed_params(tmp_path, template, iter_idx=5)

    assert set(loaded.keys()) == {"mydevice"}
    assert jnp.array_equal(loaded["mydevice"], jnp.asarray(expected))


def test_load_seed_params_dict_device(tmp_path):
    a_arr = np.full((2, 2), 3.0, dtype=np.float32)
    b_arr = np.full((2, 2), 7.0, dtype=np.float32)
    np.save(tmp_path / "params_5_mydevice_A.npy", a_arr)
    np.save(tmp_path / "params_5_mydevice_B.npy", b_arr)

    template = {
        "mydevice": {
            "A": jnp.zeros((2, 2), dtype=jnp.float32),
            "B": jnp.zeros((2, 2), dtype=jnp.float32),
        }
    }
    loaded = load_seed_params(tmp_path, template, iter_idx=5)

    assert set(loaded["mydevice"].keys()) == {"A", "B"}
    assert jnp.array_equal(loaded["mydevice"]["A"], jnp.asarray(a_arr))
    assert jnp.array_equal(loaded["mydevice"]["B"], jnp.asarray(b_arr))


def test_load_seed_params_iter_selection(tmp_path):
    arr_3 = np.full((3, 3), 3.0, dtype=np.float32)
    arr_7 = np.full((3, 3), 7.0, dtype=np.float32)
    np.save(tmp_path / "params_3_mydevice.npy", arr_3)
    np.save(tmp_path / "params_7_mydevice.npy", arr_7)

    template = {"mydevice": jnp.zeros((3, 3), dtype=jnp.float32)}

    # None -> auto-select latest
    loaded_latest = load_seed_params(tmp_path, template, iter_idx=None)
    assert jnp.array_equal(loaded_latest["mydevice"], jnp.asarray(arr_7))

    # Explicit iter=3
    loaded_3 = load_seed_params(tmp_path, template, iter_idx=3)
    assert jnp.array_equal(loaded_3["mydevice"], jnp.asarray(arr_3))


def test_load_seed_params_name_prefix_disambiguation(tmp_path):
    # Two devices whose names share a prefix — test the longest-name-first match.
    arr_short = np.full((2, 2), 1.0, dtype=np.float32)
    arr_long = np.full((2, 2), 2.0, dtype=np.float32)
    np.save(tmp_path / "params_5_Si_Layer_0.npy", arr_short)
    np.save(tmp_path / "params_5_Si_Layer_0_extra.npy", arr_long)

    template = {
        "Si_Layer_0": jnp.zeros((2, 2), dtype=jnp.float32),
        "Si_Layer_0_extra": jnp.zeros((2, 2), dtype=jnp.float32),
    }
    loaded = load_seed_params(tmp_path, template, iter_idx=5)
    assert jnp.array_equal(loaded["Si_Layer_0"], jnp.asarray(arr_short))
    assert jnp.array_equal(loaded["Si_Layer_0_extra"], jnp.asarray(arr_long))

    # And the auto-iter path must also be able to resolve the common iter
    # despite the ambiguous prefix.
    loaded_auto = load_seed_params(tmp_path, template, iter_idx=None)
    assert jnp.array_equal(loaded_auto["Si_Layer_0"], jnp.asarray(arr_short))
    assert jnp.array_equal(loaded_auto["Si_Layer_0_extra"], jnp.asarray(arr_long))


def test_load_seed_params_shape_mismatch(tmp_path):
    bad = np.zeros((2, 2), dtype=np.float32)
    np.save(tmp_path / "params_5_mydevice.npy", bad)

    template = {"mydevice": jnp.zeros((3, 3), dtype=jnp.float32)}
    with pytest.raises(ValueError):
        load_seed_params(tmp_path, template, iter_idx=5)
