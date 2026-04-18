"""Checkpoint save/load for the fdtdx optimization suite.

Two APIs:

- `save_checkpoint` / `load_checkpoint` — round-trip a full optimization state
  (params + optax state + epoch + RNG key). Used by `--resume-from`.
- `load_seed_params` — read params only from a prior run's Logger output dir.
  Used by `--seed-from` for warm-starting a fresh optimization.
"""

import json
import re
from pathlib import Path
from typing import Any, cast

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from loguru import logger

from fdtdx.fdtd.container import ParameterContainer

CHECKPOINT_FILE_GLOB = "checkpoint_*.eqx"
CHECKPOINT_META_SUFFIX = ".json"


def save_checkpoint(
    checkpoint_dir: Path | str,
    *,
    epoch: int,
    params: ParameterContainer,
    opt_state: optax.OptState,
    rng_key: jax.Array,
) -> Path:
    """Save a full optimization state for later resumption.

    Writes two files:
        {checkpoint_dir}/checkpoint_{epoch:06d}.eqx        # equinox-serialised leaves
        {checkpoint_dir}/checkpoint_{epoch:06d}.json       # metadata (epoch)

    Args:
        checkpoint_dir: Directory to write into (created if missing).
        epoch: Current epoch index. Encoded in the filename for easy discovery.
        params: ParameterContainer to serialise.
        opt_state: optax state (any pytree) to serialise.
        rng_key: Current RNG key.

    Returns:
        Path to the written .eqx file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "params": params,
        "opt_state": opt_state,
        "rng_key": rng_key,
    }
    eqx_path = checkpoint_dir / f"checkpoint_{epoch:06d}.eqx"
    meta_path = checkpoint_dir / f"checkpoint_{epoch:06d}.json"
    eqx.tree_serialise_leaves(str(eqx_path), state)
    meta_path.write_text(json.dumps({"epoch": int(epoch)}))
    logger.info(f"Saved checkpoint at epoch {epoch} -> {eqx_path}")
    return eqx_path


def load_checkpoint(
    checkpoint_path: Path | str,
    *,
    params_template: ParameterContainer,
    opt_state_template: optax.OptState,
    rng_key_template: jax.Array | None = None,
) -> tuple[int, ParameterContainer, optax.OptState, jax.Array]:
    """Load a full optimization state previously written by `save_checkpoint`.

    Args:
        checkpoint_path: Either a direct path to a `.eqx` file, or a directory
            containing one or more `checkpoint_*.eqx` files (the highest-epoch
            checkpoint is auto-selected).
        params_template: A ParameterContainer with the correct tree structure
            and dtypes — values are overwritten by the loaded leaves.
        opt_state_template: An optax state with the correct structure.
        rng_key_template: An optional RNG key template. Defaults to
            `jax.random.PRNGKey(0)` if not provided (its leaf is overwritten
            by the loaded value so the template's seed doesn't matter).

    Returns:
        Tuple of (epoch, params, opt_state, rng_key).
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        eqx_path = _find_latest_checkpoint_file(checkpoint_path)
    elif checkpoint_path.is_file():
        eqx_path = checkpoint_path
    else:
        raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")

    meta_path = eqx_path.with_suffix(CHECKPOINT_META_SUFFIX)
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint metadata: {meta_path}")
    meta = json.loads(meta_path.read_text())
    epoch = int(meta["epoch"])

    if rng_key_template is None:
        rng_key_template = jax.random.PRNGKey(0)
    template: dict[str, Any] = {
        "params": params_template,
        "opt_state": opt_state_template,
        "rng_key": rng_key_template,
    }
    state = eqx.tree_deserialise_leaves(str(eqx_path), template)
    logger.info(f"Loaded checkpoint from {eqx_path} (epoch={epoch})")
    return epoch, state["params"], state["opt_state"], state["rng_key"]


def _find_latest_checkpoint_file(checkpoint_dir: Path) -> Path:
    pattern = re.compile(r"^checkpoint_(\d+)\.eqx$")
    candidates: list[tuple[int, Path]] = []
    for p in checkpoint_dir.iterdir():
        m = pattern.match(p.name)
        if m:
            candidates.append((int(m.group(1)), p))
    if not candidates:
        raise FileNotFoundError(f"No checkpoint_*.eqx files in {checkpoint_dir}")
    candidates.sort()
    return candidates[-1][1]


def load_seed_params(
    seed_path: Path | str,
    params_template: ParameterContainer,
    iter_idx: int | None = None,
) -> ParameterContainer:
    """Seed a fresh optimization by loading params from a prior run's Logger output.

    Reads `params_{iter}_{device}.npy` files (or `params_{iter}_{device}_{key}.npy`
    for dict-valued device params) produced by `fdtdx.utils.logger.Logger.log_params`.

    Args:
        seed_path: Directory containing the `params_*.npy` files (typically
            `{logger.cwd}/params/`).
        params_template: A ParameterContainer whose keys, shapes, and dtypes
            match the target structure. Values are replaced.
        iter_idx: Iteration index to load. If None, auto-selects the highest
            iter that has files for every device.

    Returns:
        A new ParameterContainer with loaded values. Shapes and dtypes match
        `params_template`.
    """
    seed_dir = Path(seed_path)
    if not seed_dir.is_dir():
        raise FileNotFoundError(f"Seed directory does not exist: {seed_dir}")

    device_names = list(params_template.keys())
    if iter_idx is None:
        iter_idx = _find_latest_seed_iter(seed_dir, device_names)
        logger.info(f"Auto-selected seed iter_idx={iter_idx} from {seed_dir}")
    else:
        logger.info(f"Loading seed iter_idx={iter_idx} from {seed_dir}")

    new_params: ParameterContainer = {}
    for name, current in params_template.items():
        if isinstance(current, dict):
            current_dict = cast(dict[str, jax.Array], current)
            loaded: dict[str, jax.Array] = {}
            for k, v in current_dict.items():
                key_str: str = k
                path = seed_dir / f"params_{iter_idx}_{name}_{key_str}.npy"
                if not path.is_file():
                    raise FileNotFoundError(f"Missing seed file: {path}")
                arr = np.load(path)
                if tuple(arr.shape) != tuple(v.shape):
                    raise ValueError(f"Shape mismatch for {name}[{key_str}]: seed {arr.shape} vs expected {v.shape}")
                loaded[key_str] = jnp.asarray(arr, dtype=v.dtype)
            new_params[name] = loaded
        else:
            current_arr = cast(jax.Array, current)
            path = seed_dir / f"params_{iter_idx}_{name}.npy"
            if not path.is_file():
                raise FileNotFoundError(f"Missing seed file: {path}")
            arr = np.load(path)
            if tuple(arr.shape) != tuple(current_arr.shape):
                raise ValueError(f"Shape mismatch for {name}: seed {arr.shape} vs expected {current_arr.shape}")
            new_params[name] = jnp.asarray(arr, dtype=current_arr.dtype)
        logger.info(f"  loaded seed for device '{name}'")
    return new_params


def _find_latest_seed_iter(params_dir: Path, device_names: list[str]) -> int:
    pattern = re.compile(r"^params_(-?\d+)_(.+)\.npy$")
    iters_per_device: dict[str, set[int]] = {n: set() for n in device_names}
    for p in params_dir.iterdir():
        m = pattern.match(p.name)
        if not m:
            continue
        iter_idx_ = int(m.group(1))
        rest = m.group(2)
        sorted_devices = cast(list[str], sorted(device_names, key=len, reverse=True))
        for dev_name in sorted_devices:
            if rest == dev_name or rest.startswith(dev_name + "_"):
                iters_per_device[dev_name].add(iter_idx_)
                break
    common = set.intersection(*iters_per_device.values()) if iters_per_device else set()
    common.discard(-1)
    if not common:
        missing = [n for n, s in iters_per_device.items() if not s]
        raise FileNotFoundError(
            f"No common iteration across all devices in {params_dir}. Devices with no matching files: {missing}"
        )
    return max(common)
