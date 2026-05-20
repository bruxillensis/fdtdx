"""Command-line argument helpers for fdtdx optimization scripts.

`build_arg_parser()` returns a pre-populated argparse.ArgumentParser with the
standard fdtdx optimization flags:

    --seed-rng       RNG seed for initialisation (int, default 0).
    --evaluation     Run in evaluation mode — compile loss only, no grad step.
    --backward       Run a reverse-time pass for diagnostic detectors.
    --seed-from PATH Warm-start params from a prior run's Logger output dir.
                     The optimizer state and epoch counter are re-initialised.
    --seed-iter IDX  Iteration index inside --seed-from to load ("latest" or int).
    --resume-from PATH  Resume a previously interrupted optimisation from a
                     checkpoint directory or .eqx file. Restores params,
                     optimizer state, epoch, and RNG.

Users typically extend the parser with their own script-specific flags before
calling `parse_args()`:

    parser = fdtdx.build_arg_parser(description="My optimisation")
    parser.add_argument("--resolution", type=float, default=25e-9)
    args = parser.parse_args()
"""

import argparse


def build_arg_parser(description: str = "fdtdx optimization") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--seed-rng",
        type=int,
        default=0,
        help="RNG seed for initialisation (default: 0).",
    )
    parser.add_argument(
        "--evaluation",
        action="store_true",
        help="Run in evaluation mode — compile the forward pass only, no gradient update.",
    )
    parser.add_argument(
        "--backward",
        action="store_true",
        help="Run a reverse-time pass for diagnostic detectors after the forward simulation.",
    )
    parser.add_argument(
        "--seed-from",
        type=str,
        default=None,
        help=(
            "Path to a directory of params_*.npy files from a prior run "
            "(typically {logger.cwd}/params). Loads only device params; "
            "optimizer state and epoch counter are re-initialised. "
            "Mutually exclusive with --resume-from."
        ),
    )
    parser.add_argument(
        "--seed-iter",
        type=str,
        default=None,
        help=(
            "Which iteration inside --seed-from to load. "
            "Either an integer or 'latest' (default: latest common across all devices)."
        ),
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help=(
            "Path to a checkpoint directory or .eqx file from a previous run. "
            "Restores params, optimizer state, epoch counter, and RNG key. "
            "Mutually exclusive with --seed-from."
        ),
    )
    return parser
