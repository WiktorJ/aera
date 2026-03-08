"""Analyze how observations and actions change at various subsample skip intervals.

Usage:
    python -m aera.autonomous.openpi.scripts.analyze_skip_intervals --config <config_name> [--num_samples 5000] [--skips 1,5,10,20,50]

This loads the dataset using the same pipeline as train.py and computes
statistics on the deltas between observations (states) and actions at
different skip levels, helping to choose an appropriate subsample_interval.
"""

import argparse
import logging
import sys

import numpy as np
import openpi.models.model as _model
import openpi.training.data_loader as _data_loader
import torch

import aera.autonomous.openpi.training_config as _training_config


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def collect_states_and_actions(
    dataset: torch.utils.data.Dataset,
    num_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Collect state and action arrays sequentially from the dataset."""
    states = []
    actions = []
    n = min(num_samples, len(dataset))
    logging.info(f"Collecting {n} samples from dataset (total size: {len(dataset)})")

    for i in range(n):
        sample = dataset[i]
        if "state" in sample:
            states.append(np.asarray(sample["state"], dtype=np.float32))
        if "actions" in sample:
            actions.append(np.asarray(sample["actions"], dtype=np.float32))

    if not states:
        raise ValueError(
            "No 'state' key found in dataset samples. "
            "Check that the data transforms produce a 'state' field."
        )
    if not actions:
        raise ValueError(
            "No 'actions' key found in dataset samples. "
            "Check that the data transforms produce an 'actions' field."
        )

    return np.stack(states), np.stack(actions)


def compute_skip_statistics(
    data: np.ndarray,
    skips: list[int],
    label: str,
) -> None:
    """Compute and print delta statistics for various skip intervals."""
    print(f"\n{'=' * 70}")
    print(f"  Skip-interval statistics for: {label}")
    print(f"  Shape: {data.shape}")
    print(f"{'=' * 70}")
    print(f"{'Skip':>6} | {'Mean |Δ|':>12} | {'Std Δ':>12} | {'Max |Δ|':>12} | {'Median |Δ|':>12} | {'N pairs':>10}")
    print(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

    for skip in skips:
        if skip >= len(data):
            print(f"{skip:>6} | {'(skip >= N, skipped)':>53}")
            continue
        # Strided pairs: (0, skip), (skip, 2*skip), (2*skip, 3*skip), ...
        step = max(skip, 1)
        indices = np.arange(0, len(data) - skip, step)
        diffs = data[indices + skip] - data[indices]
        abs_diffs = np.abs(diffs)
        print(
            f"{skip:>6} | "
            f"{abs_diffs.mean():>12.6f} | "
            f"{diffs.std():>12.6f} | "
            f"{abs_diffs.max():>12.6f} | "
            f"{np.median(abs_diffs):>12.6f} | "
            f"n_pairs={len(indices)}"
        )


def compute_cross_statistics(
    states: np.ndarray,
    actions: np.ndarray,
    skips: list[int],
) -> None:
    """Compute delta statistics between actions and observations at various skip levels."""
    # Use the minimum dimensionality so we can compare them
    min_dim = min(states.shape[-1], actions.shape[-1])
    states_trimmed = states[..., :min_dim]
    # For actions with an action horizon, take the first timestep
    if actions.ndim == 3:
        actions_trimmed = actions[:, 0, :min_dim]
    else:
        actions_trimmed = actions[..., :min_dim]

    print(f"\n{'=' * 70}")
    print(f"  Cross statistics: actions[t+skip] - states[t]")
    print(f"  States shape: {states.shape}, Actions shape: {actions.shape}")
    print(f"  Comparing first {min_dim} dims")
    print(f"{'=' * 70}")
    print(f"{'Skip':>6} | {'Mean |Δ|':>12} | {'Std Δ':>12} | {'Max |Δ|':>12} | {'Median |Δ|':>12} | {'N pairs':>10}")
    print(f"{'-' * 6}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 12}-+-{'-' * 10}")

    n = min(len(states_trimmed), len(actions_trimmed))
    for skip in skips:
        if skip >= n:
            print(f"{skip:>6} | {'(skip >= N, skipped)':>65}")
            continue
        # Strided pairs: obs at t=0,skip,2*skip,... paired with action at t+skip
        step = max(skip, 1)
        indices = np.arange(0, n - skip, step)
        diffs = actions_trimmed[indices + skip] - states_trimmed[indices]
        abs_diffs = np.abs(diffs)
        print(
            f"{skip:>6} | "
            f"{abs_diffs.mean():>12.6f} | "
            f"{diffs.std():>12.6f} | "
            f"{abs_diffs.max():>12.6f} | "
            f"{np.median(abs_diffs):>12.6f} | "
            f"n_pairs={len(indices)}"
        )


def print_summary(states: np.ndarray, actions: np.ndarray) -> None:
    """Print basic summary statistics of the raw data."""
    print(f"\n{'=' * 70}")
    print(f"  Dataset Summary")
    print(f"{'=' * 70}")
    print(f"  States  — shape: {states.shape}, "
          f"mean: {states.mean():.6f}, std: {states.std():.6f}, "
          f"min: {states.min():.6f}, max: {states.max():.6f}")
    print(f"  Actions — shape: {actions.shape}, "
          f"mean: {actions.mean():.6f}, std: {actions.std():.6f}, "
          f"min: {actions.min():.6f}, max: {actions.max():.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze observation/action deltas at various subsample skip intervals."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Training config name (passed to training_config.get_config).",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=5000,
        help="Maximum number of sequential samples to load from the dataset.",
    )
    parser.add_argument(
        "--skips",
        type=str,
        default="1,5,10,20,50",
        help="Comma-separated list of skip intervals to analyze.",
    )
    return parser.parse_args()


def main():
    init_logging()
    args = parse_args()

    skips = [int(s) for s in args.skips.split(",")]
    logging.info(f"Config: {args.config}, num_samples: {args.num_samples}, skips: {skips}")

    # Load the training config the same way train.py does
    extended_config = _training_config.get_config(args.config)
    config = extended_config.base_config

    # Build the data config and create the raw torch dataset
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = _data_loader.create_torch_dataset(
        data_config, config.model.action_horizon, config.model
    )
    dataset = _data_loader.transform_dataset(
        dataset, data_config, skip_norm_stats=True
    )

    logging.info(f"Dataset size: {len(dataset)}")
    logging.info(f"Sample keys: {list(dataset[0].keys()) if len(dataset) > 0 else 'N/A'}")

    # Collect sequential data
    states, actions = collect_states_and_actions(dataset, args.num_samples)

    # Print results
    print_summary(states, actions)
    compute_skip_statistics(states, skips, label="States (observations)")
    compute_skip_statistics(actions.reshape(len(actions), -1), skips, label="Actions (flattened)")
    compute_cross_statistics(states, actions, skips)

    print(f"\n{'=' * 70}")
    print(f"  Analysis complete. {len(states)} samples analyzed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
