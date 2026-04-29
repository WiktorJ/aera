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
        # Strided pairs with offset: obs at t, data at t + (skip-1)
        # t = 0, skip, 2*skip, ... paired with t + skip - 1
        offset = skip - 1
        indices = np.arange(0, len(data) - offset, skip)
        diffs = data[indices + offset] - data[indices]
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
        # Strided pairs: obs at t=0,skip,2*skip,... paired with action at t + skip - 1
        offset = skip - 1
        indices = np.arange(0, n - offset, skip)
        diffs = actions_trimmed[indices + offset] - states_trimmed[indices]
        abs_diffs = np.abs(diffs)
        print(
            f"{skip:>6} | "
            f"{abs_diffs.mean():>12.6f} | "
            f"{diffs.std():>12.6f} | "
            f"{abs_diffs.max():>12.6f} | "
            f"{np.median(abs_diffs):>12.6f} | "
            f"n_pairs={len(indices)}"
        )


def compute_consecutive_action_distances(
    actions: np.ndarray,
    skips: list[int],
    num_joint_dims: int,
    thresholds: list[float],
    histogram_bins: int = 20,
) -> None:
    """Analyze L2 distances between consecutive output actions at each skip.

    For each skip, builds the sequence of "kept" actions (the ones that would
    end up in the transformed dataset) and reports the distribution of L2
    distances between successive entries over the first `num_joint_dims` dims.
    This is the same quantity used by transform_skip_dataset.py's
    --min-action-delta filter, so the percentiles / fraction-below tables
    here can be read directly as candidate thresholds.
    """
    # Take first action timestep if there's an action horizon
    if actions.ndim == 3:
        action_seq = actions[:, 0, :]
    else:
        action_seq = actions

    dim = min(num_joint_dims, action_seq.shape[-1])
    joint_actions = action_seq[..., :dim]

    print(f"\n{'=' * 78}")
    print(f"  Consecutive action L2 distances (joint dims [:{dim}])")
    print(f"  Use these to pick --min-action-delta in transform_skip_dataset.py")
    print(f"{'=' * 78}")

    percentile_qs = [50, 75, 90, 95, 99]
    header_pcts = " | ".join(f"p{q:>2}={'':>0}".rstrip() + " " * 0 for q in percentile_qs)
    print(
        f"{'Skip':>5} | {'N':>7} | {'mean':>10} | {'std':>10} | {'max':>10} | "
        + " | ".join(f"{'p'+str(q):>9}" for q in percentile_qs)
    )
    print("-" * 78)

    for skip in skips:
        offset = skip - 1
        if skip >= len(joint_actions):
            print(f"{skip:>5} | (skip >= N, skipped)")
            continue
        # Output kept actions: action[t + offset] for t = 0, skip, 2*skip, ...
        indices = np.arange(0, len(joint_actions) - offset, skip) + offset
        kept = joint_actions[indices]
        if len(kept) < 2:
            print(f"{skip:>5} | (not enough pairs)")
            continue
        diffs = kept[1:] - kept[:-1]
        dists = np.linalg.norm(diffs, axis=-1)

        pcts = [np.percentile(dists, q) for q in percentile_qs]
        print(
            f"{skip:>5} | {len(dists):>7d} | "
            f"{dists.mean():>10.6f} | {dists.std():>10.6f} | {dists.max():>10.6f} | "
            + " | ".join(f"{p:>9.6f}" for p in pcts)
        )

    # Fraction-below-threshold table: shows what % of consecutive frames
    # would be dropped at each candidate threshold.
    print(f"\n{'=' * 78}")
    print(f"  Fraction of consecutive pairs with distance < threshold")
    print(f"  (= fraction of frames that would be dropped as 'static')")
    print(f"{'=' * 78}")
    header = f"{'Skip':>5} | " + " | ".join(f"{t:>10.4f}" for t in thresholds)
    print(header)
    print("-" * len(header))
    for skip in skips:
        offset = skip - 1
        if skip >= len(joint_actions):
            continue
        indices = np.arange(0, len(joint_actions) - offset, skip) + offset
        kept = joint_actions[indices]
        if len(kept) < 2:
            continue
        diffs = kept[1:] - kept[:-1]
        dists = np.linalg.norm(diffs, axis=-1)
        fracs = [(dists < t).mean() for t in thresholds]
        print(
            f"{skip:>5} | "
            + " | ".join(f"{f * 100:>9.2f}%" for f in fracs)
        )

    # Histogram for the smallest skip — gives a visual feel for the bulk
    # of the distribution and where the "static" mode sits.
    if skips:
        skip = min(skips)
        offset = skip - 1
        if skip < len(joint_actions):
            indices = np.arange(0, len(joint_actions) - offset, skip) + offset
            kept = joint_actions[indices]
            if len(kept) >= 2:
                dists = np.linalg.norm(kept[1:] - kept[:-1], axis=-1)
                print(f"\n{'=' * 78}")
                print(f"  Histogram of consecutive distances at skip={skip} (N={len(dists)})")
                print(f"{'=' * 78}")
                hist, edges = np.histogram(dists, bins=histogram_bins)
                max_count = hist.max() if hist.max() > 0 else 1
                bar_width = 40
                cum = 0
                for i, count in enumerate(hist):
                    cum += count
                    bar = "#" * int(round(bar_width * count / max_count))
                    print(
                        f"  [{edges[i]:>9.6f}, {edges[i + 1]:>9.6f})  "
                        f"{count:>7d}  {cum / len(dists) * 100:>5.1f}%  {bar}"
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
        help=(
            "Comma-separated list of skip intervals to analyze. "
            "skip=1 means no change (original pairs). "
            "skip=5 means take every 5th frame, pairing obs[t] with action[t+4]. "
            "All values must be >= 1."
        ),
    )
    parser.add_argument(
        "--num_joint_dims",
        type=int,
        default=6,
        help="Number of joint dims used for the consecutive-action distance analysis (default: 6, matches transform_skip_dataset.py).",
    )
    parser.add_argument(
        "--thresholds",
        type=str,
        default="0.001,0.005,0.01,0.02,0.05,0.1,0.2",
        help=(
            "Comma-separated candidate thresholds to evaluate as "
            "--min-action-delta values. The script reports what fraction of "
            "consecutive frames would be dropped at each threshold per skip."
        ),
    )
    return parser.parse_args()


def main():
    init_logging()
    args = parse_args()

    skips = [int(s) for s in args.skips.split(",")]
    if any(s < 1 for s in skips):
        raise ValueError(f"All skip values must be >= 1, got {skips}")
    thresholds = [float(t) for t in args.thresholds.split(",")]
    if any(t < 0 for t in thresholds):
        raise ValueError(f"All thresholds must be >= 0, got {thresholds}")
    logging.info(
        f"Config: {args.config}, num_samples: {args.num_samples}, "
        f"skips: {skips}, num_joint_dims: {args.num_joint_dims}, thresholds: {thresholds}"
    )

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
    compute_consecutive_action_distances(
        actions, skips, args.num_joint_dims, thresholds
    )

    print(f"\n{'=' * 70}")
    print(f"  Analysis complete. {len(states)} samples analyzed.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
