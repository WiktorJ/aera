"""Create a small subset of a LeRobot dataset for fast debug runs.

Selects a fraction of the source dataset's episodes (evenly spaced across the
dataset, so the subset covers the same recording span / prompt variety as the
full data), copies their frames verbatim into a new LeRobot dataset, and
optionally pushes it to HuggingFace Hub.

Only the selected episodes' data chunks are downloaded from the hub, so this is
cheap even for large source datasets.

Note: openpi norm-stats assets are NOT copied. Debug configs should keep their
AssetsConfig pointing at the FULL dataset repo (the stats files are tiny) and
only switch repo_id to the subset.

Usage:
    python -m aera.autonomous.openpi.scripts.create_subset_dataset \
        --repo-id Purple69/aera_semi_pnp_dr_16_06_2026_skip3_delta_no_go_home_no_static_smoothed_v2 \
        --push-to-hub

    python -m aera.autonomous.openpi.scripts.create_subset_dataset \
        --repo-id Purple69/aera_semi_pnp_dr_16_06_2026_skip3_delta_no_go_home_no_static_smoothed_v2 \
        --fraction 0.01 \
        --output-repo-suffix debug
"""

import argparse
import logging
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from aera.autonomous.openpi.scripts.merge_datasets import (
    _build_frame,
    _get_episode_index,
    init_logging,
)

_META_KEYS = {"episode_index", "frame_index", "timestamp", "index", "task_index"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a small episode subset of a LeRobot dataset."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID of the source LeRobot dataset.",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.001,
        help=(
            "Fraction of episodes to keep (default: 0.001 = 0.1%%). At least one "
            "episode is always kept. Ignored if --num-episodes is set."
        ),
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=None,
        help="Exact number of episodes to keep. Overrides --fraction.",
    )
    parser.add_argument(
        "--output-repo-suffix",
        type=str,
        default=None,
        help="Suffix for the output repo ID. Defaults to 'subset{N}ep'.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push the subset dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the uploaded dataset private.",
    )
    return parser.parse_args()


def _select_episodes(total_episodes: int, fraction: float, num_episodes: int | None) -> list[int]:
    """Pick evenly spaced episode indices covering the whole dataset."""
    if num_episodes is not None:
        n = num_episodes
    else:
        n = round(fraction * total_episodes)
    n = max(1, min(n, total_episodes))
    return sorted(np.unique(np.linspace(0, total_episodes - 1, n).round().astype(int)).tolist())


def _build_output_features(source_meta: LeRobotDatasetMetadata) -> dict:
    """Copy the source feature schema, storing video features as images."""
    features = {}
    for key, feat in source_meta.features.items():
        if key in _META_KEYS:
            continue
        features[key] = {
            "dtype": "image" if feat.get("dtype") == "video" else feat["dtype"],
            "shape": tuple(feat["shape"]),
            "names": feat["names"],
        }
    return features


def copy_episodes(source: LeRobotDataset, output_dataset: LeRobotDataset) -> tuple[int, int]:
    """Copy all frames of `source` (already episode-filtered) into `output_dataset`.

    Returns (episodes_written, frames_written).
    """
    features = output_dataset.meta.features
    total = len(source)
    current_episode: int | None = None
    episodes_written = 0
    frames_written = 0

    for i in range(total):
        sample = source[i]
        ep = _get_episode_index(sample)

        if current_episode is not None and ep != current_episode:
            output_dataset.save_episode()
            episodes_written += 1
            logging.info(
                f"Saved episode (source ep={current_episode}); "
                f"{episodes_written} episodes, {frames_written} frames so far"
            )

        current_episode = ep
        output_dataset.add_frame(_build_frame(sample, features))
        frames_written += 1

        if i % 1000 == 0:
            logging.info(f"Processed {i}/{total} source frames")

    if current_episode is not None:
        output_dataset.save_episode()
        episodes_written += 1
        logging.info(f"Saved final episode (source ep={current_episode})")

    return episodes_written, frames_written


def main():
    init_logging()
    args = parse_args()

    if not 0 < args.fraction <= 1:
        raise ValueError(f"--fraction must be in (0, 1], got {args.fraction}")
    if args.num_episodes is not None and args.num_episodes < 1:
        raise ValueError(f"--num-episodes must be >= 1, got {args.num_episodes}")

    source_meta = LeRobotDatasetMetadata(args.repo_id)
    selected = _select_episodes(
        source_meta.total_episodes, args.fraction, args.num_episodes
    )
    logging.info(
        f"Selected {len(selected)}/{source_meta.total_episodes} episodes: {selected}"
    )

    suffix = args.output_repo_suffix or f"subset{len(selected)}ep"
    org, name = args.repo_id.split("/", 1)
    output_repo_id = f"{org}/{name}_{suffix}"
    logging.info(f"Output repo ID: {output_repo_id}")

    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / output_repo_id
    if hf_cache_dir.exists():
        logging.info(
            f"Output dataset already exists at {hf_cache_dir}. "
            "Skipping subsetting and loading existing dataset."
        )
        output_dataset = LeRobotDataset(output_repo_id)
        logging.info(f"Loaded existing dataset: {len(output_dataset)} frames")
    else:
        logging.info(f"Loading source dataset: {args.repo_id} (selected episodes only)")
        source = LeRobotDataset(args.repo_id, episodes=selected)
        logging.info(f"Source subset loaded: {len(source)} frames")

        output_dataset = LeRobotDataset.create(
            repo_id=output_repo_id,
            robot_type=source_meta.robot_type,
            features=_build_output_features(source_meta),
            fps=source_meta.fps,
            image_writer_processes=5,
            image_writer_threads=10,
        )

        episodes_written, frames_written = copy_episodes(source, output_dataset)
        output_dataset.finalize()
        logging.info(
            f"Subset complete: {episodes_written} episodes, {frames_written} frames."
        )

    if args.push_to_hub:
        logging.info("Pushing to HuggingFace Hub...")
        output_dataset.push_to_hub(
            tags=["aera", "ar4_mk3", "subset"],
            private=args.private,
            push_videos=True,
            license="apache-2.0",
            upload_large_folder=True,
        )
        logging.info(f"Dataset pushed to hub: {output_repo_id}")
    else:
        logging.info("Dataset saved locally. Use --push-to-hub to upload.")

    logging.info("Done.")


if __name__ == "__main__":
    main()
