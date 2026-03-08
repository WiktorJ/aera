"""Transform a LeRobot dataset by pairing observation[t] with action[t + skip].

This script loads an existing LeRobot dataset, re-pairs observations with
future actions at a configurable skip interval, optionally converts actions
to delta actions (action - current_state), and saves/uploads the result as
a new LeRobot dataset.

Episode boundaries are respected: pairs that would cross episodes are dropped.

Usage:
    python -m aera.autonomous.openpi.scripts.transform_skip_dataset \
        --repo-id Purple69/aera_semi_pnp_dr_08_01_2026 \
        --skip 5 \
        --delta-actions \
        --push-to-hub

    python -m aera.autonomous.openpi.scripts.transform_skip_dataset \
        --repo-id Purple69/aera_semi_pnp_dr_08_01_2026 \
        --skip 10 \
        --output-repo-suffix skip10_delta \
        --delta-actions
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transform a LeRobot dataset by pairing obs[t] with action[t+skip]."
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HuggingFace repo ID of the source LeRobot dataset (e.g. Purple69/aera_semi_pnp_dr_08_01_2026).",
    )
    parser.add_argument(
        "--skip",
        type=int,
        required=True,
        help="Number of frames to skip when pairing obs[t] with action[t+skip].",
    )
    parser.add_argument(
        "--delta-actions",
        action="store_true",
        default=False,
        help="Convert actions to delta actions (action[t+skip] - state[t]) for joint dims, keeping gripper absolute.",
    )
    parser.add_argument(
        "--num-joint-dims",
        type=int,
        default=6,
        help="Number of joint dimensions to apply delta conversion to (default: 6, remaining dims like gripper stay absolute).",
    )
    parser.add_argument(
        "--output-repo-suffix",
        type=str,
        default=None,
        help="Suffix for the output repo ID. Defaults to 'skip{N}' or 'skip{N}_delta'.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        default=False,
        help="Push the transformed dataset to HuggingFace Hub.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=False,
        help="Make the uploaded dataset private.",
    )
    return parser.parse_args()


def _to_numpy(value):
    """Convert a value to a numpy array, handling torch tensors and PIL images."""
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _build_output_repo_id(source_repo_id: str, skip: int, delta: bool, suffix: str | None) -> str:
    """Build the output repo ID from the source repo ID and options."""
    if suffix is not None:
        tag = suffix
    else:
        tag = f"skip{skip}_delta" if delta else f"skip{skip}"

    org, name = source_repo_id.split("/", 1)
    return f"{org}/{name}_{tag}"


def _extract_image_features(source_dataset: LeRobotDataset) -> dict:
    """Extract image feature definitions from the source dataset."""
    image_features = {}
    for key, feat in source_dataset.meta.features.items():
        if feat.get("dtype") in ("image", "video"):
            image_features[key] = {
                "dtype": "image",
                "shape": tuple(feat["shape"]),
                "names": feat["names"],
            }
    return image_features


def _extract_numeric_features(source_dataset: LeRobotDataset) -> dict:
    """Extract numeric (state/action) feature definitions from the source dataset."""
    numeric_features = {}
    for key, feat in source_dataset.meta.features.items():
        if feat.get("dtype") in ("float32", "float64", "int32", "int64"):
            numeric_features[key] = {
                "dtype": feat["dtype"],
                "shape": tuple(feat["shape"]),
                "names": feat["names"],
            }
    return numeric_features


def _extract_string_features(source_dataset: LeRobotDataset) -> dict:
    """Extract string feature definitions from the source dataset."""
    string_features = {}
    for key, feat in source_dataset.meta.features.items():
        if feat.get("dtype") == "string":
            string_features[key] = {
                "dtype": "string",
                "shape": tuple(feat["shape"]),
                "names": feat["names"],
            }
    return string_features


def _get_episode_index(sample: dict) -> int:
    """Extract the episode index from a dataset sample."""
    val = sample["episode_index"]
    if isinstance(val, torch.Tensor):
        return val.item()
    return int(val)


def _parse_image_from_sample(value) -> np.ndarray:
    """Convert an image value from the dataset to uint8 HWC numpy array."""
    arr = _to_numpy(value)
    # LeRobot may store images as float32 CHW
    if np.issubdtype(arr.dtype, np.floating):
        arr = (255 * arr).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def transform_dataset(
    source_dataset: LeRobotDataset,
    output_repo_id: str,
    skip: int,
    delta_actions: bool,
    num_joint_dims: int,
) -> LeRobotDataset:
    """Transform the source dataset by pairing obs[t] with action[t+skip].

    Args:
        source_dataset: The source LeRobot dataset.
        output_repo_id: The repo ID for the output dataset.
        skip: Number of frames to skip for action pairing.
        delta_actions: Whether to convert actions to delta (action - state).
        num_joint_dims: Number of joint dims for delta conversion.

    Returns:
        The new LeRobotDataset with transformed data.
    """
    # Build feature definitions from source
    image_features = _extract_image_features(source_dataset)
    numeric_features = _extract_numeric_features(source_dataset)
    string_features = _extract_string_features(source_dataset)

    features = {**image_features, **numeric_features, **string_features}
    logging.info(f"Output features: {list(features.keys())}")

    fps = source_dataset.meta.fps
    logging.info(f"Source FPS: {fps}")

    output_dataset = LeRobotDataset.create(
        repo_id=output_repo_id,
        robot_type="AR4_MK3",
        features=features,
        fps=fps,
        image_writer_processes=5,
        image_writer_threads=10,
    )

    total_samples = len(source_dataset)
    logging.info(f"Source dataset size: {total_samples}")

    # Identify image and non-image keys for frame construction
    image_keys = set(image_features.keys())
    # Keys we handle specially or skip
    meta_keys = {"episode_index", "frame_index", "timestamp", "index", "task_index"}

    current_episode = None
    frames_written = 0
    frames_skipped_boundary = 0
    episodes_written = 0

    step = max(skip, 1)
    for t in range(0, total_samples, step):
        if t % 1000 == 0:
            logging.info(
                f"Processing frame {t}/{total_samples} "
                f"(written: {frames_written}, skipped: {frames_skipped_boundary})"
            )

        t_future = t + skip

        # Check bounds
        if t_future >= total_samples:
            break

        sample_t = source_dataset[t]
        sample_future = source_dataset[t_future]

        episode_t = _get_episode_index(sample_t)
        episode_future = _get_episode_index(sample_future)

        # Skip pairs that cross episode boundaries
        if episode_t != episode_future:
            # If we were building an episode, save it before moving on
            if current_episode is not None and current_episode == episode_t:
                # We've reached the end of this episode's valid pairs
                pass
            frames_skipped_boundary += 1
            continue

        # Handle episode transitions — save previous episode when we enter a new one
        if current_episode is not None and episode_t != current_episode:
            output_dataset.save_episode()
            episodes_written += 1
            logging.info(f"Saved episode {current_episode} (total episodes: {episodes_written})")

        current_episode = episode_t

        # Build the output frame: obs from t, actions from t+skip
        frame = {}

        # Copy images from time t
        for key in image_keys:
            if key in sample_t:
                frame[key] = _parse_image_from_sample(sample_t[key])

        # Copy state from time t
        if "state" in sample_t:
            state_t = _to_numpy(sample_t["state"]).astype(np.float32)
            frame["state"] = state_t

        # Get actions from time t+skip
        if "actions" in sample_future:
            action_future = _to_numpy(sample_future["actions"]).astype(np.float32)

            if delta_actions and "state" in sample_t:
                # Delta = action[t+skip] - state[t] for joint dims
                # Keep gripper dims absolute
                delta = action_future.copy()
                delta[:num_joint_dims] = action_future[:num_joint_dims] - state_t[:num_joint_dims]
                frame["actions"] = delta
            else:
                frame["actions"] = action_future

        # Copy string/task fields from time t
        for key in string_features:
            if key in sample_t:
                frame[key] = sample_t[key]

        # Copy task from time t (used by LeRobot for prompt)
        if "task" in sample_t:
            frame["task"] = sample_t["task"]

        # Copy any remaining numeric features from time t (except actions, state, meta)
        for key in numeric_features:
            if key not in frame and key not in meta_keys and key in sample_t:
                frame[key] = _to_numpy(sample_t[key]).astype(np.float32)

        output_dataset.add_frame(frame)
        frames_written += 1

    # Save the last episode
    if frames_written > 0 and current_episode is not None:
        output_dataset.save_episode()
        episodes_written += 1
        logging.info(f"Saved final episode {current_episode} (total episodes: {episodes_written})")

    logging.info(
        f"\nTransformation complete:\n"
        f"  Source frames:          {total_samples}\n"
        f"  Output frames:          {frames_written}\n"
        f"  Skipped (boundary):     {frames_skipped_boundary}\n"
        f"  Skipped (tail):         {total_samples - frames_written - frames_skipped_boundary}\n"
        f"  Episodes written:       {episodes_written}\n"
        f"  Skip interval:          {skip}\n"
        f"  Delta actions:          {delta_actions}"
    )

    return output_dataset


def main():
    init_logging()
    args = parse_args()

    logging.info(
        f"Config: repo_id={args.repo_id}, skip={args.skip}, "
        f"delta_actions={args.delta_actions}, num_joint_dims={args.num_joint_dims}"
    )

    output_repo_id = _build_output_repo_id(
        args.repo_id, args.skip, args.delta_actions, args.output_repo_suffix
    )
    logging.info(f"Output repo ID: {output_repo_id}")

    # Load source dataset
    logging.info(f"Loading source dataset: {args.repo_id}")
    source_dataset = LeRobotDataset(args.repo_id)
    logging.info(f"Source dataset loaded: {len(source_dataset)} frames")

    # Log a sample to help debug
    sample = source_dataset[0]
    logging.info(f"Sample keys: {list(sample.keys())}")
    for key, val in sample.items():
        if isinstance(val, (torch.Tensor, np.ndarray)):
            logging.info(f"  {key}: shape={getattr(val, 'shape', 'N/A')}, dtype={getattr(val, 'dtype', 'N/A')}")
        else:
            logging.info(f"  {key}: {type(val).__name__} = {val}")

    # Transform
    output_dataset = transform_dataset(
        source_dataset,
        output_repo_id=output_repo_id,
        skip=args.skip,
        delta_actions=args.delta_actions,
        num_joint_dims=args.num_joint_dims,
    )

    # Optionally push to hub
    if args.push_to_hub:
        logging.info("Finalizing and pushing to HuggingFace Hub...")
        output_dataset.finalize()
        output_dataset.push_to_hub(
            tags=["aera", "ar4_mk3", f"skip{args.skip}"]
            + (["delta_actions"] if args.delta_actions else []),
            private=args.private,
            push_videos=True,
            license="apache-2.0",
        )
        logging.info(f"Dataset pushed to hub: {output_repo_id}")
    else:
        output_dataset.finalize()
        logging.info(f"Dataset saved locally. Use --push-to-hub to upload.")

    logging.info("Done.")


if __name__ == "__main__":
    main()
