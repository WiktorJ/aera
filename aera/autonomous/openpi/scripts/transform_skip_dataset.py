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

from aera.autonomous.obs_augmentation import (
    apply_state_noise,
    augment_image,
    sample_camera_profile,
    sample_state_noise_profile,
)
from aera.autonomous.openpi.dataset_transforms import compute_smoothed_arrays


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
        help=(
            "Subsample interval — a LEARNING hyperparameter, not a control-rate "
            "setting. skip=1 keeps every recorded frame; skip=N takes every Nth "
            "frame and pairs obs[t] with action[t+N]. Pick it so the per-step "
            "delta carries signal: recording is per mj-step (0.002 s), so tiny "
            "skips give near-zero deltas and the policy can learn to sit still. "
            "How the predictions are applied at deploy is a separate concern — "
            "set the env/driver's n_substeps to match the skip used here so one "
            "action = one decision interval. Must be >= 1."
        ),
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
        "--min-action-delta",
        type=float,
        default=None,
        help=(
            "If set, drop frames whose action is closer than this L2 distance "
            "(over joint dims) to the previously written frame's action within "
            "the same episode. Useful for filtering out static/idle frames "
            "where the robot pauses between sub-tasks. The gripper is guarded "
            "separately (see --gripper-eps) so grasp/release frames survive."
        ),
    )
    parser.add_argument(
        "--gripper-eps",
        type=float,
        default=0.001,
        help=(
            "Gripper guard for --min-action-delta: a frame whose gripper dims "
            "moved more than this (max abs change vs the last written frame) is "
            "always kept, even when the arm joints are static. Prevents the idle "
            "filter from deleting grasp/release transitions where only the "
            "gripper moves. Set relative to gripper travel and above the hold "
            "jitter floor (default 0.001). Ignored if there are no gripper dims."
        ),
    )
    parser.add_argument(
        "--exclude-prompts",
        type=str,
        default=None,
        help=(
            "Comma-separated list of granular_prompt values to exclude from the output dataset. "
            "Frames matching any of these prompts are dropped entirely. "
            "E.g. --exclude-prompts 'go home,reset arm'"
        ),
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=0,
        help=(
            "Savitzky-Golay window length (odd integer). 0 disables smoothing. "
            "Applied per-episode on raw source actions before skipping/filtering."
        ),
    )
    parser.add_argument(
        "--smooth-polyorder",
        type=int,
        default=3,
        help="Savitzky-Golay polynomial order (must be < smooth-window).",
    )
    parser.add_argument(
        "--smooth-state",
        action="store_true",
        default=False,
        help=(
            "Also smooth the state arrays. Off by default to keep state[t] aligned "
            "with image[t] from the source recording."
        ),
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="If set, only process the first N episodes. Useful for quick testing.",
    )
    parser.add_argument(
        "--image-aug",
        action="store_true",
        default=False,
        help=(
            "Bake per-episode camera sensor-realism into the images (noise, "
            "blur, motion blur, vignette, white-balance, gamma, jpeg, grayscale, "
            "frozen frames). Train-only by construction; complements openpi's "
            "resampled crop/rotate/color-jitter. See aera.autonomous.obs_augmentation."
        ),
    )
    parser.add_argument(
        "--state-aug",
        action="store_true",
        default=False,
        help=(
            "Add per-episode bias + per-frame Gaussian jitter to the STATE input "
            "only (never the action target). With --delta-actions the same noised "
            "state is used as the delta reference, so the noise cancels at "
            "inference and the policy learns to treat state as a noisy reference."
        ),
    )
    parser.add_argument(
        "--obs-aug-strength",
        type=float,
        default=1.0,
        help="Scale [0,1] for how far obs augmentation is pushed from neutral.",
    )
    parser.add_argument(
        "--obs-aug-seed",
        type=int,
        default=0,
        help="Seed for the obs-augmentation RNG (reproducible augmented datasets).",
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


def _build_output_repo_id(
    source_repo_id: str, skip: int, delta: bool, suffix: str | None
) -> str:
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
    exclude_prompts: set[str] | None = None,
    min_action_delta: float | None = None,
    gripper_eps: float = 0.001,
    smoothed_actions: np.ndarray | None = None,
    smoothed_state: np.ndarray | None = None,
    excluded_episodes: set[int] | None = None,
    max_episodes: int | None = None,
    image_aug: bool = False,
    state_aug: bool = False,
    aug_strength: float = 1.0,
    aug_seed: int = 0,
) -> LeRobotDataset:
    """Transform the source dataset by pairing obs[t] with action[t+skip].

    Args:
        source_dataset: The source LeRobot dataset.
        output_repo_id: The repo ID for the output dataset.
        skip: Number of frames to skip for action pairing.
        delta_actions: Whether to convert actions to delta (action - state).
        num_joint_dims: Number of joint dims for delta conversion.
        exclude_prompts: Set of granular_prompt values to exclude. Frames matching
            any of these are dropped entirely.

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
    frames_skipped_prompt = 0
    frames_skipped_static = 0
    frames_skipped_short_episode = 0
    episodes_written = 0
    last_action_for_episode: np.ndarray | None = None
    episodes_seen: set[int] = set()
    excluded_episodes = excluded_episodes or set()

    # Observation augmentation (train-only sensor realism). Profiles are sampled
    # once per episode so a clip is internally consistent; per-frame calls add
    # the stochastic noise / motion blur / frozen frames.
    aug_rng = np.random.default_rng(aug_seed)
    aug_episode: int | None = None
    cam_profile = None
    state_profile = None
    prev_aug_images: dict = {}

    offset = skip - 1
    for t in range(0, total_samples, skip):
        if t % 1000 == 0:
            logging.info(
                f"Processing frame {t}/{total_samples} "
                f"(written: {frames_written}, skipped_boundary: {frames_skipped_boundary}, "
                f"skipped_prompt: {frames_skipped_prompt}, "
                f"skipped_static: {frames_skipped_static})"
            )

        t_future = t + offset

        # Check bounds
        if t_future >= total_samples:
            break

        sample_t = source_dataset[t]
        sample_future = source_dataset[t_future]

        # Filter out frames with excluded granular prompts
        if exclude_prompts:
            prompt_t = sample_t.get("granural_prompt", None)
            prompt_future = sample_future.get("granural_prompt", None)
            if (prompt_t is not None and prompt_t in exclude_prompts) or (
                prompt_future is not None and prompt_future in exclude_prompts
            ):
                frames_skipped_prompt += 1
                continue

        episode_t = _get_episode_index(sample_t)
        episode_future = _get_episode_index(sample_future)

        # Drop frames belonging to episodes shorter than the smoothing window.
        if episode_t in excluded_episodes or episode_future in excluded_episodes:
            frames_skipped_short_episode += 1
            continue

        # Stop once we've seen the requested number of episodes.
        if max_episodes is not None and episode_t not in episodes_seen and len(episodes_seen) >= max_episodes:
            logging.info(
                f"Reached --max-episodes={max_episodes}; stopping at frame {t}."
            )
            break
        episodes_seen.add(episode_t)

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
            logging.info(
                f"Saved episode {current_episode} (total episodes: {episodes_written})"
            )
            last_action_for_episode = None

        current_episode = episode_t

        # Resample per-episode augmentation profiles when entering a new episode.
        if (image_aug or state_aug) and current_episode != aug_episode:
            aug_episode = current_episode
            cam_profile = (
                sample_camera_profile(aug_rng, strength=aug_strength)
                if image_aug
                else None
            )
            state_profile = None  # sampled lazily once we know the state dim
            prev_aug_images = {}

        # Build the output frame: obs from t, actions from t+skip
        frame = {}

        # Copy images from time t
        for key in image_keys:
            if key in sample_t:
                frame[key] = _parse_image_from_sample(sample_t[key])

        # Image sensor-realism augmentation (per-episode profile + per-frame
        # noise). A "frozen" frame reuses the previous augmented image (stale /
        # duplicated frame), so it must apply to all image keys together.
        if image_aug and cam_profile is not None:
            frozen = (
                bool(prev_aug_images)
                and aug_rng.random() < cam_profile.frame_freeze_prob
            )
            for key in image_keys:
                if key not in frame:
                    continue
                if frozen:
                    frame[key] = prev_aug_images[key]
                else:
                    frame[key] = augment_image(frame[key], cam_profile, aug_rng)
                prev_aug_images[key] = frame[key]

        # Copy state from time t (use smoothed if provided)
        state_t = None
        if smoothed_state is not None:
            state_t = smoothed_state[t].astype(np.float32)
            frame["state"] = state_t
        elif "state" in sample_t:
            state_t = _to_numpy(sample_t["state"]).astype(np.float32)
            frame["state"] = state_t

        # State (proprioception) noise on the INPUT only. Applied to state_t
        # before it is both stored and used as the delta-action reference, so
        # delta = action_future - noisy_state stays self-consistent.
        if state_aug and state_t is not None:
            if state_profile is None:
                state_profile = sample_state_noise_profile(
                    state_dim=int(state_t.shape[0]),
                    rng=aug_rng,
                    strength=aug_strength,
                )
            state_t = apply_state_noise(state_t, state_profile, aug_rng)
            frame["state"] = state_t

        # Get actions from time t+skip (use smoothed if provided)
        action_future = None
        if smoothed_actions is not None:
            action_future = smoothed_actions[t_future].astype(np.float32)
        elif "actions" in sample_future:
            action_future = _to_numpy(sample_future["actions"]).astype(np.float32)

        if action_future is not None:

            # Drop frames where the action barely moved relative to the last
            # written frame in this episode (filters static/idle pauses). The
            # gripper is guarded separately: a frame whose gripper dims moved
            # by more than gripper_eps is always kept, even when the arm joints
            # are static, so grasp/release transitions (joints held, only the
            # gripper moving) are never filtered out.
            if min_action_delta is not None and last_action_for_episode is not None:
                joint_now = action_future[:num_joint_dims]
                joint_prev = last_action_for_episode[:num_joint_dims]
                dist = float(np.linalg.norm(joint_now - joint_prev))
                gripper_now = action_future[num_joint_dims:]
                gripper_prev = last_action_for_episode[num_joint_dims:]
                gripper_moved = (
                    gripper_now.size > 0
                    and float(np.abs(gripper_now - gripper_prev).max()) > gripper_eps
                )
                if dist < min_action_delta and not gripper_moved:
                    frames_skipped_static += 1
                    continue

            if delta_actions and "state" in sample_t:
                # Delta = action[t+skip] - state[t] for joint dims
                # Keep gripper dims absolute
                delta = action_future.copy()
                delta[:num_joint_dims] = (
                    action_future[:num_joint_dims] - state_t[:num_joint_dims]
                )
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
        if action_future is not None:
            last_action_for_episode = action_future

    # Save the last episode
    if frames_written > 0 and current_episode is not None:
        output_dataset.save_episode()
        episodes_written += 1
        logging.info(
            f"Saved final episode {current_episode} (total episodes: {episodes_written})"
        )

    logging.info(
        f"\nTransformation complete:\n"
        f"  Source frames:          {total_samples}\n"
        f"  Output frames:          {frames_written}\n"
        f"  Skipped (boundary):     {frames_skipped_boundary}\n"
        f"  Skipped (prompt filter):{frames_skipped_prompt}\n"
        f"  Skipped (static):       {frames_skipped_static}\n"
        f"  Skipped (short ep):     {frames_skipped_short_episode}\n"
        f"  Episodes written:       {episodes_written}\n"
        f"  Skip interval:          {skip}\n"
        f"  Delta actions:          {delta_actions}\n"
        f"  Excluded prompts:       {exclude_prompts or 'none'}\n"
        f"  Min action delta:       {min_action_delta if min_action_delta is not None else 'disabled'}\n"
        f"  Gripper eps:            {gripper_eps if min_action_delta is not None else 'n/a'}"
    )

    return output_dataset


def main():
    init_logging()
    args = parse_args()

    if args.skip < 1:
        raise ValueError(f"--skip must be >= 1, got {args.skip}")

    # skip is a data/learning choice: it sets how far apart paired frames are so
    # the per-step delta carries signal. The deploy side reproduces this rate by
    # setting the env/driver's n_substeps equal to the skip used, so record the
    # skip with the dataset (e.g. in its repo name). See CONTROL_RATE_SPEC.md.
    logging.info(
        "Building dataset with skip=%d. Deploy/eval must set n_substeps=%d to "
        "apply one action per decision interval.",
        args.skip,
        args.skip,
    )

    # --state-aug + --smooth-state compose fine: smoothing is precomputed on the
    # raw signal, then the DR noise is layered on top (and still cancels in the
    # delta reference), so the injected jitter is not low-passed away.

    exclude_prompts = None
    if args.exclude_prompts:
        exclude_prompts = {p.strip() for p in args.exclude_prompts.split(",")}

    logging.info(
        f"Config: repo_id={args.repo_id}, skip={args.skip}, "
        f"delta_actions={args.delta_actions}, num_joint_dims={args.num_joint_dims}, "
        f"exclude_prompts={exclude_prompts}, min_action_delta={args.min_action_delta}, "
        f"gripper_eps={args.gripper_eps}, "
        f"smooth_window={args.smooth_window}, smooth_polyorder={args.smooth_polyorder}, "
        f"smooth_state={args.smooth_state}, max_episodes={args.max_episodes}, "
        f"image_aug={args.image_aug}, state_aug={args.state_aug}, "
        f"obs_aug_strength={args.obs_aug_strength}, obs_aug_seed={args.obs_aug_seed}"
    )

    output_repo_id = _build_output_repo_id(
        args.repo_id, args.skip, args.delta_actions, args.output_repo_suffix
    )
    logging.info(f"Output repo ID: {output_repo_id}")

    # Check if the output dataset already exists locally in the HuggingFace cache
    hf_cache_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / output_repo_id
    if hf_cache_dir.exists():
        logging.info(
            f"Output dataset already exists at {hf_cache_dir}. "
            "Skipping transformation and loading existing dataset."
        )
        output_dataset = LeRobotDataset(output_repo_id)
        logging.info(f"Loaded existing dataset: {len(output_dataset)} frames")
    else:
        # Load source dataset
        logging.info(f"Loading source dataset: {args.repo_id}")
        source_dataset = LeRobotDataset(args.repo_id)
        logging.info(f"Source dataset loaded: {len(source_dataset)} frames")

        # Log a sample to help debug
        sample = source_dataset[0]
        logging.info(f"Sample keys: {list(sample.keys())}")
        for key, val in sample.items():
            if isinstance(val, (torch.Tensor, np.ndarray)):
                logging.info(
                    f"  {key}: shape={getattr(val, 'shape', 'N/A')}, dtype={getattr(val, 'dtype', 'N/A')}"
                )
            else:
                logging.info(f"  {key}: {type(val).__name__} = {val}")

        # Pre-compute smoothed arrays per-episode if requested
        smoothed_actions = None
        smoothed_state = None
        excluded_episodes: set[int] = set()
        if args.smooth_window > 0:
            smoothed_actions, smoothed_state, excluded_episodes = compute_smoothed_arrays(
                source_dataset,
                window=args.smooth_window,
                polyorder=args.smooth_polyorder,
                smooth_state=args.smooth_state,
            )

        # Transform
        output_dataset = transform_dataset(
            source_dataset,
            output_repo_id=output_repo_id,
            skip=args.skip,
            delta_actions=args.delta_actions,
            num_joint_dims=args.num_joint_dims,
            exclude_prompts=exclude_prompts,
            min_action_delta=args.min_action_delta,
            gripper_eps=args.gripper_eps,
            smoothed_actions=smoothed_actions,
            smoothed_state=smoothed_state,
            excluded_episodes=excluded_episodes,
            max_episodes=args.max_episodes,
            image_aug=args.image_aug,
            state_aug=args.state_aug,
            aug_strength=args.obs_aug_strength,
            aug_seed=args.obs_aug_seed,
        )
        output_dataset.finalize()

    # Optionally push to hub
    if args.push_to_hub:
        logging.info("Pushing to HuggingFace Hub...")
        output_dataset.push_to_hub(
            tags=["aera", "ar4_mk3", f"skip{args.skip}"]
            + (["delta_actions"] if args.delta_actions else []),
            private=args.private,
            push_videos=True,
            license="apache-2.0",
            upload_large_folder=True,
        )
        logging.info(f"Dataset pushed to hub: {output_repo_id}")
    else:
        logging.info(f"Dataset saved locally. Use --push-to-hub to upload.")

    logging.info("Done.")


if __name__ == "__main__":
    main()
