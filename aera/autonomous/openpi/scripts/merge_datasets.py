"""Append a source LeRobot dataset onto a target LeRobot dataset on HF.

The target dataset is loaded locally and appended to in place. Episodes from the
source are re-indexed to continue after the target's existing episodes. After the
append, the target is pushed back to its HF repo.

Safety:
  * Pre-flight: fps and feature schemas must match exactly. Refuse to proceed otherwise.
  * The target's current HF commit SHA is written to a file BEFORE any local mutation,
    so the remote can be reverted to that SHA if the merge produces a bad result.

After merging, re-run `compute_norm_stats.py` against the merged dataset.

Usage:
    python -m aera.autonomous.openpi.scripts.merge_datasets \
        --target-repo-id Purple69/aera_big \
        --source-repo-id Purple69/aera_small \
        --sha-output-file ./pre_merge_sha.txt \
        --push-to-hub
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def init_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append a source LeRobot dataset onto a target LeRobot dataset."
    )
    parser.add_argument("--target-repo-id", type=str, required=True)
    parser.add_argument("--source-repo-id", type=str, required=True)
    parser.add_argument(
        "--sha-output-file",
        type=Path,
        required=True,
        help="Path to write the target's pre-merge HF commit SHA. Used for rollback.",
    )
    parser.add_argument("--push-to-hub", action="store_true", default=False)
    parser.add_argument("--private", action="store_true", default=False)
    return parser.parse_args()


def _features_compatible(target: LeRobotDataset, source: LeRobotDataset) -> list[str]:
    """Return a list of mismatch descriptions; empty list means compatible."""
    problems: list[str] = []

    if target.meta.fps != source.meta.fps:
        problems.append(
            f"fps mismatch: target={target.meta.fps} source={source.meta.fps}"
        )

    t_feats = target.meta.features
    s_feats = source.meta.features

    # Meta keys are added automatically by add_frame/save_episode — skip them.
    meta_keys = {"episode_index", "frame_index", "timestamp", "index", "task_index"}
    t_keys = set(t_feats) - meta_keys
    s_keys = set(s_feats) - meta_keys

    if t_keys != s_keys:
        only_t = t_keys - s_keys
        only_s = s_keys - t_keys
        if only_t:
            problems.append(f"keys only in target: {sorted(only_t)}")
        if only_s:
            problems.append(f"keys only in source: {sorted(only_s)}")

    for key in t_keys & s_keys:
        t = t_feats[key]
        s = s_feats[key]
        if t.get("dtype") != s.get("dtype"):
            problems.append(
                f"{key}: dtype mismatch target={t.get('dtype')} source={s.get('dtype')}"
            )
        if tuple(t.get("shape", ())) != tuple(s.get("shape", ())):
            problems.append(
                f"{key}: shape mismatch target={t.get('shape')} source={s.get('shape')}"
            )

    return problems


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _parse_image_from_sample(value) -> np.ndarray:
    arr = _to_numpy(value)
    if np.issubdtype(arr.dtype, np.floating):
        arr = (255 * arr).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    return arr


def _get_episode_index(sample: dict) -> int:
    val = sample["episode_index"]
    if isinstance(val, torch.Tensor):
        return val.item()
    return int(val)


def _build_frame(sample: dict, features: dict) -> dict:
    """Build an add_frame-compatible dict from a source dataset sample."""
    meta_keys = {"episode_index", "frame_index", "timestamp", "index", "task_index"}
    frame: dict = {}
    for key, feat in features.items():
        if key in meta_keys or key not in sample:
            continue
        dtype = feat.get("dtype")
        if dtype in ("image", "video"):
            frame[key] = _parse_image_from_sample(sample[key])
        elif dtype in ("float32", "float64", "int32", "int64"):
            frame[key] = _to_numpy(sample[key]).astype(np.dtype(dtype))
        else:
            frame[key] = sample[key]
    if "task" in sample:
        frame["task"] = sample["task"]
    return frame


def append_source_into_target(
    target: LeRobotDataset, source: LeRobotDataset
) -> tuple[int, int]:
    """Iterate source frames, group by episode, and append to target.

    Returns (episodes_added, frames_added).
    """
    features = target.meta.features
    total = len(source)
    current_episode: int | None = None
    episodes_added = 0
    frames_added = 0

    for i in range(total):
        sample = source[i]
        ep = _get_episode_index(sample)

        if current_episode is not None and ep != current_episode:
            target.save_episode()
            episodes_added += 1
            logging.info(
                f"Saved appended episode (source ep={current_episode}); "
                f"total appended so far: {episodes_added}"
            )

        current_episode = ep
        target.add_frame(_build_frame(sample, features))
        frames_added += 1

        if i % 1000 == 0:
            logging.info(f"Processed {i}/{total} source frames")

    if current_episode is not None:
        target.save_episode()
        episodes_added += 1
        logging.info(f"Saved final appended episode (source ep={current_episode})")

    return episodes_added, frames_added


def main():
    init_logging()
    args = parse_args()

    # Refuse to clobber an existing SHA output file.
    if args.sha_output_file.exists():
        raise FileExistsError(
            f"SHA output file already exists: {args.sha_output_file}. "
            "Refusing to overwrite — pick a fresh path or delete it deliberately."
        )

    logging.info(f"Loading target dataset: {args.target_repo_id}")
    target = LeRobotDataset(args.target_repo_id)
    logging.info(
        f"Target: {len(target)} frames, {target.meta.total_episodes} episodes, "
        f"fps={target.meta.fps}"
    )

    logging.info(f"Loading source dataset: {args.source_repo_id}")
    source = LeRobotDataset(args.source_repo_id)
    logging.info(
        f"Source: {len(source)} frames, {source.meta.total_episodes} episodes, "
        f"fps={source.meta.fps}"
    )

    problems = _features_compatible(target, source)
    if problems:
        logging.error("Pre-flight check failed — datasets are not compatible:")
        for p in problems:
            logging.error(f"  - {p}")
        raise SystemExit(1)
    logging.info("Pre-flight check passed: fps and features match.")

    # Snapshot the target's current HF SHA BEFORE any local mutation, so we have
    # a remote rollback point if anything downstream goes wrong.
    api = HfApi()
    info = api.repo_info(args.target_repo_id, repo_type="dataset")
    pre_merge_sha = info.sha
    args.sha_output_file.parent.mkdir(parents=True, exist_ok=True)
    args.sha_output_file.write_text(
        f"{args.target_repo_id} {pre_merge_sha}\n"
    )
    logging.info(
        f"Wrote pre-merge SHA to {args.sha_output_file}: "
        f"{args.target_repo_id} @ {pre_merge_sha}"
    )

    episodes_added, frames_added = append_source_into_target(target, source)
    target.finalize()
    logging.info(
        f"Append complete: +{episodes_added} episodes, +{frames_added} frames. "
        f"New totals: {target.meta.total_episodes} episodes, "
        f"{target.meta.total_frames} frames."
    )

    if args.push_to_hub:
        logging.info(f"Pushing merged dataset to {args.target_repo_id}")
        target.push_to_hub(
            tags=["aera", "ar4_mk3", "merged"],
            private=args.private,
            push_videos=True,
            license="apache-2.0",
            upload_large_folder=True,
        )
        logging.info("Push complete.")
        logging.info(
            f"If you need to revert, the pre-merge SHA is recorded in "
            f"{args.sha_output_file}."
        )
    else:
        logging.info("Skipped push (--push-to-hub not set). Local cache is updated.")

    logging.info(
        "Next step: re-run compute_norm_stats.py against the merged dataset."
    )


if __name__ == "__main__":
    main()
