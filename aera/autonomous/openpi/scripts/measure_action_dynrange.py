#!/usr/bin/env python3
"""How much of the model's normalized output range does a training action use?

pi0.5 (`ModelType.PI05` => `use_quantile_norm=True`) maps each action dim's
[q01, q99] onto [-1, +1], so the absolute size of an action delta is normalized
away. What actually governs whether the grasp descent is learnable is
`|a_i| * 2 / (q99_i - q01_i)` on descent frames — the fraction of the output
span the descent commands use. Measured on the 16_06 data that is 5.3%; the
slow-arm plan is supposed to bring it to >= ~50%.

This is verification check 4 of training_journal/06.07.2026/next_run_changes.md,
and unlike `measure_scripted_arm dynrange` it uses the REAL dataset's own
quantiles (`meta/stats.json` — the same numbers openpi normalizes with), so it
is the authoritative version. Run it on the small verification dataset before
launching a training.

Usage:
    python -m aera.autonomous.openpi.scripts.measure_action_dynrange \
        --repo-id Purple69/<dataset>_skip10_delta

    # quantiles from the full dataset, per-frame actions from a small subset
    python -m aera.autonomous.openpi.scripts.measure_action_dynrange \
        --repo-id Purple69/<full> --frames-repo-id Purple69/<full>_subset3ep
"""

import argparse
import json
import logging
import os
import pathlib

import mujoco
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

SCENE_PATH = "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml"
ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]
# Gripper action band: open is -0.014, so a value above this is a close command.
OPEN_CMD = -0.013
DESCENT_FRAMES = 12  # policy steps before the close, per the plan's definition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", required=True,
                   help="dataset whose meta/stats.json supplies the q01/q99 quantiles")
    p.add_argument("--frames-repo-id", default=None,
                   help="dataset to read per-frame actions from (default: --repo-id)")
    p.add_argument("--num-joint-dims", type=int, default=6)
    p.add_argument("--max-frames", type=int, default=20000,
                   help="cap on frames read (the full set is not needed for medians)")
    p.add_argument("--scene", default=SCENE_PATH,
                   help="MuJoCo scene used for FK, to report EEF millimetres alongside")
    p.add_argument("--json", action="store_true", default=False)
    return p.parse_args()


def load_quantiles(repo_id: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """q01/q99 per action dim from the dataset's own stats — the exact values
    openpi's quantile norm uses."""
    stats_path = (
        pathlib.Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        / "meta" / "stats.json"
    )
    if not stats_path.exists():
        raise FileNotFoundError(
            f"{stats_path} not found — pull the dataset locally first "
            "(LeRobotDataset(repo_id) caches it)."
        )
    stats = json.loads(stats_path.read_text())["actions"]
    return np.asarray(stats["q01"])[:n], np.asarray(stats["q99"])[:n]


def fk_fn(scene_path: str, n_joints: int):
    """Forward kinematics for the grip site, so normalized numbers can be read
    next to physical millimetres."""
    model = mujoco.MjModel.from_xml_path(os.path.abspath(scene_path))
    data = mujoco.MjData(model)
    adr = [model.joint(j).qposadr[0] for j in ARM_JOINTS[:n_joints]]
    site_id = model.site("grip").id

    def fk(q):
        data.qpos[adr] = q
        mujoco.mj_forward(model, data)
        return data.site_xpos[site_id].copy()

    return fk


def descent_indices(actions: np.ndarray, episodes: np.ndarray, n: int) -> np.ndarray:
    """Frames in the last `DESCENT_FRAMES` policy steps before each episode's
    first close command — the phase that decides the grasp."""
    out = []
    for ep in sorted(set(episodes.tolist())):
        mask = np.flatnonzero(episodes == ep)
        grip = actions[mask, n]
        opened = grip <= OPEN_CMD
        close_at = next(
            (i for i in range(1, len(grip)) if opened[i - 1] and not opened[i]),
            len(grip) - 1,
        )
        out.append(mask[max(0, close_at - DESCENT_FRAMES):close_at])
    return np.concatenate(out) if out else np.array([], dtype=int)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    n = args.num_joint_dims

    q01, q99 = load_quantiles(args.repo_id, n)
    span = q99 - q01  # this is what maps onto [-1, +1]

    frames_repo = args.frames_repo_id or args.repo_id
    ds = LeRobotDataset(frames_repo)
    count = min(len(ds), args.max_frames)
    logging.info("quantiles from %s | frames from %s (%d of %d)",
                 args.repo_id, frames_repo, count, len(ds))

    actions, states, episodes = [], [], []
    for t in range(count):
        s = ds[t]
        actions.append(np.asarray(s["actions"], dtype=np.float64))
        states.append(np.asarray(s["state"], dtype=np.float64))
        episodes.append(int(s["episode_index"]))
    actions = np.array(actions)
    states = np.array(states)
    episodes = np.array(episodes)

    # Affine normalization => magnitudes scale as 2*|a|/span.
    normalized = np.abs(actions[:, :n]) * 2.0 / span
    fk = fk_fn(args.scene, n)
    eef_mm = np.array([
        np.linalg.norm(fk(states[t, :n] + actions[t, :n]) - fk(states[t, :n])) * 1000
        for t in range(len(actions))
    ])

    descent = descent_indices(actions, episodes, n)
    result = {
        "quantile_repo": args.repo_id,
        "frames_repo": frames_repo,
        "frames": int(len(actions)),
        "episodes": int(len(set(episodes.tolist()))),
        "descent_frames": int(len(descent)),
        "all_eef_med_mm": round(float(np.median(eef_mm)), 2),
        "all_eef_p99_mm": round(float(np.percentile(eef_mm, 99)), 2),
        "all_worst_joint_pct_of_span": round(
            float(np.median(normalized.max(axis=1)) * 100), 1
        ),
        "descent_eef_med_mm": (
            round(float(np.median(eef_mm[descent])), 2) if len(descent) else None
        ),
        # The headline number: today 5.3%, expected >= ~50% after the re-collect.
        "descent_worst_joint_pct_of_span": (
            round(float(np.median(normalized[descent].max(axis=1)) * 100), 1)
            if len(descent) else None
        ),
        "near_static_lt2mm": round(float((eef_mm < 2).mean()), 3),
    }

    if args.json:
        print(json.dumps(result))
        return
    logging.info("\naction q01/q99 per joint (rad):")
    for i in range(n):
        logging.info("   j%d: [%+.5f, %+.5f]  span=%.5f", i + 1, q01[i], q99[i], span[i])
    logging.info("")
    for key, value in result.items():
        logging.info("   %-32s %s", key, value)


if __name__ == "__main__":
    main()
