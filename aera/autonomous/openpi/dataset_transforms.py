"""Dataset-level transformations applied to a LeRobotDataset before training.

These operate on the entire dataset (or whole episodes within it), as opposed
to per-sample model input transforms (see ``data_transform.py``).

Currently provides:
  - :func:`compute_smoothed_arrays` — Savitzky-Golay smoothing of action and
    optionally state arrays, applied per-episode on the raw source data.

More dataset-level utilities (skip/subsample, delta-action conversion, prompt
filtering, etc.) can move here over time.
"""

from __future__ import annotations

import logging

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from scipy.signal import savgol_filter


def compute_smoothed_arrays(
    source_dataset: LeRobotDataset,
    window: int,
    polyorder: int,
    smooth_state: bool,
) -> tuple[np.ndarray, np.ndarray | None, set[int]]:
    """Pre-compute Savitzky-Golay smoothed actions (and optionally state) per-episode.

    Smoothing is applied independently to each episode using a zero-phase
    Savitzky-Golay filter, so the per-step deviation from the raw signal is
    bounded by the jitter being removed and does not accumulate over time.

    Args:
        source_dataset: The source LeRobot dataset.
        window: Savitzky-Golay window length. Must be a positive odd integer.
        polyorder: Polynomial order. Must be < ``window``.
        smooth_state: If True, also smooth the ``state`` column.

    Returns:
        smoothed_actions: ``(N, action_dim)`` float32 array indexed by global
            frame index. Episodes that were excluded retain their raw values.
        smoothed_state: ``(N, state_dim)`` float32 array, or ``None`` if
            ``smooth_state`` is False.
        excluded_episodes: Set of episode indices skipped because they had
            fewer frames than ``window``. Frames belonging to these episodes
            should be dropped by the caller.
    """
    if window <= 0:
        raise ValueError(f"window must be > 0, got {window}")
    if window % 2 == 0:
        raise ValueError(f"window must be odd, got {window}")
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be < window ({window})")

    hf = source_dataset.hf_dataset
    all_actions = np.asarray(hf["actions"], dtype=np.float32)
    all_episodes = np.asarray(hf["episode_index"], dtype=np.int64)
    all_states = np.asarray(hf["state"], dtype=np.float32) if smooth_state else None

    smoothed_actions = all_actions.copy()
    smoothed_state = all_states.copy() if smooth_state else None

    excluded: set[int] = set()
    for ep in np.unique(all_episodes):
        mask = all_episodes == ep
        ep_len = int(mask.sum())
        if ep_len < window:
            logging.warning(
                f"Episode {int(ep)} has {ep_len} frames < smooth window {window}; excluding it."
            )
            excluded.add(int(ep))
            continue
        smoothed_actions[mask] = savgol_filter(
            all_actions[mask], window_length=window, polyorder=polyorder, axis=0
        )
        if smooth_state:
            smoothed_state[mask] = savgol_filter(
                all_states[mask], window_length=window, polyorder=polyorder, axis=0
            )

    logging.info(
        f"Smoothing applied: window={window}, polyorder={polyorder}, "
        f"smooth_state={smooth_state}, excluded_episodes={len(excluded)}"
    )
    return smoothed_actions, smoothed_state, excluded
