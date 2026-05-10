"""Plot raw vs Savitzky-Golay smoothed actions for one episode.

Generates one PNG (and optionally a second for state) containing:

  Per-dim overlay subplots (one per action/state dimension):
    Raw signal (thin, semi-transparent) vs smoothed signal (thick) on the
    same axes. Use these to verify the filter is removing only jitter and
    not flattening real motion. Tells: "is the smoothing strength right?"
      - Curves match closely with smoothed slightly cleaner -> good.
      - Smoothed clips peaks or rounds sharp transitions -> window too big.
      - Smoothed still looks noisy -> window too small.

  Cumulative drift subplot — ||cumsum(raw - smoothed)||:
    The L2 norm of the running sum of (raw - smoothed) across all dims.
    This is the trajectory drift you would see if actions were treated as
    deltas and integrated forward: at frame t, this is how far the
    integrated-smoothed trajectory has diverged from the integrated-raw
    trajectory. A bounded curve that oscillates back toward zero proves SG
    smoothing does not accumulate systematic error. A monotonically
    growing curve would indicate a real bias introduced by the filter.

  Relative drift subplot — cumulative drift / cumulative path length:
    Same numerator as above, divided by the running L2 path length of the
    raw action signal up to frame t. Unitless. Tells you how big the drift
    is relative to how much the arm has actually moved. Below a few percent
    is comfortably within the noise floor.

Usage:
    python -m aera.autonomous.openpi.scripts.plot_smoothing_comparison \
        --repo-id Purple69/aera_semi_pnp_dr_08_01_2026 \
        --episode-index 0 \
        --smooth-window 11 \
        --smooth-polyorder 3 \
        --output smoothing_ep0.png
"""

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from aera.autonomous.openpi.dataset_transforms import compute_smoothed_arrays


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", type=str, required=True)
    p.add_argument("--episode-index", type=int, default=0)
    p.add_argument("--smooth-window", type=int, default=11, help="Odd integer.")
    p.add_argument("--smooth-polyorder", type=int, default=3)
    p.add_argument(
        "--include-state",
        action="store_true",
        help="Also overlay raw vs smoothed state in a second figure.",
    )
    p.add_argument("--output", type=str, default="smoothing_comparison.png")
    return p.parse_args()


def _plot_overlay(raw: np.ndarray, smoothed: np.ndarray, title: str, output: str) -> None:
    """Render the per-dim overlay + cumulative-drift figure.

    Layout: one subplot per dim showing raw vs smoothed, then a final
    subplot showing the L2 norm of the cumulative residual.
    See the module docstring for how to interpret each panel.
    """
    n_frames, n_dims = raw.shape
    fig, axes = plt.subplots(
        n_dims + 2, 1, figsize=(12, 2.2 * (n_dims + 2)), sharex=True
    )

    # Per-dim panels: raw vs smoothed overlaid. Diagnoses smoothing strength.
    # The first panel carries an explanatory title that applies to all dim panels.
    for d in range(n_dims):
        axes[d].plot(raw[:, d], label="raw", alpha=0.55, linewidth=0.8)
        axes[d].plot(smoothed[:, d], label="smoothed", linewidth=1.2)
        axes[d].set_ylabel(f"dim {d}")
        axes[d].grid(True, alpha=0.3)
        if d == 0:
            axes[d].legend(loc="upper right")
            axes[d].set_title(
                "Per-dim overlay: raw (thin) vs smoothed (thick). "
                "Look for jitter removed without clipping real motion.",
                fontsize=10,
            )

    # Cumulative drift panel: ||cumsum(raw - smoothed)|| — drift that would
    # accumulate if smoothed actions were integrated forward instead of raw.
    # Bounded oscillation around zero == filter introduces no systematic bias.
    cum_diff = np.cumsum(raw - smoothed, axis=0)
    cum_norm = np.linalg.norm(cum_diff, axis=1)
    axes[-2].plot(cum_norm, color="red")
    axes[-2].set_ylabel("||cumsum(raw - smoothed)||")
    axes[-2].set_title(
        "Cumulative drift if smoothed actions were integrated as deltas. "
        "Bounded/oscillating = no accumulating bias; growing = filter is biased.",
        fontsize=10,
    )
    axes[-2].grid(True, alpha=0.3)

    # Relative drift panel: drift / cumulative path length of the raw signal.
    # Unitless. Reads as "fraction of motion that's drift" — sub-percent is fine.
    step_norm = np.linalg.norm(np.diff(raw, axis=0), axis=1)
    cum_path = np.concatenate([[0.0], np.cumsum(step_norm)])
    eps = 1e-12
    rel_drift = cum_norm / np.maximum(cum_path, eps)
    axes[-1].plot(rel_drift * 100.0, color="purple")
    axes[-1].set_ylabel("drift / path length [%]")
    axes[-1].set_xlabel("frame index in episode")
    axes[-1].set_title(
        "Relative drift: cumulative drift divided by raw path length so far. "
        "A few percent is well within the noise floor.",
        fontsize=10,
    )
    axes[-1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output, dpi=120)
    plt.close(fig)
    logging.info(f"Saved: {output}")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )
    args = parse_args()

    logging.info(f"Loading {args.repo_id} (episode {args.episode_index} only)")
    ds = LeRobotDataset(args.repo_id, episodes=[args.episode_index])
    hf = ds.hf_dataset

    if len(hf) == 0:
        raise ValueError(f"Episode {args.episode_index} not found in {args.repo_id}.")

    smoothed_actions_all, smoothed_state_all, excluded = compute_smoothed_arrays(
        ds,
        window=args.smooth_window,
        polyorder=args.smooth_polyorder,
        smooth_state=args.include_state,
    )
    if args.episode_index in excluded:
        raise ValueError(
            f"Episode {args.episode_index} is shorter than --smooth-window={args.smooth_window} "
            "and would be excluded by the transform script."
        )

    raw_actions = np.asarray(hf["actions"], dtype=np.float32)
    smoothed_actions = smoothed_actions_all
    logging.info(
        f"Episode {args.episode_index}: {raw_actions.shape[0]} frames, "
        f"action_dim={raw_actions.shape[1]}"
    )

    title = (
        f"actions | episode {args.episode_index} | "
        f"window={args.smooth_window} polyorder={args.smooth_polyorder}"
    )
    _plot_overlay(raw_actions, smoothed_actions, title, args.output)

    if args.include_state:
        raw_state = np.asarray(hf["state"], dtype=np.float32)
        smoothed_state = smoothed_state_all
        state_output = args.output.replace(".png", "_state.png")
        if state_output == args.output:
            state_output = args.output + "_state.png"
        state_title = title.replace("actions |", "state |")
        _plot_overlay(raw_state, smoothed_state, state_title, state_output)


if __name__ == "__main__":
    main()
