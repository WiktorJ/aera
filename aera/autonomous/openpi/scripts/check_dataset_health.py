#!/usr/bin/env python3
"""Pre-training gate: is this dataset the one the next-run plan asked for?

Implements verification checks 2-7 of
training_journal/06.07.2026/next_run_changes.md against a built LeRobot dataset.
Training is the expensive step and every load-bearing claim in that plan is
measurable on data, so nothing should start until these pass.

Each check REPORTS ITS MEASURED VALUE against a threshold rather than asserting
a baked-in expectation. That is deliberate: the tool is calibrated by running it
on the OLD (16_06) dataset, where most checks must come back FAILING — 40%
near-static frames, 5.3% descent dynamic range, 8 leading spurious-closed
frames, no binarization. A checker that cannot reproduce the known pathologies
of the old data cannot be trusted to certify the new data.

Checks:
    2  per-step delta distribution (median / p99 / near-static / max:median)
    3  descent resolution — the last frames before the jaws close
    4  normalized output range on descent frames  [the headline gate]
    5  grasp-window frame count — is the close event still represented?
    6  gripper action binarization + no spurious closed frames at episode start
    7  no recovery injection — one grasp cycle per episode

Exit code is 1 if any check fails, so it can gate a pipeline directly.

Usage:
    python -m aera.autonomous.openpi.scripts.check_dataset_health \
        --repo-id Purple69/<dataset>_skip10_delta

    # calibration run: on 16_06 checks 2/4/6 are EXPECTED to fail
    python -m aera.autonomous.openpi.scripts.check_dataset_health \
        --repo-id Purple69/aera_semi_pnp_dr_16_06_2026_..._v2 \
        --frames-repo-id Purple69/aera_semi_pnp_dr_16_06_2026_..._v2_subset3ep
"""

import argparse
import dataclasses
import json
import logging
import sys
from typing import Any

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from aera.autonomous.openpi.scripts.measure_action_dynrange import (
    DESCENT_FRAMES,
    OPEN_CMD,
    SCENE_PATH,
    descent_indices,
    fk_fn,
    load_quantiles,
)

# Gripper command levels the transform's --binarize-gripper emits.
GRIPPER_OPEN = -0.014
GRIPPER_CLOSED = 0.0
# Engage/release hysteresis band, mirroring Ar4Mk3Env._GRASP_{ENGAGE,RELEASE}_CTRL.
CLOSE_BAND = -0.013
RELEASE_BAND = -0.0135


@dataclasses.dataclass
class CheckResult:
    number: int
    name: str
    passed: bool
    measured: dict[str, Any]
    expectation: str
    note: str = ""

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--repo-id", required=True,
                   help="dataset to check (also supplies meta/stats.json quantiles)")
    p.add_argument("--frames-repo-id", default=None,
                   help="read per-frame data from this dataset instead (e.g. a subset), "
                        "while quantiles still come from --repo-id")
    p.add_argument("--num-joint-dims", type=int, default=6)
    p.add_argument("--max-frames", type=int, default=20000)
    p.add_argument("--scene", default=SCENE_PATH)
    p.add_argument("--absolute-actions", action="store_true", default=False,
                   help="dataset stores absolute joint targets rather than deltas")

    # Thresholds — defaults are the next_run_changes.md targets for the slow arm
    # at skip=10. Override when checking a dataset built at a different skip.
    t = p.add_argument_group("thresholds")
    t.add_argument("--delta-med-mm", type=float, nargs=2, default=[2.4, 3.0],
                   metavar=("MIN", "MAX"))
    t.add_argument("--delta-p99-max-mm", type=float, default=5.0)
    t.add_argument("--near-static-max", type=float, default=0.10,
                   help="max fraction of frames moving < 2 mm")
    t.add_argument("--max-over-med-max", type=float, default=2.0)
    t.add_argument("--descent-med-max-mm", type=float, default=3.5)
    t.add_argument("--descent-max-mm", type=float, default=7.0)
    t.add_argument("--dynrange-min-pct", type=float, default=50.0)
    t.add_argument("--grasp-window-min-frames", type=int, default=3,
                   help="floor on frames spanning the open->closed transition")
    t.add_argument("--gripper-state-eps", type=float, default=1e-5,
                   help="floor on the jaw-motion threshold; the effective value is "
                        "max(this, 5%% of the episode's close travel)")

    p.add_argument("--json", action="store_true", default=False)
    p.add_argument("--no-fail-exit", action="store_true", default=False,
                   help="always exit 0 (calibration runs, where failures are expected)")
    return p.parse_args()


# --- data loading ---------------------------------------------------------


def load_frames(repo_id: str, max_frames: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ds = LeRobotDataset(repo_id)
    count = min(len(ds), max_frames)
    if count < len(ds):
        logging.info("reading %d of %d frames (--max-frames)", count, len(ds))
    actions, states, episodes = [], [], []
    for t in range(count):
        s = ds[t]
        actions.append(np.asarray(s["actions"], dtype=np.float64))
        states.append(np.asarray(s["state"], dtype=np.float64))
        episodes.append(int(s["episode_index"]))
    return np.array(actions), np.array(states), np.array(episodes)


def eef_steps_mm(actions, states, n, scene, absolute) -> np.ndarray:
    """Per-policy-step end-effector displacement, in millimetres.

    The action is what the arm is commanded to do over one decision interval, so
    its EEF-space size is the physical step the policy has to produce — the
    quantity the plan's delta tables are in.
    """
    fk = fk_fn(scene, n)
    target = actions[:, :n] if absolute else states[:, :n] + actions[:, :n]
    return np.array([
        np.linalg.norm(fk(target[t]) - fk(states[t, :n])) * 1000
        for t in range(len(states))
    ])


def grip_heights_m(states, n, scene) -> np.ndarray:
    fk = fk_fn(scene, n)
    return np.array([fk(states[t, :n])[2] for t in range(len(states))])


# --- per-episode gripper structure ---------------------------------------


@dataclasses.dataclass
class EpisodeProfile:
    episode: int
    frames: int
    leading_closed: int  # frames commanded closed before the first open
    transitions: int  # open<->closed switches (2 = one clean grasp cycle)
    close_at: int | None  # first open->closed index
    grasp_window: int  # frames over which the jaws physically close
    release_heights_m: list[float]


def _hysteretic_closed(grip_cmd: np.ndarray) -> np.ndarray:
    """Open/closed state of the gripper command, with the same hysteresis band
    the env's own engage/release logic uses (`Ar4Mk3Env._update_grasp_engagement`).

    A single threshold miscounts cycles on unbinarized data, where the recorded
    "action" is the measured jaw qpos and jitters across any fixed line. On
    binarized data the band is irrelevant — the signal only ever takes two
    values — so this costs nothing where it isn't needed.
    """
    closed = np.zeros(len(grip_cmd), dtype=bool)
    state = bool(grip_cmd[0] > CLOSE_BAND)
    for i, value in enumerate(grip_cmd):
        if value >= CLOSE_BAND:
            state = True
        elif value <= RELEASE_BAND:
            state = False
        closed[i] = state
    return closed


def _run_length(signal: np.ndarray, start: int, direction: int, eps: float) -> int:
    """Frames of continuous motion from `start` in `direction`, tolerating one
    stalled frame so a momentarily flat sample doesn't truncate the run."""
    count, stalled, i = 0, 0, start
    while 0 <= i + direction < len(signal) and stalled < 2:
        if abs(signal[i + direction] - signal[i]) > eps:
            count += 1
            stalled = 0
        else:
            stalled += 1
        i += direction
    return count


def profile_episode(
    grip_cmd: np.ndarray,
    grip_state: np.ndarray,
    heights: np.ndarray,
    eps: float,
) -> EpisodeProfile:
    closed = _hysteretic_closed(grip_cmd)
    switches = np.flatnonzero(closed[1:] != closed[:-1]) + 1

    leading = int(np.argmax(~closed)) if closed.any() and not closed.all() else (
        len(closed) if closed.all() else 0
    )
    close_at = next((int(i) for i in switches if closed[i]), None)
    releases = [float(heights[i]) for i in switches if not closed[i]]

    # Grasp window: how many frames the jaw STATE takes to travel from open to
    # its stall on the block. Measured on the state, not the command, because
    # binarization makes the command an instantaneous step by construction.
    #
    # The run is expanded BOTH ways from close_at, which matters on unbinarized
    # data: there the "action" is the measured next jaw qpos, so it only crosses
    # the closed threshold once the jaws are already ~90% shut, putting close_at
    # near the END of the ramp. Anchoring forward-only would report ~2 frames for
    # a ramp that actually spans ~17.
    window = 0
    if close_at is not None:
        # The motion threshold is a fraction of THIS episode's close travel, not
        # an absolute number: an open jaw jitters ~0.1 mm/frame against its
        # limit stop, which is the same order as the tail of a close ramp, so a
        # fixed epsilon either bleeds into the jitter or truncates the ramp.
        lo, hi = max(close_at - 8, 0), min(close_at + 8, len(grip_state) - 1)
        travel = abs(float(grip_state[hi] - grip_state[lo]))
        step_eps = max(eps, 0.05 * travel)
        window = 1 + _run_length(grip_state, close_at, +1, step_eps) + _run_length(
            grip_state, close_at, -1, step_eps
        )

    return EpisodeProfile(
        episode=-1,
        frames=len(grip_cmd),
        leading_closed=leading,
        transitions=int(len(switches)),
        close_at=close_at,
        grasp_window=window,
        release_heights_m=releases,
    )


def profile_all(actions, states, episodes, n, heights, eps) -> list[EpisodeProfile]:
    out = []
    for ep in sorted(set(episodes.tolist())):
        mask = np.flatnonzero(episodes == ep)
        prof = profile_episode(actions[mask, n], states[mask, n], heights[mask], eps)
        prof.episode = ep
        out.append(prof)
    return out


# --- checks ---------------------------------------------------------------


def check_delta_distribution(eef_mm: np.ndarray, args) -> CheckResult:
    med = float(np.median(eef_mm))
    p99 = float(np.percentile(eef_mm, 99))
    near_static = float((eef_mm < 2).mean())
    ratio = float(eef_mm.max() / med) if med else float("inf")
    lo, hi = args.delta_med_mm
    passed = (
        lo <= med <= hi
        and p99 <= args.delta_p99_max_mm
        and near_static <= args.near_static_max
        and ratio <= args.max_over_med_max
    )
    return CheckResult(
        2, "delta distribution", passed,
        {"median_mm": round(med, 2), "p99_mm": round(p99, 2),
         "max_mm": round(float(eef_mm.max()), 2),
         "near_static_lt2mm": round(near_static, 3),
         "max_over_median": round(ratio, 1)},
        f"median in [{lo}, {hi}] mm, p99 <= {args.delta_p99_max_mm}, "
        f"near-static <= {args.near_static_max}, max:median <= {args.max_over_med_max}",
        "a fat tail spends the normalized output range on sprint steps the "
        "descent never uses; near-static mass is the imitation loss learning "
        "to sit still",
    )


def check_descent_resolution(eef_mm, descent, args) -> CheckResult:
    if not len(descent):
        return CheckResult(3, "descent resolution", False, {},
                           "descent frames must be identifiable",
                           "no close command found — check 6 explains why")
    med = float(np.median(eef_mm[descent]))
    mx = float(eef_mm[descent].max())
    passed = med <= args.descent_med_max_mm and mx <= args.descent_max_mm
    return CheckResult(
        3, "descent resolution", passed,
        {"frames": int(len(descent)), "median_mm": round(med, 2), "max_mm": round(mx, 2)},
        f"median <= {args.descent_med_max_mm} mm, max <= {args.descent_max_mm} mm",
        "the slow arm's near-uniform velocity profile gives the descent the "
        "same step as transit; this is the known risk of the timescale change "
        f"(pinch_tol is +-7 mm). Fix at transform: drop to skip 5-6.",
    )


def check_dynamic_range(actions, descent, span, n, args) -> CheckResult:
    if not len(descent):
        return CheckResult(4, "normalized output range", False, {},
                           f">= {args.dynrange_min_pct}% of span",
                           "no descent frames identified")
    normalized = np.abs(actions[:, :n]) * 2.0 / span
    descent_pct = float(np.median(normalized[descent].max(axis=1)) * 100)
    all_pct = float(np.median(normalized.max(axis=1)) * 100)
    return CheckResult(
        4, "normalized output range", descent_pct >= args.dynrange_min_pct,
        {"descent_pct_of_span": round(descent_pct, 1),
         "all_frames_pct_of_span": round(all_pct, 1)},
        f"descent >= {args.dynrange_min_pct}% of the [-1,+1] span",
        "THE HEADLINE GATE. pi0.5 quantile-normalizes actions, so absolute "
        "delta size is normalized away and this ratio is what governs whether "
        "the grasp descent is learnable. 16_06 measured 5.3%.",
    )


def check_grasp_window(profiles: list[EpisodeProfile], args) -> CheckResult:
    windows = [p.grasp_window for p in profiles]
    no_close = [p.episode for p in profiles if p.close_at is None]
    med = float(np.median(windows))
    worst = int(min(windows))
    passed = worst >= args.grasp_window_min_frames and not no_close
    return CheckResult(
        5, "grasp-window representation", passed,
        {"median_frames": med, "min_frames": worst, "max_frames": int(max(windows)),
         # A window of 0 means no close was found at all, which is a different
         # (worse) problem than a close that is under-represented.
         "episodes_without_close": len(no_close)},
        f"every episode >= {args.grasp_window_min_frames} frames spanning open->closed",
        "the gripper ramp is a fixed count of mj-steps while the episode "
        "stretches ~8x, so the close event dilutes. Fix at transform: "
        "duplicate / weight grasp-window frames.",
    )


def check_binarization(actions, states, profiles, n) -> CheckResult:
    cmd = actions[:, n]
    levels = np.unique(np.round(cmd, 6))
    binarized = len(levels) <= 2 and all(
        min(abs(v - GRIPPER_OPEN), abs(v - GRIPPER_CLOSED)) < 1e-6 for v in levels
    )
    leading = [p.leading_closed for p in profiles]
    state_levels = int(len(np.unique(np.round(states[:, n], 5))))
    passed = binarized and max(leading) == 0 and state_levels > 10
    return CheckResult(
        6, "gripper binarization", passed,
        {"action_levels": int(len(levels)),
         # Listing 251 levels is noise; the range is what says "not binarized".
         "action_values": ([round(float(v), 4) for v in levels] if len(levels) <= 4
                           else f"range [{levels.min():.4f}, {levels.max():.4f}]"),
         "episodes_with_leading_closed": int(sum(1 for v in leading if v)),
         "leading_closed_frames_median": float(np.median(leading)),
         "max_leading_closed_frames": int(max(leading)),
         "state_distinct_values": state_levels},
        "action in {-0.014, 0} only; zero leading closed frames; state still continuous",
        "width-regression and ramp mid-values are what produce partial closes. "
        "Leading closed frames come from the jaws starting shut at qpos0 — an "
        "episode-start 'closed' label with nothing in the gripper.",
    )


def check_no_recovery(profiles: list[EpisodeProfile]) -> CheckResult:
    extra = [p for p in profiles if p.transitions > 2]
    releases_aloft = [
        round(h, 3) for p in extra for h in p.release_heights_m[:-1]
    ]
    return CheckResult(
        7, "no recovery injection", not extra,
        {"episodes": len(profiles),
         "episodes_with_extra_cycles": len(extra),
         "extra_cycle_episodes": [p.episode for p in extra[:10]],
         "non_final_release_heights_m": releases_aloft[:10]},
        "exactly one grasp cycle (2 transitions) per episode",
        "a partial_grasp episode releases while aloft and re-grasps, which is a "
        "literal premature-release demonstration.",
    )


# --- reporting ------------------------------------------------------------


def report(results: list[CheckResult], header: dict) -> None:
    for key, value in header.items():
        logging.info("%-22s %s", key, value)
    logging.info("")
    for r in results:
        logging.info("%s  check %d — %s", "PASS" if r.passed else "FAIL", r.number, r.name)
        logging.info("      measured : %s", ", ".join(f"{k}={v}" for k, v in r.measured.items()))
        logging.info("      expected : %s", r.expectation)
        if not r.passed and r.note:
            logging.info("      why      : %s", r.note)
    failed = [r.number for r in results if not r.passed]
    logging.info("")
    if failed:
        logging.info("FAILED checks: %s", ", ".join(str(f) for f in failed))
    else:
        logging.info("All checks passed.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    n = args.num_joint_dims

    frames_repo = args.frames_repo_id or args.repo_id
    actions, states, episodes = load_frames(frames_repo, args.max_frames)
    q01, q99 = load_quantiles(args.repo_id, n)
    span = q99 - q01

    eef_mm = eef_steps_mm(actions, states, n, args.scene, args.absolute_actions)
    heights = grip_heights_m(states, n, args.scene)
    profiles = profile_all(actions, states, episodes, n, heights, args.gripper_state_eps)
    descent = descent_indices(actions, episodes, n)

    results = [
        check_delta_distribution(eef_mm, args),
        check_descent_resolution(eef_mm, descent, args),
        check_dynamic_range(actions, descent, span, n, args),
        check_grasp_window(profiles, args),
        check_binarization(actions, states, profiles, n),
        check_no_recovery(profiles),
    ]

    header = {
        "dataset": args.repo_id,
        "frames from": frames_repo,
        "frames": len(actions),
        "episodes": len(profiles),
        "descent frames": len(descent),
        "frames/episode": round(len(actions) / max(len(profiles), 1), 1),
    }

    if args.json:
        print(json.dumps({
            "header": header,
            "checks": [r.to_dict() for r in results],
            "passed": all(r.passed for r in results),
        }))
    else:
        report(results, header)

    if not args.no_fail_exit and not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
