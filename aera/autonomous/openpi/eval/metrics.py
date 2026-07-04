"""Granular eval metrics for AR4 MK3 pick-and-place rollouts.

Binary success rate is a poor training signal for manipulation: it is flat at 0%
early in training and noisy at small N late on. Instead we decompose each episode
into a monotonic *progress funnel* and a set of *spawn-invariant scalars* computed
directly from MuJoCo sim state.

Funnel (fraction of episodes reaching each stage):
    reached -> grasped -> lifted -> transported -> placed (success)

Scalars (mean / p50 / p90 across episodes):
    reach_progress, place_progress  -- "fraction of the gap closed", normalized by
        each episode's own initial geometry so they don't conflate "policy did
        well" with "the block spawned close to the gripper / goal".
    max_lift_height                 -- height the object was raised above spawn (m).
    grasp_count, time_to_first_grasp, grasp_drop_rate -- grasp quality / stability.
    wrong_object_grasp_count        -- kinematic-lock engagements on a distractor
        block. Kept separate from the funnel: "grasped" means object0 only, so a
        wrong-block grab reads as a grounding failure, not manipulation progress.

The tracker reads the env's sim state (object0 / grip sites, goal, and the
kinematic grasp lock) and is reused by both the manual `run_policy_on_env.py`
script and the decoupled `eval_worker.py`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from aera.autonomous.envs.kinematic_grasp import GraspEngageConfig

# Default stage thresholds (metres). reach is taken from the grasp lock's coarse
# engage bound so "reached" means "within grasp range", not centre-to-centre 0.
_DEFAULT_LIFT_THRESH = 0.03  # object raised this far above spawn counts as lifted
_DEFAULT_TRANSPORT_THRESH = 0.05  # horizontal object->goal dist while lifted
_EPS = 1e-4  # denominator floor for progress ratios


@dataclasses.dataclass
class EpisodeMetrics:
    """Per-episode granular metrics. Booleans form the funnel; floats the scalars."""

    # Funnel stages (each implies the previous in a clean episode).
    reached: bool
    grasped: bool
    lifted: bool
    transported: bool
    placed: bool  # task success

    # Spawn-invariant progress in [0, 1]: fraction of the initial gap closed.
    reach_progress: float
    place_progress: float

    # Raw scalars (kept for debugging / sanity).
    max_lift_height: float
    grasp_count: int
    held_at_end: bool
    time_to_first_grasp: int | None  # steps after settle; None if never grasped
    wrong_object_grasp_count: int  # lock engagements on a distractor block

    # Absolute distances (m), useful when comparing against the ratios.
    min_reach_dist: float
    min_place_dist: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class MetricThresholds:
    """Stage thresholds. ``reach`` is resolved from the env's grasp config."""

    reach: float = GraspEngageConfig.max_distance  # grasp-envelope coarse bound
    lift: float = _DEFAULT_LIFT_THRESH
    transport: float = _DEFAULT_TRANSPORT_THRESH


class EpisodeTracker:
    """Accumulates funnel/scalar metrics over a single episode from sim state.

    Lifecycle: construct with the env, call :meth:`start` once after the settle
    steps (so the initial geometry is the real spawn pose), call :meth:`update`
    after every ``env.step``, then :meth:`finalize` to get an
    :class:`EpisodeMetrics`.
    """

    def __init__(self, env, thresholds: MetricThresholds | None = None):
        self.env = env
        self._lock = getattr(env, "_grasp_lock", None)
        engage_cfg = (
            self._lock.engage_config if self._lock is not None else GraspEngageConfig()
        )
        # reach threshold defaults to the grasp lock's coarse engage bound.
        self.t = thresholds or MetricThresholds(reach=engage_cfg.max_distance)

    # --- sim-state readers -------------------------------------------------
    def _object_pos(self) -> np.ndarray:
        return self.env._utils.get_site_xpos(self.env.model, self.env.data, "object0")

    def _grip_pos(self) -> np.ndarray:
        return self.env._utils.get_site_xpos(self.env.model, self.env.data, "grip")

    def _held_state(self, lift_height: float) -> tuple[bool, bool]:
        """(holding object0, holding a different block). Uses the kinematic
        lock's held-object name when present — the lock can attach distractor
        blocks too, and those must not count as task grasps. Falls back to
        "object0 lifted off the table" (physical grasp), which is object0-only
        by construction and can't see wrong-object grabs."""
        if self._lock is not None:
            held_name = self._lock.held_object
            return held_name == "object0", (
                held_name is not None and held_name != "object0"
            )
        return lift_height > self.t.lift, False

    # --- lifecycle ---------------------------------------------------------
    def start(self) -> None:
        obj = self._object_pos()
        grip = self._grip_pos()
        goal = np.asarray(self.env.goal)

        self._spawn_z = float(obj[2])
        self._d0_reach = float(np.linalg.norm(grip - obj))
        self._d0_place = float(np.linalg.norm(obj - goal))

        self._min_reach = self._d0_reach
        self._min_place = self._d0_place
        self._max_lift = 0.0
        self._min_transport = float("inf")  # horizontal object->goal while lifted

        self._grasp_count = 0
        self._wrong_grasp_count = 0
        self._prev_held = False
        self._prev_wrong_held = False
        self._held_at_end = False
        self._first_grasp_step: int | None = None
        self._steps = 0

    def update(self) -> None:
        self._steps += 1
        obj = self._object_pos()
        grip = self._grip_pos()
        goal = np.asarray(self.env.goal)

        self._min_reach = min(self._min_reach, float(np.linalg.norm(grip - obj)))
        self._min_place = min(self._min_place, float(np.linalg.norm(obj - goal)))

        lift = float(obj[2]) - self._spawn_z
        self._max_lift = max(self._max_lift, lift)

        held, wrong_held = self._held_state(lift)
        if held and not self._prev_held:
            self._grasp_count += 1
            if self._first_grasp_step is None:
                self._first_grasp_step = self._steps
        if wrong_held and not self._prev_wrong_held:
            self._wrong_grasp_count += 1
        if lift > self.t.lift:
            horiz = float(np.linalg.norm(obj[:2] - goal[:2]))
            self._min_transport = min(self._min_transport, horiz)

        self._prev_held = held
        self._prev_wrong_held = wrong_held
        self._held_at_end = held

    def finalize(self) -> EpisodeMetrics:
        threshold = float(self.env.distance_threshold)

        reached = self._min_reach <= self.t.reach
        grasped = self._grasp_count > 0
        lifted = self._max_lift > self.t.lift
        transported = self._min_transport < self.t.transport
        placed = self._min_place <= threshold

        # "Fraction of the gap closed", clipped to [0, 1]. Denominators are
        # floored so a block spawning already inside the threshold (degenerate)
        # yields progress 1.0 rather than a blow-up.
        reach_progress = _progress(self._d0_reach, self._min_reach, self.t.reach)
        place_progress = _progress(self._d0_place, self._min_place, threshold)

        return EpisodeMetrics(
            reached=reached,
            grasped=grasped,
            lifted=lifted,
            transported=transported,
            placed=placed,
            reach_progress=reach_progress,
            place_progress=place_progress,
            max_lift_height=self._max_lift,
            grasp_count=self._grasp_count,
            held_at_end=self._held_at_end,
            time_to_first_grasp=self._first_grasp_step,
            wrong_object_grasp_count=self._wrong_grasp_count,
            min_reach_dist=self._min_reach,
            min_place_dist=self._min_place,
        )


def _progress(d0: float, d_min: float, target: float) -> float:
    """Fraction of the initial gap (d0 -> target) that was closed, in [0, 1].

    Reaching the target (``d_min <= target``) is full progress, which also covers
    the degenerate case of spawning already inside the target (``d0 <= target``),
    where the gap-closed ratio would otherwise be undefined."""
    if d_min <= target:
        return 1.0
    return float(np.clip((d0 - d_min) / max(d0 - target, _EPS), 0.0, 1.0))


def aggregate(episodes: list[EpisodeMetrics]) -> dict[str, float]:
    """Reduce per-episode metrics into a flat dict ready for mlflow.log_metrics.

    Funnel stages become rates; scalars get mean/p50/p90; grasp stability gets a
    drop rate (grasped then ended without holding and without success)."""
    if not episodes:
        return {}

    n = len(episodes)
    out: dict[str, float] = {"eval/num_episodes": float(n)}

    # Funnel: fraction of episodes reaching each stage.
    for stage in ("reached", "grasped", "lifted", "transported", "placed"):
        out[f"eval/funnel/{stage}_rate"] = float(
            np.mean([getattr(e, stage) for e in episodes])
        )
    # Keep a plainly-named success metric alongside the funnel.
    out["eval/success_rate"] = out["eval/funnel/placed_rate"]

    # Scalars: mean + percentiles to show distribution, not just central tendency.
    for name in ("reach_progress", "place_progress", "max_lift_height"):
        vals = np.array([getattr(e, name) for e in episodes], dtype=float)
        out[f"eval/{name}_mean"] = float(np.mean(vals))
        out[f"eval/{name}_p50"] = float(np.percentile(vals, 50))
        out[f"eval/{name}_p90"] = float(np.percentile(vals, 90))

    out["eval/grasp_count_mean"] = float(np.mean([e.grasp_count for e in episodes]))
    out["eval/wrong_object_grasp_count_mean"] = float(
        np.mean([e.wrong_object_grasp_count for e in episodes])
    )

    # Time-to-first-grasp only over episodes that actually grasped.
    grasp_times = [
        e.time_to_first_grasp for e in episodes if e.time_to_first_grasp is not None
    ]
    if grasp_times:
        out["eval/time_to_first_grasp_mean"] = float(np.mean(grasp_times))

    # Grab-then-drop instability: of episodes that grasped, the fraction that
    # ended neither holding nor successful.
    grasped = [e for e in episodes if e.grasped]
    if grasped:
        dropped = [e for e in grasped if not e.held_at_end and not e.placed]
        out["eval/grasp_drop_rate"] = float(len(dropped) / len(grasped))

    return out
