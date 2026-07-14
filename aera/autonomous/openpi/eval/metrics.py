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

Failure-mode diagnostics (manual spot checks in
training_journal/06.07.2026/NOTES.md were the only way to see *how* episodes
failed; these make the same observations automatic). All of them lean on the
kinematic grasp lock's command-driven semantics, so they are only tracked when
the lock is active:

    grasp attempts      -- a close command near the block that fails to engage
        the lock is a *missed grasp*; the grip->object offset at the attempt's
        closest moment is recorded in the gripper TOOL FRAME (pinch / finger /
        height -- the same anisotropic axes the engage gate checks), so misses
        decompose into "off to the side" vs "in front" vs "too high", plus a
        close-depth check for half-hearted close commands.
    releases            -- with the lock, a drop can ONLY be a commanded
        release. Each held->released transition records where the object was
        let go (distance to goal, height, hold duration), separating premature
        drops from intentional places.
    gripper close cycles -- open->close command cycles; a well-behaved episode
        has ~1 (the grasp) or 2 (grasp + place-release), retry loops and jaw
        "pulsing" push it far higher.
    table press         -- contact force from object0 onto the table far above
        the block's own weight while a jaw touches it and nothing is held:
        the "jaws push the block into the table" mode (catastrophic on real
        hardware, invisible in the funnel).
    pushed_dist_pre_grasp -- how far the block was shoved from its spawn before
        ever being grasped (blind nudging / dragging).
    failure_mode        -- one categorical label per episode (see
        FAILURE_MODES) classifying the furthest-stage outcome, so aggregate
        tables and video filenames say *what went wrong*, not just that
        something did.

The tracker reads the env's sim state (object0 / grip sites, goal, and the
kinematic grasp lock) and is reused by both the manual `run_policy_on_env.py`
script and the decoupled `eval_worker.py`.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import mujoco
import numpy as np

from aera.autonomous.envs.kinematic_grasp import GraspEngageConfig

# Default stage thresholds (metres). reach is taken from the grasp lock's coarse
# engage bound so "reached" means "within grasp range", not centre-to-centre 0.
_DEFAULT_LIFT_THRESH = 0.03  # object raised this far above spawn counts as lifted
_DEFAULT_TRANSPORT_THRESH = 0.05  # horizontal object->goal dist while lifted
_EPS = 1e-4  # denominator floor for progress ratios

# A close command within this grip->object distance counts as a grasp *attempt*
# (well outside the 5 cm engage bound, so clearly-off approaches still register
# as attempts rather than disappearing from the stats).
_ATTEMPT_RADIUS = 0.10
# Table-press detector: object->table normal force above this multiple of the
# block's own weight, while a jaw touches the block, means the gripper is
# pressing the block into the table rather than resting near it.
_PRESS_FORCE_RATIO = 3.0

# Why a failed grasp attempt missed, from the engage gate's own axes (see
# GraspEngageConfig): evaluated at the attempt's closest-approach sample.
# "no_pinch": every offset/depth gate passed but the jaws never physically
# pinched the block during the attempt (both-pads contact gate) — e.g. the
# close shoved it away, or it sat outside/below the jaw pads.
MISS_REASONS = (
    "coarse_far",
    "pinch",
    "finger",
    "height",
    "close_shallow",
    "no_pinch",
    "unknown",
)

# Per-episode outcome labels, ordered by how far the episode got.
FAILURE_MODES = (
    "success",
    "never_reached",  # never got within grasp range of the block
    "no_grasp_attempt",  # got close but never commanded a close there
    "grasp_missed",  # close attempts near the block, none engaged
    "wrong_object_grasp",  # only ever locked a distractor block
    "grasped_not_lifted",  # engaged but released before lifting
    "dropped_early",  # lifted, released far from the goal
    "dropped_or_missed_at_goal",  # got near the goal but didn't finish
    "timeout_holding",  # still holding the block when the episode ended
)


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

    # --- Failure-mode diagnostics (kinematic-lock rollouts only; empty/zero
    # otherwise). See module docstring.
    failure_mode: str
    # One dict per close-command-near-the-block window: start/end step, whether
    # it engaged, closest grip->object distance, tool-frame offset at that
    # moment (pinch/finger/height, m), the close command then, and (for misses)
    # which gate axes it violated.
    grasp_attempts: list[dict]
    grasp_attempt_count: int
    failed_grasp_attempts: int
    # One dict per commanded release of object0: step, hold duration, distance
    # to goal (3D + horizontal), height above spawn, and premature flag.
    releases: list[dict]
    premature_release_count: int
    gripper_close_cycles: int
    # Deepest close command seen near the block (ctrl units, -0.014 open -> 0
    # closed); None if never near. Low values = policy never commits to closing.
    deepest_close_cmd_near_object: float | None
    press_steps: int  # steps spent pressing the block into the table
    max_table_press_ratio: float  # peak object->table force / block weight
    pushed_dist_pre_grasp: float  # how far the block was shoved before any grasp

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
        self._engage_cfg = engage_cfg
        # reach threshold defaults to the grasp lock's coarse engage bound.
        self.t = thresholds or MetricThresholds(reach=engage_cfg.max_distance)

        # Gripper-command tracking mirrors the env's own engage/release
        # inference (Ar4Mk3Env._update_grasp_engagement); only meaningful with
        # the kinematic lock, whose ctrl thresholds these are.
        self._gripper_act_ids = getattr(env, "_gripper_act_ids", None)
        self._engage_ctrl = getattr(env, "_GRASP_ENGAGE_CTRL", None)
        self._release_ctrl = getattr(env, "_GRASP_RELEASE_CTRL", None)
        self._cmd_tracking = (
            self._lock is not None
            and self._gripper_act_ids is not None
            and self._engage_ctrl is not None
        )

        # Geom/body ids for the table-press detector (stable across resets;
        # sizes/masses are re-read in start() since block presets change per
        # episode). Missing geoms (e.g. the eef scene variant) disable it.
        model = env.model
        self._obj_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "object0"
        )
        self._table_geom_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "table_collision"
        )
        self._jaw_geom_ids = {
            gid
            for name in ("gripper_jaw1_contact", "gripper_jaw2_contact")
            if (gid := mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)) != -1
        }
        self._press_enabled = (
            self._obj_geom_id != -1
            and self._table_geom_id != -1
            and bool(self._jaw_geom_ids)
        )

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

    def _gripper_cmd(self) -> float:
        """Current commanded jaw target (ctrl units: -0.014 open -> 0 closed)."""
        return float(self.env.data.ctrl[self._gripper_act_ids].mean())

    def _tool_frame_offset(self) -> np.ndarray:
        """grip->object offset rotated into the gripper body frame — the same
        (pinch, finger, height) axes the engage gate bounds per-axis."""
        body_id = self.env.model.body(self._lock.gripper_body_name).id
        q_inv = np.empty(4)
        mujoco.mju_negQuat(q_inv, self.env.data.xquat[body_id])
        local = np.empty(3)
        mujoco.mju_rotVecQuat(local, self._object_pos() - self._grip_pos(), q_inv)
        return local

    def _table_press(self) -> tuple[float, bool]:
        """(total object->table normal force, any jaw touching object0)."""
        table_force = 0.0
        jaw_contact = False
        data = self.env.data
        wrench = np.zeros(6)
        for i in range(data.ncon):
            geoms = {data.contact[i].geom1, data.contact[i].geom2}
            if self._obj_geom_id not in geoms:
                continue
            if self._table_geom_id in geoms:
                mujoco.mj_contactForce(self.env.model, data, i, wrench)
                table_force += abs(float(wrench[0]))
            elif geoms & self._jaw_geom_ids:
                jaw_contact = True
        return table_force, jaw_contact

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

        # --- failure-mode diagnostics state ---
        self._spawn_xy = obj[:2].copy()
        self._attempts: list[dict] = []
        self._attempt: dict | None = None  # open attempt window, if any
        self._releases: list[dict] = []
        self._close_cycles = 0
        self._ever_held = False
        self._held_steps = 0
        self._deepest_close_near: float | None = None
        self._press_steps = 0
        self._max_press_ratio = 0.0
        self._pushed_pre_grasp = 0.0
        # Command-state for cycle counting, seeded from the current command so
        # the settle-phase closed gripper doesn't count as a cycle. States:
        # "close" / "open" / None (in the hysteresis band).
        if self._cmd_tracking:
            cmd = self._gripper_cmd()
            self._cmd_state = (
                "close"
                if cmd >= self._engage_ctrl
                else "open"
                if cmd <= self._release_ctrl
                else None
            )
        # Block half-width across the pinch axis, re-read per episode (block
        # size presets change at reset); drives the close-depth miss check.
        if self._obj_geom_id != -1:
            size = self.env.model.geom_size[self._obj_geom_id]
            self._pinch_half_width = float(min(size[0], size[1]))
        else:
            self._pinch_half_width = None
        obj_body_id = mujoco.mj_name2id(
            self.env.model, mujoco.mjtObj.mjOBJ_BODY, "object0"
        )
        gravity = abs(float(self.env.model.opt.gravity[2])) or 9.81
        self._obj_weight = (
            float(self.env.model.body_mass[obj_body_id]) * gravity
            if obj_body_id != -1
            else None
        )

    def update(self) -> None:
        self._steps += 1
        obj = self._object_pos()
        grip = self._grip_pos()
        goal = np.asarray(self.env.goal)

        reach_dist = float(np.linalg.norm(grip - obj))
        self._min_reach = min(self._min_reach, reach_dist)
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

        self._update_diagnostics(obj, goal, held, lift, reach_dist)

        self._prev_held = held
        self._prev_wrong_held = wrong_held
        self._held_at_end = held

    # --- failure-mode diagnostics --------------------------------------------
    def _update_diagnostics(
        self, obj: np.ndarray, goal: np.ndarray, held: bool, lift: float,
        reach_dist: float,
    ) -> None:
        if not self._ever_held:
            self._pushed_pre_grasp = max(
                self._pushed_pre_grasp,
                float(np.linalg.norm(obj[:2] - self._spawn_xy)),
            )
        if held:
            self._ever_held = True
            self._held_steps += 1
        elif self._prev_held:
            # With the kinematic lock a held->free transition can only be a
            # commanded release: record where the policy let go.
            horiz = float(np.linalg.norm(obj[:2] - goal[:2]))
            self._releases.append(
                {
                    "step": self._steps,
                    "hold_len": self._held_steps,
                    "dist_to_goal": float(np.linalg.norm(obj - goal)),
                    "horiz_dist_to_goal": horiz,
                    "height_above_spawn": lift,
                    "premature": horiz > self.t.transport,
                }
            )
            self._held_steps = 0

        if self._press_enabled and not held and self._obj_weight:
            force, jaw_contact = self._table_press()
            if jaw_contact:
                ratio = force / self._obj_weight
                self._max_press_ratio = max(self._max_press_ratio, ratio)
                if ratio > _PRESS_FORCE_RATIO:
                    self._press_steps += 1

        if not self._cmd_tracking:
            return
        cmd = self._gripper_cmd()
        close_cmd = cmd >= self._engage_ctrl
        # Open->close cycle counting, with the env's own hysteresis band.
        if close_cmd:
            if self._cmd_state == "open":
                self._close_cycles += 1
            self._cmd_state = "close"
        elif cmd <= self._release_ctrl:
            self._cmd_state = "open"

        near = reach_dist < _ATTEMPT_RADIUS
        if near and not held:
            self._deepest_close_near = (
                cmd
                if self._deepest_close_near is None
                else max(self._deepest_close_near, cmd)
            )

        # Grasp-attempt window: close command near the (unheld) block. Track
        # the tool-frame offset at the closest approach; the window ends when
        # the lock engages or the command/proximity condition lapses.
        if held:
            if self._attempt is not None:
                self._finish_attempt(engaged=True)
            elif not self._prev_held:
                # Engaged on the very step the close command appeared, so no
                # window ever opened; record a trivially-successful attempt.
                local = self._tool_frame_offset()
                self._attempts.append(
                    {
                        "start_step": self._steps,
                        "end_step": self._steps,
                        "engaged": True,
                        "min_dist": reach_dist,
                        "pinch": float(local[0]),
                        "finger": float(local[1]),
                        "height": float(local[2]),
                        "close_cmd": cmd,
                        "pinched": True,
                        "miss_reasons": [],
                    }
                )
        elif close_cmd and near:
            if self._attempt is None:
                self._attempt = {
                    "start_step": self._steps,
                    "end_step": self._steps,
                    "engaged": False,
                    "min_dist": float("inf"),
                    "pinch": 0.0,
                    "finger": 0.0,
                    "height": 0.0,
                    "close_cmd": cmd,
                    "pinched": False,
                    "miss_reasons": [],
                }
            if reach_dist < self._attempt["min_dist"]:
                local = self._tool_frame_offset()
                self._attempt.update(
                    min_dist=reach_dist,
                    pinch=float(local[0]),
                    finger=float(local[1]),
                    height=float(local[2]),
                    close_cmd=cmd,
                )
            # Did the jaws ever physically pinch the block during the attempt?
            # (Both-pads contact gate; drives the "no_pinch" miss reason.)
            if not self._attempt["pinched"] and self._lock is not None:
                try:
                    self._attempt["pinched"] = self._lock.jaws_pinching("object0")
                except ValueError:
                    self._attempt["pinched"] = True  # model lacks pad geoms
        elif self._attempt is not None:
            self._finish_attempt(engaged=False)

    def _finish_attempt(self, engaged: bool) -> None:
        att = self._attempt
        self._attempt = None
        att["end_step"] = self._steps
        att["engaged"] = engaged
        if not engaged:
            att["miss_reasons"] = self._classify_miss(att)
        self._attempts.append(att)

    def _classify_miss(self, att: dict) -> list[str]:
        """Which engage-gate axes the attempt's closest sample violated."""
        cfg = self._engage_cfg
        reasons = []
        if att["min_dist"] > cfg.max_distance:
            reasons.append("coarse_far")
        else:
            if abs(att["pinch"]) > cfg.pinch_tol:
                reasons.append("pinch")
            if abs(att["finger"]) > cfg.finger_tol:
                reasons.append("finger")
            if abs(att["height"]) > cfg.height_tol:
                reasons.append("height")
        if self._pinch_half_width is not None and att["close_cmd"] < -(
            self._pinch_half_width + cfg.close_depth_tol
        ):
            reasons.append("close_shallow")
        if not reasons and not att.get("pinched", True):
            reasons.append("no_pinch")
        if not reasons:
            # Gate passed at our (post-step) closest sample but the lock never
            # engaged — usually a transient the engage-time (pre-step) check
            # missed, or the jaws shoved the block while closing.
            reasons.append("unknown")
        return reasons

    def _classify_failure(
        self, placed: bool, reached: bool, grasped: bool, lifted: bool,
        transported: bool,
    ) -> str:
        if placed:
            return "success"
        if not reached:
            return "never_reached"
        if not grasped:
            if self._wrong_grasp_count > 0:
                return "wrong_object_grasp"
            if not self._attempts:
                return "no_grasp_attempt"
            return "grasp_missed"
        if self._held_at_end:
            return "timeout_holding"
        if not lifted:
            return "grasped_not_lifted"
        if not transported:
            return "dropped_early"
        return "dropped_or_missed_at_goal"

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

        # An attempt window still open at episode end is a failed attempt.
        if self._attempt is not None:
            self._finish_attempt(engaged=False)
        failed_attempts = sum(1 for a in self._attempts if not a["engaged"])

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
            failure_mode=self._classify_failure(
                placed, reached, grasped, lifted, transported
            ),
            grasp_attempts=self._attempts,
            grasp_attempt_count=len(self._attempts),
            failed_grasp_attempts=failed_attempts,
            releases=self._releases,
            premature_release_count=sum(1 for r in self._releases if r["premature"]),
            gripper_close_cycles=self._close_cycles,
            deepest_close_cmd_near_object=self._deepest_close_near,
            press_steps=self._press_steps,
            max_table_press_ratio=self._max_press_ratio,
            pushed_dist_pre_grasp=self._pushed_pre_grasp,
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

    # --- Failure-mode diagnostics -------------------------------------------
    # Outcome distribution over the fixed label set (all labels always logged,
    # so mlflow curves exist even while a mode sits at 0).
    for mode in FAILURE_MODES:
        out[f"eval/failure/{mode}_rate"] = float(
            np.mean([e.failure_mode == mode for e in episodes])
        )

    out["eval/grasp_attempts_mean"] = float(
        np.mean([e.grasp_attempt_count for e in episodes])
    )
    out["eval/failed_grasp_attempts_mean"] = float(
        np.mean([e.failed_grasp_attempts for e in episodes])
    )
    all_attempts = [a for e in episodes for a in e.grasp_attempts]
    if all_attempts:
        out["eval/grasp_attempt_success_rate"] = float(
            np.mean([a["engaged"] for a in all_attempts])
        )
    # Miss anatomy: which gate axis failed, and by how much (signed mean shows
    # a systematic bias — e.g. always in front — abs mean shows magnitude).
    misses = [a for a in all_attempts if not a["engaged"]]
    if misses:
        for reason in MISS_REASONS:
            out[f"eval/miss/{reason}_rate"] = float(
                np.mean([reason in a["miss_reasons"] for a in misses])
            )
        for axis in ("pinch", "finger", "height"):
            vals = np.array([a[axis] for a in misses], dtype=float)
            out[f"eval/miss/{axis}_offset_abs_mean"] = float(np.mean(np.abs(vals)))
            out[f"eval/miss/{axis}_offset_bias"] = float(np.mean(vals))

    # Release anatomy: with the kinematic lock every drop is a commanded
    # release, so premature releases *are* the drop failure mode.
    out["eval/premature_release_count_mean"] = float(
        np.mean([e.premature_release_count for e in episodes])
    )
    premature = [r for e in episodes for r in e.releases if r["premature"]]
    if premature:
        out["eval/premature_release_horiz_dist_mean"] = float(
            np.mean([r["horiz_dist_to_goal"] for r in premature])
        )
        out["eval/premature_release_height_mean"] = float(
            np.mean([r["height_above_spawn"] for r in premature])
        )
        out["eval/premature_release_hold_len_mean"] = float(
            np.mean([r["hold_len"] for r in premature])
        )

    out["eval/gripper_close_cycles_mean"] = float(
        np.mean([e.gripper_close_cycles for e in episodes])
    )
    deepest = [
        e.deepest_close_cmd_near_object
        for e in episodes
        if e.deepest_close_cmd_near_object is not None
    ]
    if deepest:
        out["eval/deepest_close_cmd_near_object_mean"] = float(np.mean(deepest))

    # Pressing the block into the table / shoving it around blind.
    out["eval/press/episode_rate"] = float(
        np.mean([e.press_steps > 0 for e in episodes])
    )
    out["eval/press/steps_mean"] = float(np.mean([e.press_steps for e in episodes]))
    pushed = np.array([e.pushed_dist_pre_grasp for e in episodes], dtype=float)
    out["eval/pushed_dist_pre_grasp_mean"] = float(np.mean(pushed))
    out["eval/pushed_dist_pre_grasp_p90"] = float(np.percentile(pushed, 90))

    return out


def failure_mode_counts(episodes: list[EpisodeMetrics]) -> dict[str, int]:
    """Episode count per failure-mode label (only labels that occurred)."""
    counts: dict[str, int] = {}
    for e in episodes:
        counts[e.failure_mode] = counts.get(e.failure_mode, 0) + 1
    return dict(sorted(counts.items(), key=lambda kv: -kv[1]))
