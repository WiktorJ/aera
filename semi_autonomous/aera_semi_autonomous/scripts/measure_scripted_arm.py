#!/usr/bin/env python3
"""Measure the scripted expert: timing, per-step deltas, grasp physics.

Runs the *real* collection code path (Ar4Mk3RobotInterface driving the same
go_home -> grasp_at -> release_at script as collect_trajectories) headless, with
no data collector attached, and traces per-mj-step state. Collection records one
frame per mj-step, so a trace here is exactly the raw stream a collected episode
would contain — which is what makes these numbers comparable to the dataset.

This is the harness behind the measurements in
training_journal/06.07.2026/next_run_changes.md, and the tool for verification
checks 1, 3 and 8 of that plan's pre-training gate.

Subcommands:
    timing      sim time / EEF path / average speed vs ik.integration_dt   (check 1)
    deltas      per-policy-step displacement distribution vs --skip        (checks 2, 3)
    dynrange    descent action as a fraction of the normalized output span (check 4)
    shove       how far the block is disturbed before the lock engages
    close-sweep commanded close depth -> pad contact / pinch / lock        (check 8)
    dwell       where the near-static frames come from

Examples:
    # does dt=0.005 actually land the 5-7 s / 12-15 cm/s target?
    python -m ...scripts.measure_scripted_arm timing --dt 0.005 --max-steps 3000

    # the "Measured delta vs skip" table, for the slow arm
    python -m ...scripts.measure_scripted_arm deltas --dt 0.005 --max-steps 3000 --skips 5 10 20

    # does a commanded close reach the block at all?
    python -m ...scripts.measure_scripted_arm close-sweep --dt 0.005 --max-steps 3000
"""

import argparse
import dataclasses
import json
import logging
import os
import sys

import numpy as np


def _ensure_ros_msgs() -> None:
    """Fall back to the vendored message stubs outside the ROS container.

    The interface imports geometry_msgs / sensor_msgs / std_msgs / cv_bridge at
    module scope; only the message *shapes* matter for driving the arm. Inside
    the container the real packages import first and this is a no-op.
    """
    try:
        import geometry_msgs.msg  # noqa: F401
    except ImportError:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "ros_msg_stubs"))


_ensure_ros_msgs()

from geometry_msgs.msg import Point, Pose, Quaternion  # noqa: E402
from scipy.spatial.transform import Rotation  # noqa: E402

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig, Q, T  # noqa: E402
from aera.autonomous.envs.ar4_mk3_pick_and_place import (  # noqa: E402
    Ar4Mk3PickAndPlaceEnv,
)
from aera_semi_autonomous.control.ar4_mk3_interface_config import (  # noqa: E402
    Ar4Mk3InterfaceConfig,
)
from aera_semi_autonomous.control.ar4_mk3_robot_interface import (  # noqa: E402
    Ar4Mk3RobotInterface,
)
from aera_semi_autonomous.data.pick_and_place_helpers import (  # noqa: E402
    get_object_grasp_gripper_pos,
    get_object_pose,
)

MODEL_PATH = "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml"
ARM_JOINTS = [f"joint_{i}" for i in range(1, 7)]
MJ_STEP_DT = 0.002  # sim seconds per recorded frame
# Gripper command band: the interface opens to -0.014, so a command above this
# means "closing". Used to locate the close event inside a trace.
_OPEN_CMD = -0.0135


@dataclasses.dataclass
class Trace:
    """Per-mj-step trace of one scripted pick-and-place (task segment only —
    the go_home prologue is trimmed, as it is in the training data)."""

    grip_pos: np.ndarray  # (n, 3) grip site, world
    arm_qpos: np.ndarray  # (n, 6) arm joint qpos
    grip_cmd: np.ndarray  # (n,) commanded jaw target (ctrl units)
    obj_pos: np.ndarray  # (n, 3) object0 body position
    obj_yaw: np.ndarray  # (n,) object0 yaw, radians
    held: np.ndarray  # (n,) bool, kinematic lock engaged
    sim_time_s: float
    grasp_start: int  # index where grasp_at began
    lock_engaged: bool

    @property
    def frames(self) -> int:
        return len(self.grip_pos)

    @property
    def eef_path_m(self) -> float:
        return float(np.linalg.norm(np.diff(self.grip_pos, axis=0), axis=1).sum())

    @property
    def avg_speed_cm_s(self) -> float:
        return 100.0 * self.eef_path_m / self.sim_time_s if self.sim_time_s else np.nan

    def close_index(self) -> int:
        """First frame where the jaws are commanded shut — the end of the
        descent, and the boundary the plan's 'descent' figures are measured
        against."""
        opened = self.grip_cmd <= _OPEN_CMD
        for i in range(1, len(self.grip_cmd)):
            if opened[i - 1] and not opened[i]:
                return i
        return len(self.grip_cmd) - 1


def _yaw_of(quat_wxyz) -> float:
    q = np.asarray(quat_wxyz, dtype=float)
    return float(Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz")[2])


def _build_env(seed: int, randomize_object_yaw: bool) -> Ar4Mk3PickAndPlaceEnv:
    # Mirrors collect_trajectories' env config (minus domain rand, which only
    # changes appearance/dynamics, not the scripted trajectory). TODO: swap for
    # the shared env-config factory once it lands, so this cannot drift.
    env = Ar4Mk3PickAndPlaceEnv(
        render_mode="rgb_array",
        config=Ar4Mk3EnvConfig(
            model_path=os.path.abspath(MODEL_PATH),
            reward_type="sparse",
            use_eef_control=False,
            translation=T,
            quaterion=Q,
            distance_multiplier=1.2,
            z_offset=0.3,
            use_geometric_lookat=True,
            randomize_object_yaw=randomize_object_yaw,
        ),
    )
    env.reset(seed=seed)
    return env


def run_episode(
    seed: int,
    integration_dt: float,
    max_steps: int,
    *,
    max_update_norm: float | None = None,
    gripper_pos: float | None = None,
    randomize_object_yaw: bool = False,
    place: bool = True,
) -> Trace | None:
    """Run one scripted pick(-and-place) and return its per-mj-step trace.

    ``gripper_pos`` overrides the scripted close target (pass 0.0 for the
    planned full close). ``place`` off stops after the grasp — enough for the
    grasp-physics subcommands and ~40% cheaper.

    Returns None if the script failed (IK budget exhausted, object not found),
    which is itself the signal that ``max_steps`` is too low for this dt.
    """
    env = _build_env(seed, randomize_object_yaw)
    base = Ar4Mk3InterfaceConfig()
    ik = dataclasses.replace(base.ik, integration_dt=integration_dt, max_steps=max_steps)
    if max_update_norm is not None:
        ik = dataclasses.replace(ik, max_update_norm=max_update_norm)
    robot = Ar4Mk3RobotInterface(env, config=dataclasses.replace(base, ik=ik))

    site_id = env.model.site("grip").id
    obj_body_id = env.model.body("object0").id
    jaw_act_ids = [env.model.actuator(n).id for n in ("act8", "act9")]
    arm_qpos_adr = [env.model.joint(n).qposadr[0] for n in ARM_JOINTS]

    rows: list[tuple] = []
    original_record_step = robot._record_step

    def traced():
        # Sample the same instant collection would: after the mj_step, before
        # the next IK iteration.
        rows.append(
            (
                env.data.site_xpos[site_id].copy(),
                env.data.qpos[arm_qpos_adr].copy(),
                float(env.data.ctrl[jaw_act_ids].mean()),
                env.data.xpos[obj_body_id].copy(),
                _yaw_of(env.data.xquat[obj_body_id]),
                bool(robot._grasp_lock.is_held) if robot._grasp_lock else False,
            )
        )
        return original_record_step()

    robot._record_step = traced

    try:
        if not robot.go_home():
            return None
        t_start = env.data.time
        task_start = len(rows)

        object_pose = get_object_pose(env, logging.getLogger(__name__))
        if object_pose is None:
            return None
        close_target = (
            gripper_pos
            if gripper_pos is not None
            else get_object_grasp_gripper_pos(env)
        )
        grasp_start = len(rows) - task_start
        if not robot.grasp_at(object_pose, close_target):
            return None

        if place:
            goal = env.goal
            target_pose = Pose()
            target_pose.position = Point(
                x=float(goal[0]),
                y=float(goal[1]),
                z=float(goal[2] + object_pose.position.z),
            )
            target_pose.orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
            if not robot.release_at(target_pose):
                return None

        sim_time = env.data.time - t_start
        lock_engaged = any(h for *_, h in rows[task_start:])
    finally:
        env.close()

    task = rows[task_start:]
    return Trace(
        grip_pos=np.array([r[0] for r in task]),
        arm_qpos=np.array([r[1] for r in task]),
        grip_cmd=np.array([r[2] for r in task]),
        obj_pos=np.array([r[3] for r in task]),
        obj_yaw=np.array([r[4] for r in task]),
        held=np.array([r[5] for r in task]),
        sim_time_s=float(sim_time),
        grasp_start=grasp_start,
        lock_engaged=lock_engaged,
    )


# --- subcommands ----------------------------------------------------------


def cmd_timing(args) -> None:
    """Check 1: episode sim time 5-7 s, average EEF speed 12-15 cm/s."""
    for dt in args.dt:
        for seed in args.seeds:
            tr = run_episode(seed, dt, args.max_steps, max_update_norm=args.max_update_norm)
            if tr is None:
                print(f"dt={dt} seed={seed}: FAILED (raise --max-steps?)", flush=True)
                continue
            out = {
                "dt": dt,
                "seed": seed,
                "sim_time_s": round(tr.sim_time_s, 3),
                "raw_frames": tr.frames,
                "eef_path_m": round(tr.eef_path_m, 3),
                "avg_speed_cm_s": round(tr.avg_speed_cm_s, 1),
                "lock_engaged": tr.lock_engaged,
            }
            print(json.dumps(out) if args.json else _fmt(out), flush=True)


def cmd_deltas(args) -> None:
    """Checks 2 + 3: per-step delta distribution and descent resolution at skip."""
    for seed in args.seeds:
        tr = run_episode(seed, args.dt, args.max_steps)
        if tr is None:
            print(f"dt={args.dt} seed={seed}: FAILED", flush=True)
            continue
        print(
            f"dt={args.dt} seed={seed} raw_frames={tr.frames} "
            f"sim_time={tr.frames * MJ_STEP_DT:.2f}s",
            flush=True,
        )
        for skip in args.skips:
            print("   " + (json.dumps(_delta_stats(tr, skip)) if args.json
                           else _fmt(_delta_stats(tr, skip))), flush=True)


def _delta_stats(tr: Trace, skip: int) -> dict:
    sub = tr.grip_pos[::skip]
    d = np.linalg.norm(np.diff(sub, axis=0), axis=1) * 1000.0  # mm
    jd = np.linalg.norm(np.diff(tr.arm_qpos[::skip], axis=0), axis=1)  # rad, L2 over 6
    # Descent = the last 12 policy steps before the jaws are commanded shut.
    descent = np.linalg.norm(
        np.diff(tr.grip_pos[: tr.close_index() + 1][::skip], axis=0), axis=1
    )[-12:] * 1000.0
    return {
        "skip": skip,
        "frames": len(sub),
        "eef_med_mm": round(float(np.median(d)), 2),
        "eef_p99_mm": round(float(np.percentile(d, 99)), 2),
        "eef_max_mm": round(float(d.max()), 2),
        "max_over_med": round(float(d.max() / np.median(d)), 1),
        "joint_med_rad": round(float(np.median(jd)), 4),
        "descent_med_mm": round(float(np.median(descent)), 2) if len(descent) else None,
        "descent_max_mm": round(float(descent.max()), 2) if len(descent) else None,
        "near_static_lt2mm": round(float((d < 2).mean()), 3),
        "lt5mm": round(float((d < 5).mean()), 3),
    }


def cmd_dynrange(args) -> None:
    """Check 4 (in-sim proxy): the descent action as a fraction of the
    normalized [-1,+1] output span.

    pi0.5 quantile-normalizes actions ([q01, q99] -> [-1, +1]), so absolute
    delta size is normalized away and this ratio is what governs how much of the
    model's output resolution the grasp descent actually uses. Quantiles here
    come from the single traced episode; the authoritative check uses the real
    dataset's meta/stats.json (see measure_action_dynrange.py).
    """
    for seed in args.seeds:
        tr = run_episode(seed, args.dt, args.max_steps)
        if tr is None:
            print(f"dt={args.dt} seed={seed}: FAILED", flush=True)
            continue
        for skip in args.skips:
            deltas = np.diff(tr.arm_qpos[::skip], axis=0)
            span = np.percentile(deltas, 99, axis=0) - np.percentile(deltas, 1, axis=0)
            descent = np.diff(
                tr.arm_qpos[: tr.close_index() + 1][::skip], axis=0
            )[-12:]
            worst_all = np.abs(deltas) * 2.0 / span
            worst_descent = np.abs(descent) * 2.0 / span
            out = {
                "dt": args.dt,
                "seed": seed,
                "skip": skip,
                "frames": len(deltas) + 1,
                "all_frames_pct_of_span": round(
                    float(np.median(worst_all.max(axis=1)) * 100), 1
                ),
                "descent_pct_of_span": round(
                    float(np.median(worst_descent.max(axis=1)) * 100), 1
                ),
            }
            print(json.dumps(out) if args.json else _fmt(out), flush=True)


def cmd_shove(args) -> None:
    """How far the block is disturbed between the start of grasp_at and the
    moment the lock engages — the only window where it is under real contact
    physics (afterwards its pose is pinned by the weld)."""
    for dt in args.dt:
        for seed in args.seeds:
            tr = run_episode(
                seed,
                dt,
                args.max_steps,
                gripper_pos=0.0 if args.full_close else None,
                randomize_object_yaw=args.randomize_object_yaw,
                place=False,
            )
            if tr is None:
                print(f"dt={dt} seed={seed}: FAILED", flush=True)
                continue
            engaged = np.flatnonzero(tr.held[tr.grasp_start:])
            end = tr.grasp_start + (int(engaged[0]) if len(engaged) else len(tr.held) - 1 - tr.grasp_start)
            window = slice(tr.grasp_start, end + 1)
            p0, y0 = tr.obj_pos[tr.grasp_start], tr.obj_yaw[tr.grasp_start]
            shove = np.linalg.norm(tr.obj_pos[window, :2] - p0[:2], axis=1) * 1000.0
            dyaw = np.degrees(
                np.abs(np.arctan2(np.sin(tr.obj_yaw[window] - y0),
                                  np.cos(tr.obj_yaw[window] - y0)))
            )
            z = tr.obj_pos[window, 2]
            out = {
                "dt": dt,
                "seed": seed,
                "window_frames": int(shove.size),
                "lock_engaged": bool(len(engaged)),
                "max_shove_mm": round(float(shove.max()), 2),
                "shove_at_engage_mm": round(float(shove[-1]), 2),
                "max_yaw_deg": round(float(dyaw.max()), 2),
                "z_dip_mm": round(float((z[0] - z.min()) * 1000), 2),
                "z_rise_mm": round(float((z.max() - z[0]) * 1000), 2),
            }
            print(json.dumps(out) if args.json else _fmt(out), flush=True)


def cmd_close_sweep(args) -> None:
    """Check 8: does a commanded close actually reach the block?

    For each commanded jaw target, reports the final jaw qpos, how many
    pad<->object contacts exist, whether the lock's pinch gate is satisfied, and
    whether the grasp locked. The engagement threshold this finds is the number
    the close-depth model (jaw_geometry / metrics' close_shallow) must agree
    with — measured, a 24 mm block needs a command >= -11.0 mm while the old
    formula asked for -11.5 mm and never touched the block.
    """
    for seed in args.seeds:
        for cmd_mm in args.commands_mm:
            res = _close_probe(seed, args.dt, args.max_steps, cmd_mm / 1000.0)
            if res is None:
                print(f"seed={seed} cmd={cmd_mm}mm: FAILED", flush=True)
                continue
            res.update(seed=seed, cmd_mm=cmd_mm)
            print(json.dumps(res) if args.json else _fmt(res), flush=True)


def _close_probe(seed: int, dt: float, max_steps: int, close_target: float) -> dict | None:
    """One grasp at a fixed commanded close depth, inspected at the instant the
    jaws have finished closing and BEFORE the lift.

    The snapshot has to happen there: a command that never reaches the block
    leaves it on the table, so after the lift the jaws are empty and every depth
    would read "no contact". grasp_at calls _engage_kinematic_grasp between the
    close and the lift, so patching that is exactly the right hook.
    """
    env = _build_env(seed, randomize_object_yaw=False)
    base = Ar4Mk3InterfaceConfig()
    robot = Ar4Mk3RobotInterface(
        env,
        config=dataclasses.replace(
            base, ik=dataclasses.replace(base.ik, integration_dt=dt, max_steps=max_steps)
        ),
    )
    obj_geom_id = env.model.geom("object0").id
    pad_ids = {
        env.model.geom(n).id for n in ("gripper_jaw1_contact", "gripper_jaw2_contact")
    }
    jaw_adr = [
        env.model.joint(n).qposadr[0]
        for n in ("gripper_jaw1_joint", "gripper_jaw2_joint")
    ]
    snapshot: dict = {}
    original_engage = robot._engage_kinematic_grasp

    def probed_engage(*a, **kw):
        pad_contacts = sum(
            1
            for i in range(env.data.ncon)
            if obj_geom_id in {env.data.contact[i].geom1, env.data.contact[i].geom2}
            and {env.data.contact[i].geom1, env.data.contact[i].geom2} & pad_ids
        )
        try:
            pinching = bool(robot._grasp_lock.jaws_pinching("object0"))
        except ValueError:  # model without pad geoms
            pinching = False
        snapshot.update(
            final_jaw_mm=round(float(env.data.qpos[jaw_adr].mean()) * 1000, 3),
            pad_contacts=pad_contacts,
            pinching=pinching,
        )
        result = original_engage(*a, **kw)
        snapshot["lock"] = bool(robot._grasp_lock.is_held)
        return result

    robot._engage_kinematic_grasp = probed_engage

    try:
        if not robot.go_home():
            return None
        pose = get_object_pose(env, logging.getLogger(__name__))
        if pose is None:
            return None
        half_width = float(min(env.model.geom_size[obj_geom_id][:2]))
        ok = robot.grasp_at(pose, close_target)
        if not snapshot:  # grasp_at bailed before the close
            return None
        return {"block_mm": round(half_width * 2000, 1), **snapshot, "script_ok": bool(ok)}
    finally:
        env.close()


def cmd_dwell(args) -> None:
    """Where the near-static frames come from: arm parked during a gripper ramp
    vs parked with the gripper idle (IK convergence tails / settles)."""
    for seed in args.seeds:
        tr = run_episode(seed, args.dt, args.max_steps)
        if tr is None:
            print(f"dt={args.dt} seed={seed}: FAILED", flush=True)
            continue
        step_mm = np.linalg.norm(np.diff(tr.grip_pos, axis=0), axis=1) * 1000.0
        ramping = np.abs(np.diff(tr.grip_cmd)) > 1e-9
        parked = step_mm < 0.05  # < 0.05 mm per 2 ms = < 2.5 cm/s
        out = {
            "dt": args.dt,
            "seed": seed,
            "raw_frames": tr.frames,
            "parked": round(float(parked.mean()), 3),
            "gripper_ramping": round(float(ramping.mean()), 3),
            "parked_and_ramping": round(float((parked & ramping).mean()), 3),
            "parked_gripper_idle": round(float((parked & ~ramping).mean()), 3),
        }
        print(json.dumps(out) if args.json else _fmt(out), flush=True)


def _fmt(d: dict) -> str:
    return "  ".join(f"{k}={v}" for k, v in d.items())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--verbose", action="store_true", default=False,
                        help="show the interface's own INFO logs (IK failures etc.)")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p, *, dt_default=0.005, multi_dt=False):
        if multi_dt:
            p.add_argument("--dt", type=float, nargs="+", default=[dt_default],
                           help="ik.integration_dt value(s) to measure")
        else:
            p.add_argument("--dt", type=float, default=dt_default,
                           help="ik.integration_dt")
        p.add_argument("--max-steps", type=int, default=3000, help="ik.max_steps")
        p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
        p.add_argument("--json", action="store_true", default=False,
                       help="emit one JSON object per line (for the verification gate)")
        return p

    p = common(sub.add_parser("timing", help="sim time / EEF speed (check 1)"), multi_dt=True)
    p.add_argument("--max-update-norm", type=float, default=None)
    p.set_defaults(func=cmd_timing)

    p = common(sub.add_parser("deltas", help="delta distribution vs skip (checks 2, 3)"))
    p.add_argument("--skips", type=int, nargs="+", default=[10])
    p.set_defaults(func=cmd_deltas)

    p = common(sub.add_parser("dynrange", help="descent action as %% of output span (check 4)"))
    p.add_argument("--skips", type=int, nargs="+", default=[10])
    p.set_defaults(func=cmd_dynrange)

    p = common(sub.add_parser("shove", help="block disturbance before the lock engages"), multi_dt=True)
    p.add_argument("--full-close", action="store_true", default=False,
                   help="command 0 instead of the computed preload target")
    p.add_argument("--randomize-object-yaw", action="store_true", default=False)
    p.set_defaults(func=cmd_shove)

    p = common(sub.add_parser("close-sweep", help="close depth -> contact / pinch / lock (check 8)"))
    p.add_argument("--commands-mm", type=float, nargs="+",
                   default=[-12.0, -11.6, -11.5, -11.0, -10.5, 0.0],
                   help="commanded jaw targets in mm (0 = full close)")
    p.set_defaults(func=cmd_close_sweep)

    p = common(sub.add_parser("dwell", help="near-static frame attribution"))
    p.set_defaults(func=cmd_dwell)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.ERROR)
    args.func(args)


if __name__ == "__main__":
    main()
