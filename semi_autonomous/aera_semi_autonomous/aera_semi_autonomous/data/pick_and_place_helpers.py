"""Shared helpers for pick-and-place scripts."""

import copy
import logging
from typing import Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from scipy.spatial.transform import Rotation

from aera_semi_autonomous.data.trajectory_perturbation import (
    RecoveryPerturbation,
    sample_wrong_approach_poses,
)

# Gripper jaw kinematics (see ar4_mk3.xml: gripper_jaw{1,2}_joint range="-0.014 0").
# Each jaw is a symmetric slide joint: qpos=0 -> fully closed, qpos=-0.014 -> fully open.
# Pinch gap between pads ≈ 2 * |qpos|, so to land on an object of half-width h
# the symmetric jaw qpos target is -h. A small positive preload pushes slightly
# past the surface so contact force is bounded and non-zero rather than zero.
GRIPPER_JAW_QPOS_MIN = -0.014
GRIPPER_JAW_QPOS_MAX = 0.0
DEFAULT_GRASP_PRELOAD = 0.0005  # 0.5 mm — just enough to keep the jaws visually touching the object; the kinematic lock prevents this preload from causing any penetration.

# get_object_pose reports z as 2*object_center (the block top surface). A
# graspable block (half-height <= 0.012) rests at top <= 0.024 m, so anything
# well above this means the object is still up in the gripper rather than on the
# table — used to detect a partial grasp that held through the lift.
_SETTLED_OBJECT_MAX_Z = 0.045


# --- Recovery / grasp-time failure injection (sim2real plan #1+#2) ----------
#
# Reproduce the two ways real grasps fail at grasp time so the policy learns to
# recover — and crucially, neither ever presses the object into the table
# (forcing the arm into a rigid object is a real-world disaster we must not
# reinforce):
#   - wrong_approach: line up over the wrong spot at HOVER height, then correct
#     laterally and descend cleanly. Mis-alignment lives only at hover, so the
#     gripper never touches the object while off-target.
#   - partial_grasp: a marginal contact-only grip (no kinematic lock) that
#     slips out from between the fingers as the arm lifts, then re-detect and
#     re-grasp.
# Shared by the bulk collector and the quick demo. All soft: a failure is logged
# and skipped so injecting recovery never discards an otherwise-good demo.
# Frames are captured by the interface's own per-step recording, so the caller
# only sets the granular prompt before invoking these.


def inject_wrong_approach(
    robot,
    object_pose: Pose,
    recovery: RecoveryPerturbation,
    logger: Optional[logging.Logger] = None,
) -> None:
    """Line up over the wrong spot at hover, then correct laterally — all above
    the object so the gripper never touches it. The caller's real grasp then
    descends cleanly from the corrected hover. Soft."""
    _log = logger or logging.getLogger(__name__)
    try:
        bad_hover, good_hover = sample_wrong_approach_poses(object_pose, recovery)
        robot.release_gripper()  # open, as a real approach would be
        robot.move_to(bad_hover)  # mis-aligned hover (above the object, no contact)
        robot.move_to(good_hover)  # correct laterally, still at hover
    except Exception as e:
        _log.warning(f"Wrong-approach injection raised {e}; skipping.")


def inject_partial_grasp(
    robot,
    env,
    object_pose: Pose,
    grasp_gripper_pos: float,
    recovery: RecoveryPerturbation,
    logger: Optional[logging.Logger] = None,
) -> Pose:
    """Marginally grasp so the object slips out under physics on lift,
    ``max_grasp_retries`` times, re-detecting between slips. Returns the latest
    object pose for the caller's real grasp. Soft: any failure ends early."""
    _log = logger or logging.getLogger(__name__)
    pose = object_pose
    gripper_pos = grasp_gripper_pos
    # Lower the jaw+block contact friction during the slip so a CENTRED grasp
    # (no gap, no top-edge weirdness) can't hold the block — it lifts a little
    # then slides out. MuJoCo takes the max of the two geoms' friction, so both
    # the jaws and the block must be lowered; the table-block friction is
    # untouched (table geom keeps its friction), so the block settles normally.
    fric_geoms = [
        env.model.geom("object0").id,
        env.model.geom("gripper_jaw1_contact").id,
        env.model.geom("gripper_jaw2_contact").id,
    ]
    saved_fric = {gi: env.model.geom_friction[gi].copy() for gi in fric_geoms}
    for _ in range(max(1, recovery.max_grasp_retries)):
        lift = float(np.random.uniform(*recovery.partial_grasp_lift_range))
        pause = (recovery.partial_grasp_pause_steps
                 if np.random.random() < recovery.partial_grasp_pause_prob else 0)
        for gi in fric_geoms:
            env.model.geom_friction[gi][0] = recovery.partial_grasp_slip_friction
        try:
            # Centred grip (no height offset): jaws straddle the full box so there
            # is no visual gap and the tips clear the table; the lowered friction
            # is what makes it slip.
            robot.grasp_and_slip(copy.deepcopy(pose), gripper_pos, lift, pause_steps=pause)
        except Exception as e:
            _log.warning(f"Partial-grasp injection raised {e}; stopping.")
            break
        finally:
            for gi in fric_geoms:
                env.model.geom_friction[gi] = saved_fric[gi]
        # The object slipped out and settled (maybe shifted) — re-detect for the
        # next attempt and for the caller's real grasp. If the marginal grip held
        # through the lift (object still aloft, not on the table), open to drop it
        # — only fires in that failure case, so a real slip needs no extra open.
        redetected = get_object_pose(env, _log)
        if redetected is not None and redetected.position.z > _SETTLED_OBJECT_MAX_Z:
            _log.info("Partial grasp held through lift; opening to drop the object.")
            robot.release_gripper()
            redetected = get_object_pose(env, _log)
        if redetected is not None:
            # Use the block's KNOWN resting height for the re-grasp DEPTH (xy from
            # the measurement). A just-slipped block can be read mid-bounce or
            # slightly tilted, giving a bad z that makes the re-grasp miss high or
            # dig into the table; a block always rests with its top at 2*half.
            box_id = env.model.geom("object0").id
            redetected.position.z = 2.0 * float(env.model.geom_size[box_id][2])
            pose = redetected
            gripper_pos = get_object_grasp_gripper_pos(env, logger=_log)
    return pose


def get_object_pose(env, logger: Optional[logging.Logger] = None) -> Optional[Pose]:
    """Get the current pose of object0 from the environment.

    Reads the object's joint state from MuJoCo, computes the grasp orientation
    (top-down, aligned with the object's yaw), and returns a Pose at the top
    surface of the object.

    Args:
        env: The Ar4Mk3PickAndPlaceEnv instance.
        logger: Optional logger for debug/error output.

    Returns:
        A Pose at the object's top surface, or None on failure.
    """
    _log = logger or logging.getLogger(__name__)
    try:
        object_qpos = env._utils.get_joint_qpos(env.model, env.data, "object0:joint")
        object_pos = object_qpos[:3]

        object_body_id = env.model.body("object0").id
        geom_id = -1
        for i in range(env.model.ngeom):
            if env.model.geom_bodyid[i] == object_body_id:
                geom_id = i
                break

        # For a box, size is [dx, dy, dz] (half-lengths).
        # If the object is longer along its y-axis (dy > dx), align the gripper
        # with the object's y-axis via an additional 90-degree rotation.
        additional_yaw = 0.0
        if geom_id != -1:
            geom_size = env.model.geom_size[geom_id]
            if geom_size[1] < geom_size[0]:
                additional_yaw = 90.0

        # MuJoCo quat is w,x,y,z. Scipy is x,y,z,w
        object_quat_wxyz = object_qpos[3:]
        object_quat_xyzw = np.array(
            [
                object_quat_wxyz[1],
                object_quat_wxyz[2],
                object_quat_wxyz[3],
                object_quat_wxyz[0],
            ]
        )
        object_rotation = Rotation.from_quat(object_quat_xyzw)
        object_yaw_deg = object_rotation.as_euler("xyz", degrees=True)[2]

        pose = Pose()
        pose.position = Point(
            x=float(object_pos[0]),
            y=float(object_pos[1]),
            z=2 * float(object_pos[2]),
        )

        _log.info(f"object_yaw_deg: {object_yaw_deg}")
        top_down_rot = Rotation.from_quat([0, 1, 0, 0])  # x, y, z, w
        z_rot = Rotation.from_euler("z", object_yaw_deg + additional_yaw, degrees=True)
        grasp_rot = z_rot * top_down_rot
        grasp_quat_xyzw = grasp_rot.as_quat()

        pose.orientation = Quaternion(
            x=grasp_quat_xyzw[0],
            y=grasp_quat_xyzw[1],
            z=grasp_quat_xyzw[2],
            w=grasp_quat_xyzw[3],
        )
        return pose
    except Exception as e:
        _log.error(f"Failed to get object pose: {e}")
        return None


def get_object_grasp_gripper_pos(
    env,
    preload: float = DEFAULT_GRASP_PRELOAD,
    logger: Optional[logging.Logger] = None,
) -> float:
    """Compute a gripper_pos target that lands the jaws on the object surface.

    Reads object0's box half-extents and picks the pinch dimension (the shorter
    of the two horizontal half-widths) — this matches the yaw-alignment logic
    in `get_object_pose`, which rotates the gripper so the jaws close across
    the narrower side.

    Falls back to a safe partially-closed target if the object can't be located.
    """
    _log = logger or logging.getLogger(__name__)
    try:
        object_body_id = env.model.body("object0").id
        geom_id = -1
        for i in range(env.model.ngeom):
            if env.model.geom_bodyid[i] == object_body_id:
                geom_id = i
                break
        if geom_id == -1:
            _log.warning("object0 geom not found; using fully-open grasp target.")
            return GRIPPER_JAW_QPOS_MIN

        geom_size = env.model.geom_size[geom_id]
        pinch_half_width = float(min(geom_size[0], geom_size[1]))
        target = -(pinch_half_width - preload)
        clamped = float(np.clip(target, GRIPPER_JAW_QPOS_MIN, GRIPPER_JAW_QPOS_MAX))
        if clamped != target:
            _log.warning(
                f"Object pinch half-width {pinch_half_width:.4f} is outside the "
                f"gripper's usable range; clamping gripper_pos to {clamped:.4f}."
            )
        return clamped
    except Exception as e:
        _log.error(f"Failed to compute grasp gripper_pos: {e}")
        return GRIPPER_JAW_QPOS_MIN
