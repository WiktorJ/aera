"""Shared helpers for pick-and-place scripts."""

import logging
from typing import Optional

import numpy as np
from geometry_msgs.msg import Point, Pose, Quaternion
from scipy.spatial.transform import Rotation

# Gripper jaw kinematics (see ar4_mk3.xml: gripper_jaw{1,2}_joint range="-0.014 0").
# Each jaw is a symmetric slide joint: qpos=0 -> fully closed, qpos=-0.014 -> fully open.
# Pinch gap between pads ≈ 2 * |qpos|, so to land on an object of half-width h
# the symmetric jaw qpos target is -h. A small positive preload pushes slightly
# past the surface so contact force is bounded and non-zero rather than zero.
GRIPPER_JAW_QPOS_MIN = -0.014
GRIPPER_JAW_QPOS_MAX = 0.0
DEFAULT_GRASP_PRELOAD = 0.0005  # 0.5 mm — just enough to keep the jaws visually touching the object; the kinematic lock prevents this preload from causing any penetration.


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
