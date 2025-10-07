import logging
import time
from typing import Optional
import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation
import collections

from aera_semi_autonomous.control.robot_interface import RobotInterface
from aera.autonomous.envs.ar4_mk3_base import Ar4Mk3Env

from dm_control.mujoco.wrapper import mjbindings

mjlib = mjbindings.mjlib
enums = mjbindings.enums

# IK Result namedtuple
IKResult = collections.namedtuple("IKResult", ["qpos", "err_norm", "steps", "success"])


class Ar4Mk3RobotInterface(RobotInterface):
    """
    RobotInterface implementation for Ar4Mk3Env MuJoCo simulation.
    Provides a bridge between the semi-autonomous system and the RL environment.
    """

    def __init__(self, env: Ar4Mk3Env, camera_config: Optional[dict] = None):
        self.env = env
        self.logger = logging.getLogger(__name__)

        # Camera configuration
        self.camera_config = camera_config or {
            "width": 640,
            "height": 480,
            "fx": 525.0,
            "fy": 525.0,
            "cx": 320.0,
            "cy": 240.0,
        }

        # Create camera intrinsics
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=self.camera_config["width"],
            height=self.camera_config["height"],
            fx=self.camera_config["fx"],
            fy=self.camera_config["fy"],
            cx=self.camera_config["cx"],
            cy=self.camera_config["cy"],
        )

        # Camera to base transform (example values - adjust based on your setup)
        self.cam_to_base_transform = np.array(
            [
                [0.0, -1.0, 0.0, 0.3],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Store latest images
        self._latest_rgb_image = None
        self._latest_depth_image = None
        self._latest_rgb_msg = None

        # Home pose for the robot (will be set after environment initialization)
        self.home_pose = self._initialize_home_pose()

    def get_logger(self) -> logging.Logger:
        return self.logger

    def go_home(self) -> bool:
        """Move robot to home position and release gripper."""
        try:
            self._initialize_home_pose()

            if not self.move_to(self.home_pose):
                self.logger.error("Failed to move to home pose.")
                return False

            if not self.release_gripper():
                self.logger.warning("Failed to release gripper after moving home.")
                return False

            return True
        except Exception as e:
            self.logger.error(f"An error occurred during go_home: {e}", exc_info=True)
            return False

    def _nullspace_method(self, jac_joints, delta, regularization_strength=0.0):
        """Calculates the joint velocities to achieve a specified end effector delta."""
        hess_approx = jac_joints.T.dot(jac_joints)
        joint_delta = jac_joints.T.dot(delta)
        if regularization_strength > 0:
            # L2 regularization
            hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
            return np.linalg.solve(hess_approx, joint_delta)
        else:
            return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]

    def _get_dof_indices(self, model, joint_names):
        """Get the list of DoF indices for a given list of joint names."""
        if joint_names is None:
            return np.arange(model.nv)
        dof_indices = []
        for name in joint_names:
            joint_id = mjlib.mj_name2id(model.ptr, enums.mjtObj.mjOBJ_JOINT, name)
            if joint_id == -1:
                raise ValueError(f'No joint named "{name}" found.')
            dof_addr = model.jnt_dofadr[joint_id]
            joint_type = model.jnt_type[joint_id]
            if joint_type == enums.mjtJnt.mjJNT_FREE:
                ndof = 6
            elif joint_type == enums.mjtJnt.mjJNT_BALL:
                ndof = 3
            elif joint_type in (enums.mjtJnt.mjJNT_SLIDE, enums.mjtJnt.mjJNT_HINGE):
                ndof = 1
            else:
                ndof = 0
            dof_indices.extend(range(dof_addr, dof_addr + ndof))
        return np.array(dof_indices)

    def _solve_ik_for_site_pose(
        self,
        site_name,
        target_pos=None,
        target_quat=None,
        joint_names=None,
        tol=1e-14,
        rot_weight=1.0,
        regularization_threshold=0.1,
        regularization_strength=3e-2,
        max_update_norm=2.0,
        progress_thresh=20.0,
        max_steps=100,
        inplace=False,
    ):
        """Find joint positions that satisfy a target site position and/or rotation."""
        model = self.env.model
        if inplace:
            data = self.env.data
        else:
            data = mjlib.mj_makeData(model.ptr)
            mjlib.mj_copyData(data, model.ptr, self.env.data.ptr)

        dtype = data.qpos.dtype
        err_norm = 0.0
        success = False
        steps = 0

        if target_pos is not None and target_quat is not None:
            jac = np.empty((6, model.nv), dtype=dtype)
            err = np.empty(6, dtype=dtype)
            jac_pos, jac_rot = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        elif target_pos is not None:
            jac = np.empty((3, model.nv), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            jac_pos, jac_rot = jac, None
            err_pos, err_rot = err, None
        elif target_quat is not None:
            jac = np.empty((3, model.nv), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            jac_pos, jac_rot = None, jac
            err_pos, err_rot = None, err
        else:
            raise ValueError("At least one of `target_pos` or `target_quat` must be specified.")

        site_id = mjlib.mj_name2id(model.ptr, enums.mjtObj.mjOBJ_SITE, site_name)
        dof_indices = self._get_dof_indices(model, joint_names)
        jac_joints = jac[:, dof_indices]

        for steps in range(max_steps):
            mjlib.mj_fwdPosition(model.ptr, data.ptr)

            site_xpos = data.site_xpos[site_id]
            site_xmat = data.site_xmat[site_id].reshape(3, 3)
            site_quat = np.empty(4, dtype=dtype)
            mjlib.mju_mat2Quat(site_quat, site_xmat.flatten())

            err_norm = 0
            if target_pos is not None:
                err_pos[:] = target_pos - site_xpos
                err_norm += np.linalg.norm(err_pos)

            if target_quat is not None:
                neg_site_quat = np.empty(4, dtype=dtype)
                mjlib.mju_negQuat(neg_site_quat, site_quat)
                err_rot_quat = np.empty(4, dtype=dtype)
                mjlib.mju_mulQuat(err_rot_quat, target_quat, neg_site_quat)
                mjlib.mju_quat2Vel(err_rot, err_rot_quat, 1.0)
                err_norm += np.linalg.norm(err_rot) * rot_weight

            if err_norm < tol:
                success = True
                break

            mjlib.mj_jacSite(model.ptr, data.ptr, jac_pos, jac_rot, site_id)

            reg_strength = regularization_strength if err_norm > regularization_threshold else 0.0
            update_joints = self._nullspace_method(jac_joints, err, reg_strength)
            update_norm = np.linalg.norm(update_joints)

            if update_norm < 1e-6:
                break

            progress_criterion = err_norm / update_norm
            if progress_criterion > progress_thresh:
                break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            update_nv = np.zeros(model.nv, dtype=dtype)
            update_nv[dof_indices] = update_joints
            mjlib.mj_integratePos(model.ptr, data.qpos, update_nv, 1.0)
        else:
            success = False

        qpos = data.qpos.copy()
        if not inplace:
            mjlib.mj_deleteData(data)

        return IKResult(qpos=qpos, err_norm=err_norm, steps=steps + 1, success=success)

    def _apply_joint_positions(self, qpos: np.ndarray):
        """Teleports the robot to the given joint positions."""
        self.env.data.qpos[:] = qpos
        mjlib.mj_fwdPosition(self.env.model.ptr, self.env.data.ptr)

    def _initialize_home_pose(self):
        """Initialize the home pose to the robot's actual initial position."""
        # Use the actual initial gripper position from the environment
        # This preserves the "right angle position" that the robot starts in
        home_pos = self.env.initial_gripper_xpos

        print(f"Setting home to initial gripper position: {home_pos}")

        # Get the actual initial gripper orientation to preserve the starting pose
        grip_rot = self.env._utils.get_site_xmat(self.env.model, self.env.data, "grip")
        rotation = Rotation.from_matrix(grip_rot.reshape(3, 3))
        quat = rotation.as_quat()  # [x, y, z, w]

        self.home_pose = Pose()
        self.home_pose.position.x = float(home_pos[0])
        self.home_pose.position.y = float(home_pos[1])
        self.home_pose.position.z = float(home_pos[2])
        self.home_pose.orientation.x = float(quat[0])
        self.home_pose.orientation.y = float(quat[1])
        self.home_pose.orientation.z = float(quat[2])
        self.home_pose.orientation.w = float(quat[3])

    def move_to(self, pose: Pose) -> bool:
        """Move end-effector to specified pose using inverse kinematics."""
        try:
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            # Convert from ROS quaternion (x, y, z, w) to MuJoCo quaternion (w, x, y, z)
            target_quat = np.array(
                [pose.orientation.w, pose.orientation.x, pose.orientation.y, pose.orientation.z]
            )

            # Assuming standard AR4 joint names
            joint_names = [f"joint{i}" for i in range(1, 7)]

            ik_result = self._solve_ik_for_site_pose(
                site_name="grip",
                target_pos=target_pos,
                target_quat=target_quat,
                joint_names=joint_names,
                inplace=False,  # We want the target qpos, not to modify the state yet
            )

            if not ik_result.success:
                self.logger.warning(f"IK failed to converge. Error: {ik_result.err_norm:.4f}")
                return False

            self._apply_joint_positions(ik_result.qpos)

            # Verify final position
            final_pose = self.get_end_effector_pose()
            if final_pose is None:
                return False

            final_pos = np.array(
                [final_pose.position.x, final_pose.position.y, final_pose.position.z]
            )
            pos_error = np.linalg.norm(target_pos - final_pos)

            # Check if we are close enough to the target
            if pos_error > 0.01:  # 1cm tolerance
                self.logger.warning(
                    f"Move to failed. Final position error: {pos_error:.4f}m"
                )
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to move to pose: {e}", exc_info=True)
            return False

    def release_gripper(self) -> bool:
        return False

    def grasp_at(self, pose: Pose, gripper_pos: float) -> bool:
        return False

    def release_at(self, pose: Pose) -> bool:
        return False

    def get_end_effector_pose(self) -> Optional[Pose]:
        """Get current end-effector pose."""
        try:
            # Get gripper position from MuJoCo
            grip_pos = self.env._utils.get_site_xpos(
                self.env.model, self.env.data, "grip"
            )

            # Get gripper orientation (simplified - assumes fixed orientation)
            # In a real implementation, you'd extract this from the simulation
            grip_rot = self.env._utils.get_site_xmat(
                self.env.model, self.env.data, "grip"
            )
            rotation = Rotation.from_matrix(grip_rot.reshape(3, 3))
            quat = rotation.as_quat()  # [x, y, z, w]

            pose = Pose()
            pose.position.x = float(grip_pos[0])
            pose.position.y = float(grip_pos[1])
            pose.position.z = float(grip_pos[2])
            pose.orientation.x = float(quat[0])
            pose.orientation.y = float(quat[1])
            pose.orientation.z = float(quat[2])
            pose.orientation.w = float(quat[3])

            return pose

        except Exception as e:
            self.logger.error(f"Failed to get end-effector pose: {e}", exc_info=True)
            return None

    def get_latest_rgb_image(self) -> Optional[np.ndarray]:
        """Get latest RGB image from simulation."""
        try:
            # Temporarily set render mode to rgb_array to get image
            original_render_mode = self.env.render_mode
            self.env.render_mode = "rgb_array"
            rgb_image = self.env.render()
            self.env.render_mode = original_render_mode

            if rgb_image is not None:
                self._latest_rgb_image = rgb_image
                return rgb_image  # type: ignore
            return self._latest_rgb_image  # type: ignore

        except Exception as e:
            self.logger.error(f"Failed to get RGB image: {e}", exc_info=True)
            return None

    def get_latest_depth_image(self) -> Optional[np.ndarray]:
        """Get latest depth image from simulation."""
        try:
            # Temporarily set render mode to depth_array to get depth image
            original_render_mode = self.env.render_mode
            self.env.render_mode = "depth_array"
            depth_image = self.env.render()
            self.env.render_mode = original_render_mode

            if depth_image is not None:
                self._latest_depth_image = depth_image
                return depth_image
            return self._latest_depth_image

        except Exception as e:
            self.logger.error(f"Failed to get depth image: {e}", exc_info=True)
            return None

    def get_camera_intrinsics(self) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        """Get camera intrinsic parameters."""
        return self.camera_intrinsics

    def get_cam_to_base_transform(self) -> Optional[np.ndarray]:
        """Get camera to base frame transformation."""
        return self.cam_to_base_transform

    def get_last_rgb_msg(self) -> Optional[Image]:
        """Get last RGB image message (simulation doesn't use ROS messages)."""
        # For simulation, we don't have ROS messages
        # Return None or create a mock message if needed
        return self._latest_rgb_msg
