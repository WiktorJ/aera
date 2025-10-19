import copy
import logging
import time
from typing import Any, Dict, Optional

import mujoco
import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation

from aera.autonomous.envs.ar4_mk3_base import Ar4Mk3Env
from aera_semi_autonomous.control.robot_interface import RobotInterface


# Constants
DEFAULT_CAMERA_CONFIG: Dict[str, Any] = {
    "width": 640,
    "height": 480,
    "fx": 525.0,
    "fy": 525.0,
    "cx": 320.0,
    "cy": 240.0,
}
"""Default camera configuration if none is provided."""

MOVE_TO_POS_TOLERANCE = 0.01  # 1cm
"""Position tolerance for move_to command."""

ABOVE_TARGET_OFFSET = 0.1  # 10cm
"""Offset to move above a target for grasping."""

GRIPPER_ACTION_STEPS = 50
"""Number of simulation steps to apply for gripper actions."""

HOME_QPOS_ERROR_TOLERANCE = 1e-3
"""Tolerance for joint position error when going home."""


GRIPPER_POS_TOLERANCE = 1e-3
"""Position tolerance for gripper actions."""


class Ar4Mk3RobotInterface(RobotInterface):
    """
    RobotInterface implementation for Ar4Mk3Env MuJoCo simulation.
    Provides a bridge between the semi-autonomous system and the RL environment.
    """

    def __init__(self, env: Ar4Mk3Env, camera_config: Optional[Dict[str, Any]] = None):
        self.env = env
        self.logger = logging.getLogger(__name__)

        # Camera configuration
        self.camera_config = camera_config or DEFAULT_CAMERA_CONFIG

        # Create camera intrinsics
        self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=self.camera_config["width"],
            height=self.camera_config["height"],
            fx=self.camera_config["fx"],
            fy=self.camera_config["fy"],
            cx=self.camera_config["cx"],
            cy=self.camera_config["cy"],
        )

        # Store latest images
        self._latest_rgb_image: Optional[np.ndarray] = None
        self._latest_depth_image: Optional[np.ndarray] = None

        # Home pose for the robot (will be set after environment initialization)
        self.home_pose = self._initialize_home_pose()
        self.joint_names = [f"joint_{i}" for i in range(1, 7)]
        self.actuator_names = [f"act{i}" for i in range(1, 7)]

    def _nullspace_method(
        self,
        jac_joints: np.ndarray,
        delta: np.ndarray,
        regularization_strength: float = 0.0,
    ) -> np.ndarray:
        """Calculates the joint velocities to achieve a specified end effector delta."""
        hess_approx = jac_joints.T.dot(jac_joints)
        joint_delta = jac_joints.T.dot(delta)
        if regularization_strength > 0:
            hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
            return np.linalg.solve(hess_approx, joint_delta)
        else:
            return np.linalg.lstsq(hess_approx, joint_delta, rcond=-1)[0]

    def _get_joint_indices(
        self,
        model: mujoco.MjModel,  # type: ignore
        joint_names: list[str],
        total_dims: int,
        addr_attr: str,
        free_joint_dims: int,
        ball_joint_dims: int,
    ) -> np.ndarray:
        """Helper to get DoF or qpos indices for a given list of joint names."""
        indices = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
            if joint_id == -1:
                raise ValueError(f'No joint named "{name}" found.')

            addr = getattr(model, addr_attr)[joint_id]
            joint_type = model.jnt_type[joint_id]

            if joint_type == mujoco.mjtJoint.mjJNT_FREE:  # type: ignore
                num_dims = free_joint_dims
            elif joint_type == mujoco.mjtJoint.mjJNT_BALL:  # type: ignore
                num_dims = ball_joint_dims
            elif joint_type in (
                mujoco.mjtJoint.mjJNT_SLIDE,  # type: ignore
                mujoco.mjtJoint.mjJNT_HINGE,  # type: ignore
            ):
                num_dims = 1
            else:
                num_dims = 0

            indices.extend(range(addr, addr + num_dims))
        return np.array(indices)

    def _get_dof_indices(
        self,
        model: mujoco.MjModel,  # type: ignore
        joint_names: list[str],
    ) -> np.ndarray:
        """Get the list of DoF indices for a given list of joint names."""
        return self._get_joint_indices(model, joint_names, model.nv, "jnt_dofadr", 6, 3)

    def _get_qpos_indices(
        self,
        model: mujoco.MjModel,  # type: ignore
        joint_names: list[str],
    ) -> np.ndarray:
        """Get the list of qpos indices for a given list of joint names."""
        return self._get_joint_indices(
            model, joint_names, model.nq, "jnt_qposadr", 7, 4
        )

    def _interpolate_gripper(self, target_gripper_qpos: np.ndarray) -> bool:
        """Move the gripper jaws to the target position, waiting for convergence."""
        try:
            # Set arm controls to current joint positions to hold it steady
            arm_qpos_indices = self._get_qpos_indices(self.env.model, self.joint_names)
            self.env.data.ctrl[:6] = self.env.data.qpos[arm_qpos_indices]
            gripper_joint_names = ["gripper_jaw1_joint", "gripper_jaw2_joint"]
            gripper_actuator_names = ["act8", "act9"]
            gripper_ctrl_indices = [
                mujoco.mj_name2id(self.env.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
                for name in gripper_actuator_names
            ]
            gripper_qpos_indices = self._get_qpos_indices(
                self.env.model, gripper_joint_names
            )
            self.logger.info(
                f"Current gripper position: {self.env.data.qpos[gripper_qpos_indices]}"
            )
            self.logger.info(f"Target gripper position: {target_gripper_qpos}")

            start_gripper_qpos = self.env.data.qpos[gripper_qpos_indices].copy()

            # Use a timeout to prevent infinite loops.
            # The loop allows for convergence check while interpolating the setpoint.
            max_steps = GRIPPER_ACTION_STEPS * 2

            for i in range(max_steps):
                current_gripper_qpos = self.env.data.qpos[gripper_qpos_indices]
                if (
                    np.linalg.norm(target_gripper_qpos - current_gripper_qpos)
                    < GRIPPER_POS_TOLERANCE
                ):
                    break  # Converged

                # Interpolate control setpoint for smooth motion over GRIPPER_ACTION_STEPS
                alpha = min(1.0, i / GRIPPER_ACTION_STEPS)
                interpolated_qpos = (
                    1 - alpha
                ) * start_gripper_qpos + alpha * target_gripper_qpos
                self.env.data.ctrl[gripper_ctrl_indices] = interpolated_qpos

                mujoco.mj_step(self.env.model, self.env.data)  # type: ignore
                self.env.render()

            # Verify final position
            final_gripper_qpos = self.env.data.qpos[gripper_qpos_indices]
            final_error = np.linalg.norm(target_gripper_qpos - final_gripper_qpos)
            self.logger.info(
                f"Final gripper position: {final_gripper_qpos} in {i} steps"
            )
            if final_error > GRIPPER_POS_TOLERANCE:
                self.logger.warning(
                    f"Gripper interpolation may not have reached target precisely. "
                    f"Final error: {final_error:.4f}"
                )
            return True
        except Exception as e:
            self.logger.error(f"Failed to interpolate gripper: {e}", exc_info=True)
            return False

    def _solve_ik_for_site_pose(
        self,
        site_name: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        tol: float = 1e-3,
        regularization_threshold: float = 0.01,
        regularization_strength: float = 1e-4,
        max_update_norm: float = 0.75,
        progress_thresh: float = 100.0,
        integration_dt: float = 0.1,
        pos_gain: float = 0.95,
        orientation_gain: float = 0.95,
        max_steps: int = 20000,
        inplace: bool = False,
        min_height: float = 0.01,
    ) -> bool:
        """Find joint positions that satisfy a target site position and/or rotation."""
        model = self.env.model
        if inplace:
            data = self.env.data
        else:
            data = mujoco.MjData(model)  # type: ignore
            data = copy.deepcopy(self.env.data)

        dtype = data.qpos.dtype
        err_norm = 0.0
        success = False
        steps = 0
        failure_reason = "Unknown"
        # Increased nullspace gain to encourage solutions closer to the home configuration,
        # which helps avoid undesirable solutions like the arm going through the floor.
        nullspace_gain = np.asarray([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])

        dof_indices = self._get_dof_indices(model, self.joint_names)
        qpos_indices = self._get_qpos_indices(model, self.joint_names)
        actuator_ids = np.array(
            [model.actuator(name).id for name in self.actuator_names]
        )
        home_joint_configuration = self.env.initial_qpos[qpos_indices]

        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)  # type: ignore
            for name in self.joint_names
        ]
        joint_limits = model.jnt_range[joint_ids]

        jac = np.empty((6, model.nv), dtype=dtype)
        err = np.empty(6, dtype=dtype)
        jac_pos, jac_rot = jac[:3], jac[3:]
        err_pos, err_rot = err[:3], err[3:]

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)  # type: ignore

        # Keep track of the previous site_is position to check for convergence AI!
        for steps in range(max_steps):
            self.logger.info(
                f"current position: {data.site_xpos[site_id]}, target_position: {target_pos}"
            )
            site_xpos = data.site_xpos[site_id]
            err_pos[:] = pos_gain * (target_pos - site_xpos) / integration_dt

            site_xmat = data.site_xmat[site_id].reshape(3, 3)
            site_quat = np.empty(4, dtype=dtype)
            mujoco.mju_mat2Quat(site_quat, site_xmat.flatten())  # type: ignore
            neg_site_quat = np.empty(4, dtype=dtype)
            mujoco.mju_negQuat(neg_site_quat, site_quat)  # type: ignore
            err_rot_quat = np.empty(4, dtype=dtype)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_quat)  # type: ignore
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1.0)  # type: ignore
            err_rot *= orientation_gain / integration_dt

            err_norm = np.linalg.norm(
                err_pos * integration_dt / pos_gain
            )  # + np.linalg.norm(err_rot * integration_dt)
            # self.logger.info(
            #     f"error_pos: {err_pos}, error_rot: {err_rot}, error_norm: {err_norm}"
            # )

            if err_norm < tol:
                success = True
                failure_reason = ""
                break

            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)  # type: ignore
            jac_joints = jac[:, dof_indices]

            # reg_strength = (
            #     regularization_strength if err_norm > regularization_threshold else 0.0
            # )
            # update_joints = self._nullspace_method(jac_joints, err, reg_strength)
            diag = regularization_strength * np.eye(6)
            update_joints = jac_joints.T @ np.linalg.solve(
                jac_joints @ jac_joints.T + diag, err
            )

            # Nullspace projection for redundancy resolution, pulling towards home config
            nullspace_projector = (
                np.eye(len(dof_indices)) - np.linalg.pinv(jac_joints) @ jac_joints
            )
            # nullspace_projector = np.eye(len(dof_indices))
            nullspace_term = nullspace_projector @ (
                nullspace_gain * (home_joint_configuration - data.qpos[qpos_indices])
            )
            # print(f"nullspace_term: {nullspace_term}")
            update_joints += nullspace_term
            update_norm = np.linalg.norm(update_joints)
            # update_norm = np.abs(update_joints).max()

            #     failure_reason = f"Update norm too small ({update_norm:.2e})"
            #     break

            # progress_criterion = err_norm / update_norm
            # if progress_criterion > progress_thresh:
            #     failure_reason = f"Progress criterion not met ({progress_criterion:.2f} > {progress_thresh:.2f})"
            #     break

            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            update_nv = np.zeros(model.nv, dtype=dtype)
            update_nv[dof_indices] = update_joints

            q = data.qpos.copy()

            # prev_qpos = data.qpos.copy()
            mujoco.mj_integratePos(model, q, update_nv, integration_dt)  # type: ignore

            # Enforce joint limits
            # data.qpos[qpos_indices] = np.clip(
            #     data.qpos[qpos_indices], joint_limits[:, 0], joint_limits[:, 1]
            # )

            q[qpos_indices] = np.clip(
                q[qpos_indices], joint_limits[:, 0], joint_limits[:, 1]
            )
            data.ctrl[actuator_ids] = q[qpos_indices]

            # Check for floor collision
            # mujoco.mj_fwdPosition(model, data)
            mujoco.mj_step(model, data)
            # new_site_xpos = data.site_xpos[site_id]
            # if new_site_xpos[2] < min_height:
            #     data.qpos[:] = prev_qpos
            #     failure_reason = f"IK step would move gripper below minimum height ({new_site_xpos[2]} < {min_height})"
            #     break

            self.env.render()
            # time.sleep(0.002)
        else:
            success = False
            failure_reason = f"Max steps ({max_steps}) reached"

        if not success:
            self.logger.warning(
                f"IK failed to converge. Error: {err_norm:.4f}. "
                f"Reason: {failure_reason}. "
                f"Target Pos: {target_pos}, Target Quat: {target_quat}. "
                f"Curent Pos: {data.site_xpos[site_id]}. "
                f"Steps: {steps}."
            )
            return False
        return True

    def _initialize_home_pose(self) -> Pose:
        """Initialize the home pose to the robot's actual initial position."""
        # Use the actual initial gripper position from the environment
        # This preserves the "right angle position" that the robot starts in
        home_pos = self.env.initial_gripper_xpos

        # Get the actual initial gripper orientation to preserve the starting pose
        grip_rot = self.env._utils.get_site_xmat(self.env.model, self.env.data, "grip")

        rotation = Rotation.from_matrix(grip_rot.reshape(3, 3))
        quat = rotation.as_quat()  # [x, y, z, w]

        home_pose = Pose()
        home_pose.position.x = float(home_pos[0])
        home_pose.position.y = float(home_pos[1])
        home_pose.position.z = float(home_pos[2])
        home_pose.orientation.x = float(quat[0])
        home_pose.orientation.y = float(quat[1])
        home_pose.orientation.z = float(quat[2])
        home_pose.orientation.w = float(quat[3])
        return home_pose

    def get_logger(self) -> logging.Logger:
        return self.logger

    def go_home(self) -> bool:
        """Move robot to home position by interpolating joint positions."""
        try:
            qpos_indices = self._get_qpos_indices(self.env.model, self.joint_names)
            target_qpos = self.env.initial_qpos[qpos_indices]
            current_qpos = self.env.data.qpos[qpos_indices].copy()

            if np.linalg.norm(target_qpos - current_qpos) < HOME_QPOS_ERROR_TOLERANCE:
                return True

            num_steps = 100  # for smooth movement

            for i in range(num_steps + 1):
                alpha = i / num_steps
                interpolated_qpos = (1 - alpha) * current_qpos + alpha * target_qpos
                self.env.data.qpos[qpos_indices] = interpolated_qpos
                mujoco.mj_forward(self.env.model, self.env.data)  # type: ignore
                self.env.render()
                time.sleep(0.01)

            # Verify final joint positions
            final_qpos = self.env.data.qpos[qpos_indices]
            qpos_error = np.linalg.norm(target_qpos - final_qpos)
            if qpos_error > HOME_QPOS_ERROR_TOLERANCE:
                self.logger.warning(
                    f"Go home may not have reached target precisely. Final qpos error: {qpos_error:.4f}"
                )

            return True

        except Exception as e:
            self.logger.error(f"An error occurred during go_home: {e}", exc_info=True)
            return False

    def move_to(self, pose: Pose) -> bool:
        """Move end-effector to specified pose using inverse kinematics."""
        try:
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            # Convert from ROS quaternion (x, y, z, w) to MuJoCo quaternion (w, x, y, z)
            target_quat = np.array(
                [
                    pose.orientation.w,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                ]
            )

            if not self._solve_ik_for_site_pose(
                site_name="grip",
                target_pos=target_pos,
                target_quat=target_quat,
                inplace=True,
            ):
                return False

            # Verify final position
            final_pose = self.get_end_effector_pose()
            if final_pose is None:
                return False

            final_pos = np.array(
                [final_pose.position.x, final_pose.position.y, final_pose.position.z]
            )
            pos_error = np.linalg.norm(target_pos - final_pos)

            # Check if we are close enough to the target
            if pos_error > MOVE_TO_POS_TOLERANCE:
                self.logger.warning(
                    f"Move to failed. Final position error: {pos_error:.4f}m"
                )
                return False

            return True
        except Exception as e:
            self.logger.error(f"Failed to move to pose: {e}", exc_info=True)
            return False

    def release_gripper(self) -> bool:
        """Release the gripper by interpolating to an open state."""
        try:
            self.logger.info("Releasing gripper")
            gripper_qpos_indices = np.arange(self.env.model.nq - 2, self.env.model.nq)
            # Fully open gripper
            target_gripper_qpos = np.array([-0.014, -0.014])

            if not self._interpolate_gripper(target_gripper_qpos):
                return False

            # Check if gripper is open (should have negative values when open)
            final_gripper_qpos = self.env.data.qpos[gripper_qpos_indices]
            if np.any(final_gripper_qpos >= -0.001):  # Should be negative for open
                self.logger.warning(
                    f"Gripper may not be fully open after release action. Final qpos: {final_gripper_qpos}"
                )

            return True
        except Exception as e:
            self.logger.error(f"Failed to release gripper: {e}", exc_info=True)
            return False

    def grasp_at(self, pose: Pose, gripper_pos: float) -> bool:
        """Move to a pose and grasp.

        Args:
            pose: Target pose for grasping
            gripper_pos: Gripper position in actual range [-0.014, 0] where -0.014 is fully open and 0 is fully closed
        """
        try:
            # Validate gripper_pos is in the correct range
            if not (-0.014 <= gripper_pos <= 0.0):
                self.logger.error(
                    f"Invalid gripper_pos {gripper_pos}. Must be in range [-0.014, 0]"
                )
                return False

            # 1. Open the gripper first
            if not self.release_gripper():
                self.logger.error("Grasp failed: could not open gripper.")
                return False
            self.logger.info("Opened gripper.")

            # 2. Move to a position slightly above the target
            above_pose = copy.deepcopy(pose)
            above_pose.position.z += ABOVE_TARGET_OFFSET
            if not self.move_to(above_pose):
                self.logger.error("Grasp failed: could not move above target.")
                return False
            self.logger.info("Moved above target.")

            # 3. Move to the grasp pose
            if not self.move_to(pose):
                self.logger.error("Grasp failed: could not move to target.")
                return False
            self.logger.info("Moved to target.")

            # 4. Close the gripper to the specified position
            target_gripper_qpos = np.array([gripper_pos, gripper_pos])

            if not self._interpolate_gripper(target_gripper_qpos):
                self.logger.error("Grasp failed: could not close gripper.")
                return False
            self.logger.info("Closed gripper.")

            # 5. Lift the object
            if not self.move_to(above_pose):
                self.logger.warning("Grasp succeeded, but failed to lift.")
            self.logger.info("Lifted object.")

            return True
        except Exception as e:
            self.logger.error(f"Failed to grasp at pose: {e}", exc_info=True)
            return False

    def release_at(self, pose: Pose) -> bool:
        """Move to a pose and release the gripper."""
        try:
            # 1. Move to a position slightly above the target
            above_pose = copy.deepcopy(pose)
            above_pose.position.z += ABOVE_TARGET_OFFSET
            if not self.move_to(above_pose):
                self.logger.error("Release at failed: could not move above target.")
                return False
            self.logger.info("Moved above target for release.")

            # 2. Move to the release pose
            if not self.move_to(pose):
                self.logger.error("Release at failed: could not move to target.")
                return False
            self.logger.info("Moved to release target.")

            # 3. Release the gripper
            if not self.release_gripper():
                self.logger.error("Release at failed: could not release gripper.")
                return False
            self.logger.info("Released gripper.")

            # 4. Move back up
            if not self.move_to(above_pose):
                self.logger.warning(
                    "Release succeeded, but failed to move up afterwards."
                )
            self.logger.info("Moved up after release.")

            return True
        except Exception as e:
            self.logger.error(f"Failed to release at pose: {e}", exc_info=True)
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
            render_result = self.env.render()
            self.env.render_mode = original_render_mode

            # Handle different return types from render()
            if render_result is not None and isinstance(render_result, np.ndarray):
                self._latest_rgb_image = render_result
                return render_result
            return self._latest_rgb_image

        except Exception as e:
            self.logger.error(f"Failed to get RGB image: {e}", exc_info=True)
            return None

    def get_latest_depth_image(self) -> Optional[np.ndarray]:
        """Get latest depth image from simulation."""
        try:
            # Temporarily set render mode to depth_array to get depth image
            original_render_mode = self.env.render_mode
            self.env.render_mode = "depth_array"
            render_result = self.env.render()
            self.env.render_mode = original_render_mode

            # Handle different return types from render()
            if render_result is not None and isinstance(render_result, np.ndarray):
                self._latest_depth_image = render_result
                return render_result
            return self._latest_depth_image

        except Exception as e:
            self.logger.error(f"Failed to get depth image: {e}", exc_info=True)
            return None

    def get_camera_intrinsics(self) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        """Get camera intrinsic parameters."""
        return self.camera_intrinsics

    def get_cam_to_base_transform(self) -> Optional[np.ndarray]:
        """Get camera to base frame transformation."""
        return None
