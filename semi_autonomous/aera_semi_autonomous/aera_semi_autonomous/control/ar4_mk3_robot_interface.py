import copy
import logging
import time
from typing import Any, Dict, Optional

import mujoco
import numpy as np
import open3d as o3d
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Header

from aera.autonomous.envs.ar4_mk3_base import Ar4Mk3Env
from aera_semi_autonomous.control.ar4_mk3_interface_config import (
    Ar4Mk3InterfaceConfig,
)
from aera_semi_autonomous.control.robot_interface import RobotInterface
from aera_semi_autonomous.data.trajectory_data_collector import TrajectoryDataCollector


class Ar4Mk3RobotInterface(RobotInterface):
    """
    RobotInterface implementation for Ar4Mk3Env MuJoCo simulation.
    Provides a bridge between the semi-autonomous system and the RL environment.
    """

    def __init__(self, env: Ar4Mk3Env, config: Ar4Mk3InterfaceConfig):
        self.env = env
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.data_collector: Optional[TrajectoryDataCollector] = None
        self.cv_bridge = CvBridge()

        # Camera configuration

        # Store latest images
        self._latest_rgb_image: Optional[Dict[str, np.ndarray]] = None
        self._latest_depth_image: Optional[Dict[str, np.ndarray]] = None

        # Home pose for the robot (will be set after environment initialization)
        self.home_pose = self._initialize_home_pose()
        self.joint_names = [f"joint_{i}" for i in range(1, 7)]
        self.actuator_names = [f"act{i}" for i in range(1, 7)]

    def set_data_collector(self, data_collector: Optional[TrajectoryDataCollector]):
        """Sets the data collector for recording trajectories."""
        self.data_collector = data_collector

    def _create_joint_state_msg(self, now: float) -> JointState:
        """Creates a JointState message from the current simulation state."""
        msg = JointState()
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        # Create a mock stamp object that has sec and nanosec attributes
        stamp = type("stamp", (), {"sec": sec, "nanosec": nanosec})()
        msg.header = Header(stamp=stamp)

        # Arm joints
        arm_joint_names = [f"joint_{i}" for i in range(1, 7)]
        qpos_indices = self._get_qpos_indices(self.env.model, arm_joint_names)
        dof_indices = self._get_dof_indices(self.env.model, arm_joint_names)
        msg.name.extend(arm_joint_names)
        msg.position.extend(self.env.data.qpos[qpos_indices])
        msg.velocity.extend(self.env.data.qvel[dof_indices])

        # Gripper joints
        gripper_joint_names = ["gripper_jaw1_joint", "gripper_jaw2_joint"]
        qpos_indices = self._get_qpos_indices(self.env.model, gripper_joint_names)
        dof_indices = self._get_dof_indices(self.env.model, gripper_joint_names)
        msg.name.extend(gripper_joint_names)
        msg.position.extend(self.env.data.qpos[qpos_indices])
        msg.velocity.extend(self.env.data.qvel[dof_indices])

        return msg

    def _create_image_msg(
        self, image_array: np.ndarray, encoding: str, now: float
    ) -> Image:
        """Creates an Image message from a numpy array."""
        sec = int(now)
        nanosec = int((now - sec) * 1e9)
        # Create a mock stamp object that has sec and nanosec attributes
        stamp = type("stamp", (), {"sec": sec, "nanosec": nanosec})()
        header = Header(stamp=stamp)

        # The cv_bridge in trajectory_data_collector expects bgr8 for rgb
        if encoding == "rgb8":
            image_array = image_array[..., ::-1]  # RGB to BGR
            encoding = "bgr8"

        img_msg = self.cv_bridge.cv2_to_imgmsg(image_array, encoding=encoding)
        img_msg.header = header
        return img_msg

    def _record_step(self):
        """Records a single step of simulation data if a data collector is set."""
        if self.data_collector:
            now = time.time()
            joint_state_msg = self._create_joint_state_msg(now)
            self.data_collector.record_joint_state(joint_state_msg)

            ros_timestamp = (
                joint_state_msg.header.stamp.sec
                + joint_state_msg.header.stamp.nanosec * 1e-9
            )

            rgb_imgs = self.get_latest_rgb_image()
            if rgb_imgs is not None:
                for cam_name, rgb_img in rgb_imgs.items():
                    # TODO: The data collector is not aware of camera names.
                    # This will be fixed in a future update.
                    rgb_msg = self._create_image_msg(rgb_img, "rgb8", now)
                    self.data_collector.record_rgb_image(rgb_msg)

            depth_imgs = self.get_latest_depth_image()
            if depth_imgs is not None:
                for cam_name, depth_img in depth_imgs.items():
                    # TODO: The data collector is not aware of camera names.
                    # This will be fixed in a future update.
                    depth_msg = self._create_image_msg(depth_img, "32FC1", now)
                    self.data_collector.record_depth_image(depth_msg)

            pose = self.get_end_effector_pose()
            if pose:
                self.data_collector.record_pose(pose, ros_timestamp)

    def _nullspace_method(
        self,
        jac_joints: np.ndarray,
        delta: np.ndarray,
        regularization_strength: float = 0.0,
    ) -> np.ndarray:
        """Calculates the joint velocities to achieve a specified end effector delta."""
        hess_approx = jac_joints.T @ jac_joints
        joint_delta = jac_joints.T @ delta
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
            start_gripper_qpos = self.env.data.qpos[gripper_qpos_indices].copy()

            # Use a timeout to prevent infinite loops.
            # The loop allows for convergence check while interpolating the setpoint.
            max_steps = self.config.gripper_action_steps * 2

            for i in range(max_steps):
                current_gripper_qpos = self.env.data.qpos[gripper_qpos_indices]
                if (
                    np.linalg.norm(target_gripper_qpos - current_gripper_qpos)
                    < self.config.gripper_pos_tolerance
                ):
                    break  # Converged

                # Interpolate control setpoint for smooth motion over GRIPPER_ACTION_STEPS
                alpha = min(1.0, i / self.config.gripper_action_steps)
                interpolated_qpos = (
                    1 - alpha
                ) * start_gripper_qpos + alpha * target_gripper_qpos
                self.env.data.ctrl[gripper_ctrl_indices] = interpolated_qpos

                mujoco.mj_step(self.env.model, self.env.data)  # type: ignore
                self._record_step()
                if self.config.render_steps:
                    self.env.render()

            # Verify final position
            final_gripper_qpos = self.env.data.qpos[gripper_qpos_indices]
            final_error = np.linalg.norm(target_gripper_qpos - final_gripper_qpos)
            if final_error > self.config.gripper_pos_tolerance:
                self.logger.info(
                    f"Gripper interpolation may not have reached target."
                    f"Final error: {final_error:.4f}"
                )
                # TODO: Because we are closing on giving the gripper 0.0 pos, this will always happen if there is object in the gripper.
                #   Eventually we should pass the object dims and then we can return false from here.
                # return False
            return True
        except Exception as e:
            self.logger.error(f"Failed to interpolate gripper: {e}", exc_info=True)
            return False

    def _solve_ik_for_site_pose(
        self,
        site_name: str,
        target_pos: np.ndarray,
        target_quat: np.ndarray,
        inplace: bool = False,
        tol: Optional[float] = None,
    ) -> bool:
        """Find joint positions that satisfy a target site position and/or rotation."""
        model = self.env.model
        if inplace:
            data = self.env.data
        else:
            data = mujoco.MjData(model)  # type: ignore
            data = copy.deepcopy(self.env.data)

        ik_tol = tol if tol is not None else self.config.ik_tolerance

        dtype = data.qpos.dtype
        err_norm = 0.0
        success = False
        steps = 0
        failure_reason = "Unknown"
        joints_update_scaling = np.asarray(self.config.ik_joints_update_scaling)

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
        previous_site_xpos = np.full_like(data.site_xpos[site_id], np.inf)
        for steps in range(self.config.ik_max_steps):
            site_xpos = data.site_xpos[site_id]

            err_pos[:] = (
                self.config.ik_pos_gain
                * (target_pos - site_xpos)
                / self.config.ik_integration_dt
            )

            site_xmat = data.site_xmat[site_id].reshape(3, 3)
            site_quat = np.empty(4, dtype=dtype)
            mujoco.mju_mat2Quat(site_quat, site_xmat.flatten())  # type: ignore
            neg_site_quat = np.empty(4, dtype=dtype)
            mujoco.mju_negQuat(neg_site_quat, site_quat)  # type: ignore
            err_rot_quat = np.empty(4, dtype=dtype)
            mujoco.mju_mulQuat(err_rot_quat, target_quat, neg_site_quat)  # type: ignore
            mujoco.mju_quat2Vel(err_rot, err_rot_quat, 1.0)  # type: ignore
            err_norm = np.linalg.norm(target_pos - site_xpos)
            err_rot *= self.config.ik_orientation_gain / self.config.ik_integration_dt

            if self.config.ik_include_rotation_in_target_error_measure:
                err_norm += np.linalg.norm(err_rot)

            if err_norm < ik_tol:
                success = True
                failure_reason = ""
                break

            mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)  # type: ignore
            jac_joints = jac[:, dof_indices]

            reg_strength = (
                self.config.ik_regularization_strength
                if err_norm > self.config.ik_regularization_threshold
                else 0.0
            )
            update_joints = self._nullspace_method(jac_joints, err, reg_strength)
            update_joints *= joints_update_scaling
            update_norm = np.linalg.norm(update_joints)

            if update_norm > self.config.ik_max_update_norm:
                update_joints *= self.config.ik_max_update_norm / update_norm

            update_nv = np.zeros(model.nv, dtype=dtype)
            update_nv[dof_indices] = update_joints

            q = data.qpos.copy()

            # prev_qpos = data.qpos.copy()
            mujoco.mj_integratePos(model, q, update_nv, self.config.ik_integration_dt)  # type: ignore

            # Enforce joint limits
            q[qpos_indices] = np.clip(
                q[qpos_indices], joint_limits[:, 0], joint_limits[:, 1]
            )
            if np.linalg.norm(site_xpos - previous_site_xpos) < 1e-7:
                success = False
                failure_reason = "IK step failed to converge"
                break

            data.ctrl[actuator_ids] = q[qpos_indices]

            previous_site_xpos = site_xpos.copy()
            mujoco.mj_step(model, data)  # type: ignore
            self._record_step()

            new_site_xpos = data.site_xpos[site_id]
            if new_site_xpos[2] < self.config.ik_min_height:
                success = False
                failure_reason = f"IK step moved gripper below minimum height ({new_site_xpos[2]} < {self.config.ik_min_height})"
                break

            if self.config.render_steps:
                self.env.render()
        else:
            success = False
            failure_reason = f"Max steps ({self.config.ik_max_steps}) reached"

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

    def _format_pose(self, pose: Pose) -> str:
        """Formats a Pose object into a readable string."""
        pos = pose.position
        orient = pose.orientation
        return (
            f"Position(x={pos.x:.3f}, y={pos.y:.3f}, z={pos.z:.3f}), "
            f"Orientation(x={orient.x:.3f}, y={orient.y:.3f}, z={orient.z:.3f}, w={orient.w:.3f})"
        )

    def get_logger(self) -> logging.Logger:
        return self.logger

    def go_home(self) -> bool:
        """Move robot to home position by interpolating joint positions."""
        try:
            self.logger.info("Going home.")
            qpos_indices = self._get_qpos_indices(self.env.model, self.joint_names)
            target_qpos = self.env.initial_qpos[qpos_indices]
            start_qpos = self.env.data.qpos[qpos_indices].copy()

            if (
                np.linalg.norm(target_qpos - start_qpos)
                < self.config.home_qpos_error_tolerance
            ):
                return True

            actuator_ids = np.array(
                [self.env.model.actuator(name).id for name in self.actuator_names]
            )

            num_steps = self.config.go_home_interpolation_steps  # for smooth movement
            max_steps = num_steps * 2  # timeout

            for i in range(max_steps):
                current_qpos = self.env.data.qpos[qpos_indices]
                if (
                    np.linalg.norm(target_qpos - current_qpos)
                    < self.config.home_qpos_error_tolerance
                ):
                    break  # Converged

                # Interpolate control setpoint for smooth motion
                alpha = min(1.0, i / num_steps)
                interpolated_qpos = (1 - alpha) * start_qpos + alpha * target_qpos
                self.env.data.ctrl[actuator_ids] = interpolated_qpos
                mujoco.mj_step(self.env.model, self.env.data)  # type: ignore
                self._record_step()
                if self.config.render_steps:
                    self.env.render()

            # Verify final joint positions
            final_qpos = self.env.data.qpos[qpos_indices]
            qpos_error = np.linalg.norm(target_qpos - final_qpos)
            if qpos_error > self.config.home_qpos_error_tolerance:
                self.logger.warning(
                    f"Go home may not have reached target precisely. Final qpos error: {qpos_error:.4f}"
                )
                return False

            return True

        except Exception as e:
            self.logger.error(f"An error occurred during go_home: {e}", exc_info=True)
            return False

    def move_to(self, pose: Pose) -> bool:
        """Move end-effector to specified pose using inverse kinematics."""
        try:
            self.logger.info(f"Moving to pose: {self._format_pose(pose)}")
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
                tol=self.config.move_to_pos_tolerance,
            ):
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
            self.logger.info(f"Grasping at pose: {self._format_pose(pose)}")
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

            # 2. Move to a position slightly above the target
            above_pose = copy.deepcopy(pose)
            above_pose.position.z += self.config.above_target_offset
            if not self.move_to(above_pose):
                self.logger.error("Grasp failed: could not move above target.")
                return False

            # 3. Move to the grasp pose
            if not self.move_to(pose):
                self.logger.error("Grasp failed: could not move to target.")
                return False

            # 4. Close the gripper to the specified position
            target_gripper_qpos = np.array([gripper_pos, gripper_pos])

            if not self._interpolate_gripper(target_gripper_qpos):
                self.logger.error("Grasp failed: could not close gripper.")
                return False

            # 5. Lift the object
            if not self.move_to(above_pose):
                self.logger.warning("Grasp succeeded, but failed to lift.")

            return True
        except Exception as e:
            self.logger.error(f"Failed to grasp at pose: {e}", exc_info=True)
            return False

    def release_at(self, pose: Pose) -> bool:
        """Move to a pose and release the gripper."""
        try:
            self.logger.info(f"Releasing at pose: {self._format_pose(pose)}")
            # 1. Move to a position slightly above the target
            above_pose = copy.deepcopy(pose)
            above_pose.position.z += self.config.above_target_offset
            if not self.move_to(above_pose):
                self.logger.error("Release at failed: could not move above target.")
                return False

            # 2. Move to the release pose
            if not self.move_to(pose):
                self.logger.error("Release at failed: could not move to target.")
                return False

            # 3. Release the gripper
            if not self.release_gripper():
                self.logger.error("Release at failed: could not release gripper.")
                return False

            # 4. Move back up
            if not self.move_to(above_pose):
                self.logger.warning(
                    "Release succeeded, but failed to move up afterwards."
                )
                return False

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

    def get_latest_rgb_image(self) -> Optional[Dict[str, np.ndarray]]:
        """Get latest RGB images from all cameras in the simulation."""
        try:
            images = {}
            for i in range(self.env.model.ncam):
                cam_name = mujoco.mj_id2name(
                    self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, i
                )
                if not cam_name:
                    continue
                img = self.env.mujoco_renderer.render("rgb_array", camera_name=cam_name)
                if img is not None:
                    images[cam_name] = img

            if images:
                self._latest_rgb_image = images
                return self._latest_rgb_image
            return self._latest_rgb_image

        except Exception as e:
            self.logger.error(f"Failed to get RGB image: {e}", exc_info=True)
            return None

    def get_latest_depth_image(self) -> Optional[Dict[str, np.ndarray]]:
        """Get latest depth images from all cameras in the simulation."""
        try:
            images = {}
            for i in range(self.env.model.ncam):
                cam_name = mujoco.mj_id2name(
                    self.env.model, mujoco.mjtObj.mjOBJ_CAMERA, i
                )
                if not cam_name:
                    continue
                depth_image = self.env.mujoco_renderer.render(
                    "depth_array", camera_name=cam_name
                )

                if depth_image is not None and isinstance(depth_image, np.ndarray):
                    # If the values are already large (max > 1.0), they are likely
                    # already linearized distances in meters.
                    if np.max(depth_image) > 1.0:
                        images[cam_name] = depth_image
                        continue

                    # The depth buffer from MuJoCo is non-linear [0, 1].
                    # We convert it to a linear distance array (in meters).
                    znear = self.env.model.vis.map.znear
                    zfar = self.env.model.vis.map.zfar
                    # Correct for auto-scaling of znear and zfar
                    extent = self.env.model.stat.extent
                    znear *= extent
                    zfar *= extent

                    # The formula to convert is:
                    # dist = znear / (1 - depth_buffer * (1 - znear / zfar))
                    # To avoid division by zero for values at zfar, we clip.
                    epsilon = 1e-6
                    depth_image = np.clip(depth_image, 0.0, 1.0 - epsilon)
                    dist = znear / (1 - depth_image * (1 - znear / zfar))
                    images[cam_name] = dist

            if images:
                self._latest_depth_image = images
                return self._latest_depth_image
            return self._latest_depth_image

        except Exception as e:
            self.logger.error(f"Failed to get depth image: {e}", exc_info=True)
            return None

    def get_camera_intrinsics(self) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        """Get camera intrinsic parameters."""
        return None

    def get_cam_to_base_transform(self) -> Optional[np.ndarray]:
        """Get camera to base frame transformation."""
        return None
