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

try:
    from dm_control.mujoco.wrapper import mjbindings
    mjlib = mjbindings.mjlib
except ImportError:
    # Fallback for environments without dm_control
    mjlib = None

# IK Result namedtuple
IKResult = collections.namedtuple(
    'IKResult', ['qpos', 'err_norm', 'steps', 'success'])


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
        self.home_pose = None

    def get_logger(self) -> logging.Logger:
        return self.logger

    def go_home(self) -> bool:
        """Move robot to home position and release gripper."""
        try:
            if self.home_pose is None:
                self._initialize_home_pose()

            # Get current position to check if we're already at home
            current_pose = self.get_end_effector_pose()
            if current_pose is not None:
                current_pos = np.array(
                    [
                        current_pose.position.x,
                        current_pose.position.y,
                        current_pose.position.z,
                    ]
                )
                home_pos = np.array(
                    [
                        self.home_pose.position.x,
                        self.home_pose.position.y,
                        self.home_pose.position.z,
                    ]
                )

                # If we're already close to home position, just release gripper
                position_tolerance = 0.01  # 1cm tolerance
                if np.linalg.norm(current_pos - home_pos) < position_tolerance:
                    self.logger.info("Already at home position, just releasing gripper")
                    return self.release_gripper()

            # Move to home position if not already there
            if not self.move_to(self.home_pose):
                return False

            # Then release gripper
            return self.release_gripper()

        except Exception as e:
            self.logger.error(f"Failed to go home: {e}", exc_info=True)
            return False

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
            # Convert ROS Pose to target position and orientation
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            target_quat = np.array([
                pose.orientation.x, pose.orientation.y, 
                pose.orientation.z, pose.orientation.w
            ])

            self.logger.info(f"Moving to target position: {target_pos}")
            self.logger.info(f"Target orientation (quat): {target_quat}")

            # Use inverse kinematics to find joint positions
            ik_result = self._solve_ik_for_site_pose(
                site_name="grip",
                target_pos=target_pos,
                target_quat=target_quat,
                tol=1e-3,
                max_steps=100
            )

            if not ik_result.success:
                self.logger.warning(
                    f"IK failed to converge. Error norm: {ik_result.err_norm:.6f}, "
                    f"Steps: {ik_result.steps}"
                )
                return False

            # Apply the computed joint positions
            self._apply_joint_positions(ik_result.qpos)

            # Verify the final position
            final_pos = self.env._utils.get_site_xpos(
                self.env.model, self.env.data, "grip"
            )
            final_error = np.linalg.norm(target_pos - final_pos)

            self.logger.info(
                f"IK converged in {ik_result.steps} steps. "
                f"Final position error: {final_error:.6f}m"
            )

            return True

        except Exception as e:
            self.logger.error(f"Failed to move to pose: {e}", exc_info=True)
            return False

    def release_gripper(self) -> bool:
        """Open the gripper gradually."""
        try:
            # Get current gripper state to start from current position
            current_gripper_state = self.env.data.qpos[-2:]  # Last 2 joints are gripper
            current_gripper_value = np.mean(
                current_gripper_state
            )  # Average of both gripper joints

            # Convert from joint position to action space (-1 to 1)
            # Gripper joint range is approximately -0.014 to 0.0
            current_action_value = (current_gripper_value / -0.014) * 2.0 - 1.0
            current_action_value = np.clip(current_action_value, -1.0, 1.0)

            target_gripper_value = -1.0  # Fully open

            # Only move if we're not already open
            if abs(current_action_value - target_gripper_value) < 0.1:
                self.logger.info("Gripper already open")
                return True

            # Single step to open gripper
            if self.env.use_eef_control:
                action = np.array([0.0, 0.0, 0.0, target_gripper_value])
            else:
                action = np.array(
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, target_gripper_value]
                )

            _, _, _, _, _ = self.env.step(action)

            self.logger.info("Gripper released")
            return True

        except Exception as e:
            self.logger.error(f"Failed to release gripper: {e}", exc_info=True)
            return False

    def grasp_at(self, pose: Pose, gripper_pos: float) -> bool:
        """Perform grasp at specified pose with given gripper position."""
        try:
            # First move above the target
            above_pose = Pose()
            above_pose.position.x = pose.position.x
            above_pose.position.y = pose.position.y
            above_pose.position.z = pose.position.z + 0.05
            above_pose.orientation = pose.orientation

            if not self.move_to(above_pose):
                return False

            # Move down to grasp position
            if not self.move_to(pose):
                return False

            # Close gripper gradually
            target_gripper_value = gripper_pos  # Target gripper position
            max_gripper_steps = 50  # Number of steps for smooth gripper closing

            for step in range(max_gripper_steps):
                # Gradually move gripper to target position
                progress = (step + 1) / max_gripper_steps
                current_gripper_value = target_gripper_value * progress

                if self.env.use_eef_control:
                    action = np.array([0.0, 0.0, 0.0, current_gripper_value])
                else:
                    action = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_gripper_value]
                    )

                _, _, _, _, _ = self.env.step(action)

            # Lift object
            lift_pose = Pose()
            lift_pose.position.x = pose.position.x
            lift_pose.position.y = pose.position.y
            lift_pose.position.z = pose.position.z + 0.1
            lift_pose.orientation = pose.orientation

            if not self.move_to(lift_pose):
                return False

            self.logger.info(f"Grasped object at pose: {pose}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to grasp at pose: {e}", exc_info=True)
            return False

    def release_at(self, pose: Pose) -> bool:
        """Move to pose and release gripper."""
        try:
            if not self.move_to(pose):
                return False
            return self.release_gripper()

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
            depth_image = self.env.render(mode="depth_array")
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

    def set_cam_to_base_transform(self, transform: np.ndarray):
        """Set the camera to base transformation matrix."""
        if transform.shape == (4, 4):
            self.cam_to_base_transform = transform
        else:
            raise ValueError("Transform must be a 4x4 matrix")

    def _solve_ik_for_site_pose(self, site_name: str, target_pos: Optional[np.ndarray] = None,
                               target_quat: Optional[np.ndarray] = None, joint_names: Optional[list] = None,
                               tol: float = 1e-14, rot_weight: float = 1.0,
                               regularization_threshold: float = 0.1,
                               regularization_strength: float = 3e-2,
                               max_update_norm: float = 2.0,
                               progress_thresh: float = 20.0,
                               max_steps: int = 100) -> IKResult:
        """
        Find joint positions that satisfy a target site position and/or rotation.
        Based on dm_control's inverse kinematics implementation.
        """
        if mjlib is None:
            raise RuntimeError("mjlib not available. Install dm_control for IK support.")
        
        if target_pos is None and target_quat is None:
            raise ValueError("At least one of target_pos or target_quat must be specified.")

        dtype = self.env.data.qpos.dtype
        
        # Get site ID
        site_id = self.env._model_names.site_name2id.get(site_name)
        if site_id is None:
            raise ValueError(f"Site '{site_name}' not found in model")

        # Determine which joints to use (default to arm joints, excluding gripper)
        if joint_names is None:
            # Get all joint names and exclude gripper joints
            all_joint_names = [self.env.model.joint(i).name for i in range(self.env.model.njnt)]
            joint_names = [name for name in all_joint_names 
                          if not any(gripper_keyword in name.lower() 
                                   for gripper_keyword in ['gripper', 'finger'])]

        # Get joint indices and DOF indices
        joint_indices = []
        dof_indices = []
        for name in joint_names:
            try:
                joint_id = self.env._model_names.joint_name2id[name]
                joint_indices.append(joint_id)
                # Get DOF index for this joint
                dof_start = self.env.model.jnt_dofadr[joint_id]
                dof_count = 1  # Assuming single DOF joints
                dof_indices.extend(range(dof_start, dof_start + dof_count))
            except KeyError:
                self.logger.warning(f"Joint '{name}' not found, skipping")

        dof_indices = np.array(dof_indices)
        
        # Set up Jacobian and error arrays
        if target_pos is not None and target_quat is not None:
            jac = np.empty((6, self.env.model.nv), dtype=dtype)
            err = np.empty(6, dtype=dtype)
            jac_pos, jac_rot = jac[:3], jac[3:]
            err_pos, err_rot = err[:3], err[3:]
        else:
            jac = np.empty((3, self.env.model.nv), dtype=dtype)
            err = np.empty(3, dtype=dtype)
            if target_pos is not None:
                jac_pos, jac_rot = jac, None
                err_pos, err_rot = err, None
            else:  # target_quat is not None
                jac_pos, jac_rot = None, jac
                err_pos, err_rot = None, err

        update_nv = np.zeros(self.env.model.nv, dtype=dtype)
        
        if target_quat is not None:
            # Normalize target quaternion
            target_quat = target_quat / np.linalg.norm(target_quat)

        # Main IK loop
        steps = 0
        success = False
        
        for steps in range(max_steps):
            # Forward kinematics to get current site pose
            mjlib.mj_fwdPosition(self.env.model.ptr, self.env.data.ptr)
            
            # Get current site position and orientation
            current_pos = self.env._utils.get_site_xpos(self.env.model, self.env.data, site_name)
            
            # Compute position error
            if target_pos is not None:
                err_pos[:] = target_pos - current_pos
                # Compute position Jacobian
                mjlib.mj_jacSite(self.env.model.ptr, self.env.data.ptr, 
                               jac_pos, None, site_id)

            # Compute orientation error
            if target_quat is not None:
                current_quat = self._get_site_quaternion(site_name)
                err_rot[:] = self._compute_quaternion_error(target_quat, current_quat)
                # Compute rotation Jacobian
                mjlib.mj_jacSite(self.env.model.ptr, self.env.data.ptr, 
                               None, jac_rot, site_id)

            # Compute weighted error norm
            if target_pos is not None and target_quat is not None:
                err_norm = np.linalg.norm(err_pos) + rot_weight * np.linalg.norm(err_rot)
            elif target_pos is not None:
                err_norm = np.linalg.norm(err_pos)
            else:
                err_norm = rot_weight * np.linalg.norm(err_rot)

            # Check convergence
            if err_norm < tol:
                success = True
                break

            # Extract Jacobian for the specified joints
            jac_joints = jac[:, dof_indices]

            # Determine regularization strength
            reg_strength = (regularization_strength if err_norm > regularization_threshold 
                          else 0.0)

            # Compute joint update using nullspace method
            update_joints = self._nullspace_method(
                jac_joints, err, regularization_strength=reg_strength)
            update_norm = np.linalg.norm(update_joints)

            # Check progress
            if update_norm > 0:
                progress_criterion = err_norm / update_norm
                if progress_criterion > progress_thresh:
                    self.logger.debug(
                        f'Step {steps}: err_norm / update_norm ({progress_criterion:.3g}) > '
                        f'tolerance ({progress_thresh:.3g}). Halting due to insufficient progress'
                    )
                    break

            # Limit update norm
            if update_norm > max_update_norm:
                update_joints *= max_update_norm / update_norm

            # Apply update to full DOF vector
            update_nv[dof_indices] = update_joints

            # Update joint positions
            mjlib.mj_integratePos(self.env.model.ptr, self.env.data.qpos, update_nv, 1)

            self.logger.debug(f'Step {steps}: err_norm={err_norm:.3g} update_norm={update_norm:.3g}')

        if not success and steps == max_steps - 1:
            self.logger.warning(f'IK failed to converge after {steps} steps: err_norm={err_norm:.3g}')

        return IKResult(qpos=self.env.data.qpos.copy(), err_norm=err_norm, 
                       steps=steps, success=success)

    def _get_site_quaternion(self, site_name: str) -> np.ndarray:
        """Get the quaternion orientation of a site."""
        # Get rotation matrix
        rot_matrix = self.env._utils.get_site_xmat(self.env.model, self.env.data, site_name)
        rot_matrix = rot_matrix.reshape(3, 3)
        
        # Convert to quaternion [x, y, z, w]
        rotation = Rotation.from_matrix(rot_matrix)
        return rotation.as_quat()

    def _compute_quaternion_error(self, target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
        """Compute the quaternion error for IK."""
        # Ensure quaternions are normalized
        target_quat = target_quat / np.linalg.norm(target_quat)
        current_quat = current_quat / np.linalg.norm(current_quat)
        
        # Compute quaternion difference
        # q_error = q_target * q_current^(-1)
        current_quat_inv = np.array([-current_quat[0], -current_quat[1], -current_quat[2], current_quat[3]])
        
        # Quaternion multiplication: q1 * q2
        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1[3], q1[0], q1[1], q1[2]
            w2, x2, y2, z2 = q2[3], q2[0], q2[1], q2[2]
            return np.array([
                w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
                w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
                w1*z2 + x1*y2 - y1*x2 + z1*w2,  # z
                w1*w2 - x1*x2 - y1*y2 - z1*z2   # w
            ])
        
        q_error = quat_mult(target_quat, current_quat_inv)
        
        # Convert to axis-angle representation (scaled by angle)
        if q_error[3] < 0:
            q_error = -q_error
        
        # Return the vector part scaled by 2 * angle
        return 2.0 * q_error[:3]

    def _nullspace_method(self, jac_joints: np.ndarray, delta: np.ndarray, 
                         regularization_strength: float = 0.0) -> np.ndarray:
        """
        Calculate joint velocities to achieve specified end effector delta.
        Based on dm_control's nullspace method.
        """
        hess_approx = jac_joints.T.dot(jac_joints)
        joint_delta = jac_joints.T.dot(delta)
        
        if regularization_strength > 0:
            # L2 regularization
            hess_approx += np.eye(hess_approx.shape[0]) * regularization_strength
            return np.linalg.solve(hess_approx, joint_delta)
        else:
            return np.linalg.lstsq(hess_approx, joint_delta, rcond=None)[0]

    def _apply_joint_positions(self, qpos: np.ndarray):
        """Apply joint positions to the environment."""
        # Copy the joint positions to the environment
        self.env.data.qpos[:] = qpos
        
        # Forward kinematics to update all dependent quantities
        mjlib.mj_fwdPosition(self.env.model.ptr, self.env.data.ptr)
        
        # If using joint control, we need to step the environment to apply the positions
        if not self.env.use_eef_control:
            # For joint control, create action from joint positions
            # This is a simplified approach - in practice you might want to use
            # position control or compute joint velocities
            current_qpos = self.env.data.qpos[:-2]  # Exclude gripper joints
            target_qpos = qpos[:-2]  # Exclude gripper joints
            
            # Simple proportional control
            action = (target_qpos - current_qpos) * 10.0  # Scale factor
            action = np.clip(action, -1.0, 1.0)  # Clip to action space
            
            # Add gripper action (keep current state)
            gripper_action = 0.0
            action = np.append(action, gripper_action)
            
            # Step the environment
            self.env.step(action)
