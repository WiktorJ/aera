import logging
import time
from typing import Optional
import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image
from scipy.spatial.transform import Rotation

from aera_semi_autonomous.control.robot_interface import RobotInterface
from aera.autonomous.envs.ar4_mk3_base import Ar4Mk3Env


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
        """Move end-effector to specified pose."""
        try:
            # Convert ROS Pose to target position
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])

            max_steps = 1000  # Safety limit to prevent infinite loops
            position_tolerance = 0.005  # Position tolerance in meters
            max_step_size = 0.5  # Larger step size to account for environment scaling (0.05x)
            print(f"target_pos: {target_pos}")
            print("----------------------------------")

            if self.env.use_eef_control:
                # Get mocap body info
                body_id = self.env._model_names.body_name2id["robot0:mocap"]
                mocap_id = self.env.model.body_mocapid[body_id]

                step_count = 0
                while step_count < max_steps:
                    # Use mocap position instead of gripper site position for consistency
                    current_mocap_pos = self.env.data.mocap_pos[mocap_id].copy()
                    pos_diff = target_pos - current_mocap_pos

                    # Check if we've reached the target
                    if np.linalg.norm(pos_diff) < position_tolerance:
                        break

                    # Limit movement per step for smooth motion
                    pos_diff_clipped = np.clip(pos_diff, -max_step_size, max_step_size)
                    action = np.concatenate(
                        [pos_diff_clipped, [0.0]]
                    )  # Keep gripper state unchanged
                    print(f"step: {step_count}, pos_diff_clipped: {pos_diff_clipped}")
                    print(f"current_pos: {current_mocap_pos}")

                    # Get mocap position before and after step to debug
                    body_id = self.env._model_names.body_name2id["robot0:mocap"]
                    mocap_id = self.env.model.body_mocapid[body_id]
                    mocap_pos_before = self.env.data.mocap_pos[mocap_id].copy()

                    _, _, _, _, _ = self.env.step(action)
                    time.sleep(1)
                    new_mocap_pos = self.env.data.mocap_pos[mocap_id].copy()
                    print(f"new_mocap_pos: {new_mocap_pos}")
                    print(
                        f"mocap_pos_diff: {new_mocap_pos - mocap_pos_before}, distance: {np.linalg.norm(new_mocap_pos - mocap_pos_before)}"
                    )

                    # Apply the action
                    _, _, _, _, _ = self.env.step(action)

                    step_count += 1

                # Final check using actual gripper position
                final_grip_pos = self.env._utils.get_site_xpos(
                    self.env.model, self.env.data, "grip"
                )
                final_error = np.linalg.norm(target_pos - final_grip_pos)

                if step_count >= max_steps:
                    self.logger.warning(
                        f"Move to pose reached maximum steps ({max_steps}) without full convergence"
                    )
                    self.logger.warning(f"Final position error: {final_error:.6f}m")
                else:
                    self.logger.info(
                        f"Robot moved to pose in {step_count} steps, final error: {final_error:.6f}m"
                    )

            else:
                # For joint control, this is more complex - would need inverse kinematics
                # For now, implement a simplified version
                self.logger.warning("Joint control mode move_to not fully implemented")
                return False

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

            max_steps = 30  # Reduced steps for smoother motion
            target_gripper_value = -1.0  # Fully open

            # Only move if we're not already open
            if abs(current_action_value - target_gripper_value) < 0.1:
                self.logger.info("Gripper already open")
                return True

            for step in range(max_steps):
                # Gradually move gripper from current position to open position
                progress = (step + 1) / max_steps
                current_gripper_value = (
                    current_action_value
                    + (target_gripper_value - current_action_value) * progress
                )

                # Keep position unchanged, only modify gripper
                if self.env.use_eef_control:
                    action = np.array([0.0, 0.0, 0.0, current_gripper_value])
                else:
                    action = np.array(
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, current_gripper_value]
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

    def set_cam_to_base_transform(self, transform: np.ndarray):
        """Set the camera to base transformation matrix."""
        if transform.shape == (4, 4):
            self.cam_to_base_transform = transform
        else:
            raise ValueError("Transform must be a 4x4 matrix")
