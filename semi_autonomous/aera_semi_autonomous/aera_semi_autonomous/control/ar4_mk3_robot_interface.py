import logging
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

        # Home position for the robot
        self.home_joint_positions = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def get_logger(self) -> logging.Logger:
        return self.logger

    def go_home(self) -> bool:
        """Move robot to home position using joint control."""
        try:
            # Create action for joint control (assuming joint control mode)
            if self.env.use_eef_control:
                # For end-effector control, we need to compute the home pose
                # This is a simplified approach - you may need to adjust
                home_action = np.array(
                    [0.0, 0.0, 0.0, -1.0]
                )  # [dx, dy, dz, gripper_open]
            else:
                # For joint control, move towards home position
                current_qpos = self.env.data.qpos[:6]  # First 6 joints are arm joints
                joint_diff = self.home_joint_positions - current_qpos
                # Limit the movement per step
                joint_diff = np.clip(joint_diff, -0.1, 0.1)
                home_action = np.concatenate([joint_diff, [-1.0]])  # Open gripper

            # Execute multiple steps to reach home
            for _ in range(50):  # Adjust number of steps as needed
                _, _, _, _, _ = self.env.step(home_action)
                # Check if close enough to home
                if self.env.use_eef_control:
                    break  # For EEF control, single action might be enough
                else:
                    current_qpos = self.env.data.qpos[:6]
                    if np.allclose(current_qpos, self.home_joint_positions, atol=0.05):
                        break

            self.logger.info("Robot moved to home position")
            return True

        except Exception as e:
            self.logger.error(f"Failed to go home: {e}")
            return False

    def move_to(self, pose: Pose) -> bool:
        """Move end-effector to specified pose."""
        try:
            # Convert ROS Pose to target position and orientation
            target_pos = np.array([pose.position.x, pose.position.y, pose.position.z])
            _ = np.array(
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )

            if self.env.use_eef_control:
                # For end-effector control
                current_pos = self.env._utils.get_site_xpos(
                    self.env.model, self.env.data, "grip"
                )
                pos_diff = target_pos - current_pos
                # Limit movement per step
                pos_diff = np.clip(pos_diff, -0.05, 0.05)
                action = np.concatenate([pos_diff, [0.0]])  # Keep gripper state

                # Execute multiple steps to reach target
                for _ in range(100):
                    _, _, _, _, _ = self.env.step(action)
                    current_pos = self.env._utils.get_site_xpos(
                        self.env.model, self.env.data, "grip"
                    )
                    if np.linalg.norm(current_pos - target_pos) < 0.01:
                        break
                    pos_diff = target_pos - current_pos
                    pos_diff = np.clip(pos_diff, -0.05, 0.05)
                    action = np.concatenate([pos_diff, [0.0]])
            else:
                # For joint control, this is more complex - would need inverse kinematics
                # For now, implement a simplified version
                self.logger.warning("Joint control mode move_to not fully implemented")
                return False

            self.logger.info(f"Robot moved to pose: {pose}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to move to pose: {e}")
            return False

    def release_gripper(self) -> bool:
        """Open the gripper."""
        try:
            if self.env.use_eef_control:
                action = np.array([0.0, 0.0, 0.0, -1.0])  # Open gripper
            else:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])  # Open gripper

            # Execute action to open gripper
            _, _, _, _, _ = self.env.step(action)
            self.logger.info("Gripper released")
            return True

        except Exception as e:
            self.logger.error(f"Failed to release gripper: {e}")
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

            # Close gripper
            gripper_action = (gripper_pos + 1.0) / 2.0  # Convert to [0,1] range
            if self.env.use_eef_control:
                action = np.array([0.0, 0.0, 0.0, gripper_action])
            else:
                action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_action])

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
            self.logger.error(f"Failed to grasp at pose: {e}")
            return False

    def release_at(self, pose: Pose) -> bool:
        """Move to pose and release gripper."""
        try:
            if not self.move_to(pose):
                return False
            return self.release_gripper()

        except Exception as e:
            self.logger.error(f"Failed to release at pose: {e}")
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
            self.logger.error(f"Failed to get end-effector pose: {e}")
            return None

    def get_latest_rgb_image(self) -> Optional[np.ndarray]:
        """Get latest RGB image from simulation."""
        try:
            # Render RGB image from MuJoCo
            rgb_image = self.env.render()
            if rgb_image is not None:
                self._latest_rgb_image = rgb_image
                return rgb_image  # type: ignore
            return self._latest_rgb_image  # type: ignore

        except Exception as e:
            self.logger.error(f"Failed to get RGB image: {e}")
            return None

    def get_latest_depth_image(self) -> Optional[np.ndarray]:
        """Get latest depth image from simulation."""
        try:
            # MuJoCo depth rendering (this is a simplified implementation)
            # You may need to configure MuJoCo rendering for depth
            depth_image = self.env.render(mode="depth_array")
            if depth_image is not None:
                self._latest_depth_image = depth_image
                return depth_image
            return self._latest_depth_image

        except Exception as e:
            self.logger.error(f"Failed to get depth image: {e}")
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
