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
