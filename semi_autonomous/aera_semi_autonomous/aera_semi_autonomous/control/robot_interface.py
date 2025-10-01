from abc import ABC, abstractmethod
import logging
from typing import Optional

import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose
from sensor_msgs.msg import Image


class RobotInterface(ABC):
    """
    An abstract interface for controlling a robot arm and sensors, allowing for
    different backend implementations (e.g., real robot via ROS, or a
    simulation).
    """

    @abstractmethod
    def get_logger(self) -> logging.Logger:
        """Returns the logger object."""
        pass

    @abstractmethod
    def go_home(self) -> bool:
        """Moves the robot to a predefined home position."""
        pass

    @abstractmethod
    def move_to(self, pose: Pose) -> bool:
        """Moves the end-effector to a specific pose."""
        pass

    @abstractmethod
    def release_gripper(self) -> bool:
        """Opens the gripper."""
        pass

    @abstractmethod
    def grasp_at(self, pose: Pose, gripper_pos: float) -> bool:
        """Perform a grasp at the specified pose with the given gripper position."""
        pass

    @abstractmethod
    def release_at(self, pose: Pose) -> bool:
        """Move to the specified pose and release the gripper."""
        pass

    @abstractmethod
    def get_end_effector_pose(self) -> Optional[Pose]:
        """Returns the current pose of the end-effector."""
        pass

    @abstractmethod
    def get_latest_rgb_image(self) -> Optional[np.ndarray]:
        """Returns the latest RGB image as a NumPy array."""
        pass

    @abstractmethod
    def get_latest_depth_image(self) -> Optional[np.ndarray]:
        """Returns the latest depth image as a NumPy array."""
        pass

    @abstractmethod
    def get_camera_intrinsics(self) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        """Returns the camera intrinsic parameters."""
        pass

    @abstractmethod
    def get_cam_to_base_transform(self) -> Optional[np.ndarray]:
        """
        Returns the 4x4 affine transformation from camera frame to base frame.
        """
        pass

    @abstractmethod
    def get_last_rgb_msg(self) -> Optional[Image]:
        """Returns the last raw RGB image message."""
        pass
