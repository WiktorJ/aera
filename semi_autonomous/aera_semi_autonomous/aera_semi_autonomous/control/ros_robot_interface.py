from functools import cached_property
from typing import Optional, Any, Dict

import numpy as np
import open3d as o3d
import tf2_ros
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import CameraInfo, Image

from aera_semi_autonomous.config.constants import BASE_LINK_NAME, _TF_PREFIX
from aera_semi_autonomous.control.robot_controller import RobotController
from aera_semi_autonomous.control.robot_interface import RobotInterface
from aera_semi_autonomous.data.trajectory_data_collector import TrajectoryDataCollector


class RosRobotInterface(RobotInterface):
    def __init__(
        self,
        node: Node,
        arm_joint_names: list,
        debug_mode: bool,
        feedback_callback,
        trajectory_collector: Optional[TrajectoryDataCollector] = None,
    ):
        self._node = node
        self._logger = node.get_logger()
        self.cv_bridge = CvBridge()
        self.debug_mode = debug_mode
        self.trajectory_collector = trajectory_collector

        self.robot_controller = RobotController(
            node=node,
            arm_joint_names=arm_joint_names,
            tf_prefix=_TF_PREFIX,
            debug_mode=debug_mode,
            feedback_callback=feedback_callback,
        )

        self._last_rgb_msg: Optional[Image] = None
        self._last_depth_msg: Optional[Image] = None
        self.camera_intrinsics: Optional[o3d.camera.PinholeCameraIntrinsic] = None

        # Initialize TF components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self._node)

        # Initialize subscriptions
        self.camera_info_sub = self._node.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self._camera_info_callback,
            10,
        )

        self.rbg_image_callback_group = ReentrantCallbackGroup()
        self.depth_image_callback_group = ReentrantCallbackGroup()
        self.image_sub = self._node.create_subscription(
            Image,
            "/camera/camera/color/image_raw",
            self._image_callback,
            10,
            callback_group=self.rbg_image_callback_group,
        )
        self.depth_sub = self._node.create_subscription(
            Image,
            "/camera/camera/depth/image_rect_raw",
            self._depth_callback,
            10,
            callback_group=self.depth_image_callback_group,
        )

    def get_logger(self) -> Any:
        return self._logger

    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self._logger.info("RosRobotInterface: Received camera intrinsics.")
            self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=msg.width,
                height=msg.height,
                fx=msg.k[0],
                fy=msg.k[4],
                cx=msg.k[2],
                cy=msg.k[5],
            )
            self._node.destroy_subscription(self.camera_info_sub)

    def _depth_callback(self, msg: Image):
        if self.trajectory_collector:
            self.trajectory_collector.record_depth_image(msg)
        if self.debug_mode and self._last_depth_msg is not None:
            return
        self._last_depth_msg = msg

    def _image_callback(self, msg: Image):
        if self.trajectory_collector:
            self.trajectory_collector.record_rgb_image(msg)
        if self.debug_mode and self._last_rgb_msg is not None:
            return
        self._last_rgb_msg = msg

    def go_home(self) -> bool:
        return self.robot_controller.go_home()

    def move_to(self, pose: Pose) -> bool:
        return self.robot_controller.move_to(pose)

    def release_gripper(self) -> bool:
        return self.robot_controller.release_gripper()

    def grasp_at(self, pose: Pose, gripper_pos: float) -> bool:
        return self.robot_controller.grasp_at(pose, gripper_pos)

    def release_at(self, pose: Pose) -> bool:
        return self.robot_controller.release_at(pose)

    def get_end_effector_pose(self) -> Optional[Pose]:
        return self.robot_controller.get_current_end_effector_pose()

    def get_latest_rgb_image(self) -> Optional[Dict[str, np.ndarray]]:
        if self._last_rgb_msg is None:
            return None
        image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg, "bgr8")
        return {"default_camera": image}

    def get_latest_depth_image(self) -> Optional[Dict[str, np.ndarray]]:
        if self._last_depth_msg is None:
            return None
        image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        return {"default_camera": image}

    def get_camera_intrinsics(self) -> Optional[o3d.camera.PinholeCameraIntrinsic]:
        return self.camera_intrinsics

    @cached_property
    def _cam_to_base_affine(self) -> Optional[np.ndarray]:
        try:
            cam_to_base_link_tf = self.tf_buffer.lookup_transform(
                target_frame=BASE_LINK_NAME,
                source_frame="camera_color_optical_frame",
                time=Time(),
                timeout=Duration(seconds=5),
            )
            cam_to_base_rot = Rotation.from_quat(
                [
                    cam_to_base_link_tf.transform.rotation.x,
                    cam_to_base_link_tf.transform.rotation.y,
                    cam_to_base_link_tf.transform.rotation.z,
                    cam_to_base_link_tf.transform.rotation.w,
                ]
            )
            cam_to_base_pos = np.array(
                [
                    cam_to_base_link_tf.transform.translation.x,
                    cam_to_base_link_tf.transform.translation.y,
                    cam_to_base_link_tf.transform.translation.z,
                ]
            )
            affine = np.eye(4)
            affine[:3, :3] = cam_to_base_rot.as_matrix()
            affine[:3, 3] = cam_to_base_pos
            return affine
        except (
            tf2_ros.LookupException,  # type: ignore
            tf2_ros.ConnectivityException,  # type: ignore
            tf2_ros.ExtrapolationException,  # type: ignore
        ) as e:
            self._logger.error(f"Could not get cam_to_base_transform: {e}")
            return None

    def get_cam_to_base_transform(self) -> Optional[np.ndarray]:
        return self._cam_to_base_affine
