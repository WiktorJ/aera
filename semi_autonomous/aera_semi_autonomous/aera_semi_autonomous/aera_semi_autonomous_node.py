import faulthandler
import time
from functools import cached_property

import numpy as np
import open3d as o3d
import rclpy
import tf2_ros
from cv_bridge import CvBridge
from rclpy.callback_groups import ReentrantCallbackGroup, MutuallyExclusiveCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String

from .point_cloud_conversion import point_cloud_to_msg
from .config.constants import _TF_PREFIX, BASE_LINK_NAME
from .vision.object_detector import ObjectDetector
from .vision.point_cloud_processor import PointCloudProcessor
from .control.robot_controller import RobotController
from .utils.debug_utils import DebugUtils
from .commands.command_processor import CommandProcessor
from .manipulation.manipulation_handler import ManipulationHandler


class AeraSemiAutonomous(Node):
    def __init__(self):
        super().__init__("aera_semi_autonomous_node")

        # Declare parameters
        self.declare_parameter("offset_x", 0.045)
        self.declare_parameter("offset_y", 0.05)
        self.declare_parameter("offset_z", 0.1)
        self.declare_parameter("gripper_squeeze_factor", 0.2)
        self.declare_parameter("debug_mode", False)

        # Get parameter values
        self.offset_x = (
            self.get_parameter("offset_x").get_parameter_value().double_value
        )
        self.offset_y = (
            self.get_parameter("offset_y").get_parameter_value().double_value
        )
        self.offset_z = (
            self.get_parameter("offset_z").get_parameter_value().double_value
        )
        self.gripper_squeeze_factor = (
            self.get_parameter("gripper_squeeze_factor")
            .get_parameter_value()
            .double_value
        )
        self.debug_mode = (
            self.get_parameter("debug_mode").get_parameter_value().bool_value
        )

        # Initialize basic components
        self.logger = self.get_logger()
        self.cv_bridge = CvBridge()
        self.n_frames_processed = 0
        self._last_depth_msg = None
        self._last_rgb_msg = None
        self._object_in_gripper: bool = False
        self.camera_intrinsics = None
        self.image_width = None
        self.image_height = None
        
        # Feedback tracking for infrequent logging
        self._last_feedback_log_time = 0.0
        self._feedback_log_interval_seconds = 5.0  # Log every 5 seconds

        # Initialize callback groups
        arm_callback_group = ReentrantCallbackGroup()
        gripper_callback_group = ReentrantCallbackGroup()
        prompt_callback_group = MutuallyExclusiveCallbackGroup()

        # Initialize arm joint names
        self.arm_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        self.arm_joint_names = [
            f"{_TF_PREFIX}{joint_name}" for joint_name in self.arm_joint_names
        ]

        # Initialize TF components
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Initialize modular components
        self.object_detector = ObjectDetector(self.logger)
        self.point_cloud_processor = PointCloudProcessor(self.logger)
        self.robot_controller = RobotController(
            self, self.arm_joint_names, _TF_PREFIX, self.debug_mode, self._moveit_feedback_callback
        )
        self.debug_utils = DebugUtils(
            self.logger, save_debug_images=True, debug_visualizations=False
        )
        self.command_processor = CommandProcessor(self.logger)

        # Initialize subscriptions
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self._camera_info_callback,
            10,
        )

        # Create delayed subscriptions
        self.image_sub = None
        self.depth_sub = None
        self.create_timer(3.0, self._create_delayed_image_subscription)
        self.create_timer(3.0, self._create_delayed_depth_subscription)

        self.prompt_sub = self.create_subscription(
            String,
            "/prompt",
            self.start,
            10,
            callback_group=prompt_callback_group,
        )

        self.logger.info("Aera Semi Autonomous node initialized.")

    def _moveit_feedback_callback(self, feedback):
        """Callback for MoveIt2 execution feedback. Logs infrequently to avoid spam."""
        current_time = time.time()
        if current_time - self._last_feedback_log_time >= self._feedback_log_interval_seconds:
            self.logger.info(f"MoveIt2 feedback: {feedback}")
            self._last_feedback_log_time = current_time

    def _create_delayed_image_subscription(self):
        """Create the image subscription after a delay."""
        if self.image_sub is None:
            self.image_sub = self.create_subscription(
                Image, "/camera/camera/color/image_raw", self.image_callback, 10
            )
            self.logger.info("Image subscription created after 3 second delay.")

    def _create_delayed_depth_subscription(self):
        """Create the depth subscription after a delay."""
        if self.depth_sub is None:
            self.depth_sub = self.create_subscription(
                Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, 10
            )
            self.logger.info("Depth subscription created after 3 second delay.")

    def _camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.logger.info("Received camera intrinsics.")
            self.image_width = msg.width
            self.image_height = msg.height
            # K is a 3x3 matrix (row-major order in a list of 9)
            # K = [fx, 0,  cx,
            #      0,  fy, cy,
            #      0,  0,  1]
            self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=msg.width,
                height=msg.height,
                fx=msg.k[0],
                fy=msg.k[4],
                cx=msg.k[2],
                cy=msg.k[5],
            )
            # Unsubscribe after getting the info because it's static
            self.destroy_subscription(self.camera_info_sub)

    def _initialize_manipulation_handler(self):
        """Initialize the manipulation handler with current parameters."""
        if self.camera_intrinsics is None:
            return None

        return ManipulationHandler(
            self.point_cloud_processor,
            self.robot_controller,
            self.debug_utils,
            self.camera_intrinsics,
            self.cam_to_base_affine,
            self.offset_x,
            self.offset_y,
            self.offset_z,
            self.gripper_squeeze_factor,
            self.n_frames_processed,
        )

    def start(self, msg: String):
        """Main entry point for processing commands."""
        commands = self.command_processor.parse_prompt_message(msg.data)
        if not commands:
            self.logger.warn(f"Could not parse commands from: {msg.data}")
            return

        self.logger.info(f"Processing: {msg.data}")
        self.debug_utils.setup_debug_logging(msg.data, self.camera_intrinsics)

        # Initialize manipulation handler
        manipulation_handler = self._initialize_manipulation_handler()
        if manipulation_handler is None:
            self.logger.error(
                "Cannot initialize manipulation handler without camera intrinsics."
            )
            return

        for action, object_to_detect in commands:
            if not self._last_rgb_msg or not self._last_depth_msg:
                self.logger.warn(
                    f"rgb_msg present: {self._last_rgb_msg is not None}, depth_msg present: {self._last_depth_msg is not None}"
                )
                self.logger.error("No image messages received. Aborting command chain.")
                return

            # Use the latest images for each action
            rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
            depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)

            self.logger.info(
                f"Executing action: '{action}' on object: '{object_to_detect}'"
            )

            # Handle the command and update object_in_gripper status
            self._object_in_gripper = self.command_processor.handle_tool_call(
                action,
                object_to_detect,
                self.object_detector,
                self.robot_controller,
                manipulation_handler,
                rgb_image,
                depth_image,
                self._object_in_gripper,
            )

        self.robot_controller.go_home()
        self.logger.info("Task completed.")

    @cached_property
    def cam_to_base_affine(self):
        cam_to_base_link_tf = self.tf_buffer.lookup_transform(
            target_frame=BASE_LINK_NAME,
            source_frame="camera_color_optical_frame",
            # source_frame="camera_color_frame",
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

    def depth_callback(self, msg):
        """Callback for depth image messages."""
        if self.debug_mode and self._last_depth_msg is not None:
            return
        self._last_depth_msg = msg

    def image_callback(self, msg):
        """Callback for RGB image messages."""
        if self.debug_mode and self._last_rgb_msg is not None:
            return
        self._last_rgb_msg = msg


def main():
    faulthandler.enable()
    rclpy.init()
    node = AeraSemiAutonomous()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
