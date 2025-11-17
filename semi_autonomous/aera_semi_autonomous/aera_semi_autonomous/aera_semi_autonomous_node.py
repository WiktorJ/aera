import faulthandler
import time

import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from aera_semi_autonomous.config.constants import (
    _TF_PREFIX,
    ACTION_DESCRIPTIONS,
    AVAILABLE_ACTIONS,
)
from aera_semi_autonomous.vision.object_detector import ObjectDetector
from aera_semi_autonomous.vision.point_cloud_processor import PointCloudProcessor
from aera_semi_autonomous.control.ros_robot_interface import RosRobotInterface
from aera_semi_autonomous.utils.debug_utils import DebugUtils
from aera_semi_autonomous.commands.command_processor import CommandProcessor
from aera_semi_autonomous.manipulation.manipulation_handler import ManipulationHandler
from aera_semi_autonomous.data.trajectory_data_collector import TrajectoryDataCollector


class AeraSemiAutonomous(Node):
    def __init__(self):
        super().__init__("aera_semi_autonomous_node")

        # Declare parameters
        self.declare_parameter("offset_x", 0.045)
        self.declare_parameter("offset_y", 0.05)
        self.declare_parameter("offset_z", 0.1)
        self.declare_parameter("gripper_squeeze_factor", 0.2)
        self.declare_parameter("debug_mode", False)
        self.declare_parameter("sync_tolerance", 0.08)
        self.declare_parameter("collect_trajectory_data", True)

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
        self.sync_tolerance = (
            self.get_parameter("sync_tolerance").get_parameter_value().double_value
        )
        self.collect_trajectory_data = (
            self.get_parameter("collect_trajectory_data")
            .get_parameter_value()
            .bool_value
        )

        # Initialize basic components
        self.logger = self.get_logger()
        self.n_frames_processed = 0
        self._object_in_gripper: bool = False

        # Feedback tracking for infrequent logging
        self._last_feedback_log_time = 0.0
        self._feedback_log_interval_seconds = 0.5  # Log every 0.5 seconds

        # Initialize callback groups
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

        # Initialize gripper joint names
        self.gripper_joint_names = [f"{_TF_PREFIX}gripper_jaw1_joint"]

        # Initialize modular components
        self.debug_utils = DebugUtils(
            self.logger, save_debug_images=True, debug_visualizations=False
        )
        self.object_detector = ObjectDetector(self.logger, self.debug_utils)
        self.point_cloud_processor = PointCloudProcessor(self.logger)
        self.command_processor = CommandProcessor(self.logger)

        self.trajectory_collector = None
        if self.collect_trajectory_data:
            self.trajectory_collector = TrajectoryDataCollector(
                self.logger,
                self.arm_joint_names,
                self.gripper_joint_names,
                sync_tolerance=self.sync_tolerance,
            )

        self.robot_interface = RosRobotInterface(
            self,
            self.arm_joint_names,
            self.debug_mode,
            self._moveit_feedback_callback,
            trajectory_collector=self.trajectory_collector,
        )

        self.prompt_sub = self.create_subscription(
            String,
            "/prompt",
            self.start,
            10,
            callback_group=prompt_callback_group,
        )

        # Subscribe to joint states for RL data collection
        if self.collect_trajectory_data:
            self.joint_state_sub = self.create_subscription(
                JointState, "/joint_states", self._joint_state_callback_for_rl, 10
            )

        self.logger.info("Aera Semi Autonomous node initialized.")

    def _joint_state_callback_for_rl(self, msg: JointState):
        """Callback for joint state updates for RL data collection."""
        if not self.trajectory_collector:
            return

        self.trajectory_collector.record_joint_state(msg)

        # Extract arm joint positions for FK computation
        arm_positions = []
        for joint_name in self.arm_joint_names:
            if joint_name in msg.name:
                idx = list(msg.name).index(joint_name)
                arm_positions.append(msg.position[idx])

        # Only compute pose if we have complete arm joint data
        if len(arm_positions) == len(self.arm_joint_names):
            try:
                # Compute end effector pose using forward kinematics
                end_effector_pose = (
                    self.robot_interface.robot_controller.compute_end_effector_pose(
                        arm_positions
                    )
                )
                ros_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
                self.trajectory_collector.record_pose(end_effector_pose, ros_timestamp)
            except Exception as e:
                self.logger.error(
                    f"Failed to compute end effector pose for data collection: {e}"
                )

    def _moveit_feedback_callback(self, feedback_msg):
        """Callback for MoveIt2 execution feedback. Logs infrequently to avoid spam."""
        current_time = time.time()
        if (
            current_time - self._last_feedback_log_time
            >= self._feedback_log_interval_seconds
        ):
            self.logger.info(f"MoveIt2 feedback arg type: {type(feedback_msg)}")
            self.logger.info(f"Feedback_msg attributes: {dir(feedback_msg)}")
            self.logger.info(f"Feedback attributes: {dir(feedback_msg.feedback)}")
            self.logger.info(f"MoveIt2 feedback: {feedback_msg}")
            self._last_feedback_log_time = current_time

    def _initialize_manipulation_handler(self):
        """Initialize the manipulation handler with current parameters."""
        camera_intrinsics = self.robot_interface.get_camera_intrinsics()
        if camera_intrinsics is None:
            self.logger.warn("Waiting for camera intrinsics...")
            return None

        cam_to_base_affine = self.robot_interface.get_cam_to_base_transform()
        if cam_to_base_affine is None:
            self.logger.warn("Waiting for camera to base transform...")
            return None

        manipulation_handler = ManipulationHandler(
            self.point_cloud_processor,
            self.robot_interface,
            self.debug_utils,
            camera_intrinsics,
            cam_to_base_affine,
            self.offset_x,
            self.offset_y,
            self.offset_z,
            self.gripper_squeeze_factor,
            self.n_frames_processed,
        )
        # Initialize with default offsets
        manipulation_handler.update_offsets()
        return manipulation_handler

    def start(self, msg: String):
        """Main entry point for processing commands."""
        parse_result = self.command_processor.parse_prompt_message(msg.data)
        if not parse_result:
            self.logger.warn(f"Could not parse commands from: {msg.data}")
            return

        commands, offsets = parse_result
        self.logger.info(f"Processing: {msg.data}")
        self.debug_utils.setup_debug_logging(
            msg.data, self.robot_interface.get_camera_intrinsics()
        )

        # Start RL data collection
        if self.trajectory_collector:
            self.trajectory_collector.start_episode(msg.data)

        # Initialize manipulation handler
        manipulation_handler = self._initialize_manipulation_handler()
        if manipulation_handler is None:
            self.logger.error(
                "Cannot initialize manipulation handler. Check camera intrinsics and TF transform."
            )
            return

        # Update offsets if provided in the prompt
        if offsets:
            self.logger.info(f"Using custom offsets: {offsets}")
            manipulation_handler.update_offsets(
                offset_x=offsets.get("offset_x"),
                offset_y=offsets.get("offset_y"),
                offset_z=offsets.get("offset_z"),
            )
        else:
            self.logger.info(
                f"Using default offsets: x={self.offset_x}, y={self.offset_y}, z={self.offset_z}"
            )

        for action, object_to_detect in commands:
            rgb_image = self.robot_interface.get_latest_rgb_image()
            depth_image = self.robot_interface.get_latest_depth_image()

            if rgb_image is None or depth_image is None:
                self.logger.error("No image messages received. Aborting command chain.")
                return

            if action not in AVAILABLE_ACTIONS:
                self.logger.error(
                    f"Action: {action} is not valid. Valid actions: {AVAILABLE_ACTIONS}"
                )
                return

            self.logger.info(
                f"Executing action: '{action}' on object: '{object_to_detect}'"
            )
            # Use the verbose action description with object name placeholder
            if self.trajectory_collector:
                action_description = ACTION_DESCRIPTIONS.get(action, action)
                formatted_description = action_description.format(
                    object_name=object_to_detect
                )
                self.trajectory_collector.record_current_prompt(formatted_description)

            # Handle the command and update object_in_gripper status
            self._object_in_gripper = self.command_processor.handle_tool_call(
                action,
                object_to_detect,
                self.object_detector,
                self.robot_interface,
                manipulation_handler,
                rgb_image,
                depth_image,
                self._object_in_gripper,
            )

        self.robot_interface.go_home()

        # Stop RL data collection and log summary
        if self.trajectory_collector:
            self.trajectory_collector.stop_episode()
            self.trajectory_collector.log_trajectory_summary()


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
