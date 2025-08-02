import time
import json
import os
import cv2
from typing import List, Dict, Any, Optional
import numpy as np
from sensor_msgs.msg import Image, JointState
from rclpy.node import Node
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose
from sortedcontainers import SortedDict


class TrajectoryDataCollector:
    """
    Collects and logs trajectory data, camera images, and prompts for RL training.

    This class is responsible for:
    - Recording robot joint states during trajectory execution
    - Capturing camera images (RGB and depth)
    - Logging command prompts and action sequences
    - Storing timestamped data for RL training
    """

    def __init__(
        self,
        logger,
        arm_joint_names: List[str],
        gripper_joint_names: List[str],
        robot_controller=None,
        save_directory: str = "rl_training_data",
    ):
        """
        Initialize the trajectory data collector.

        Args:
            logger: ROS logger instance
            arm_joint_names: List of arm joint names to track
            gripper_joint_names: List of gripper joint names to track
            robot_controller: RobotController instance for FK computation
            save_directory: Directory to save collected data
        """
        self.logger = logger
        self.arm_joint_names = arm_joint_names
        self.gripper_joint_names = gripper_joint_names
        self.robot_controller = robot_controller
        self.save_directory = save_directory
        self.cv_bridge = CvBridge()

        # Data collection state
        self.is_collecting = False
        self.current_episode_data = {}
        self.episode_id = None
        self.episode_directory = None

        # Synchronized data buffers using SortedDict for O(log n) operations
        self.joint_state_buffer = SortedDict()
        self.rgb_buffer = SortedDict()
        self.depth_buffer = SortedDict()
        self.pose_buffer = SortedDict()
        self.sync_tolerance = 0.05  # 50ms tolerance for synchronization

        # Create save directory
        os.makedirs(self.save_directory, exist_ok=True)

    def start_episode(self, prompt: str, episode_id: Optional[str] = None) -> str:
        """
        Start a new data collection episode.

        Args:
            prompt: The command prompt that initiated this episode
            episode_id: Optional custom episode ID, auto-generated if None

        Returns:
            The episode ID for this collection session
        """
        if self.is_collecting:
            self.logger.warn("Episode already in progress. Stopping previous episode.")
            self.stop_episode()

        self.episode_id = episode_id if episode_id else self._generate_episode_id()
        self.episode_directory = self._create_episode_directory(self.episode_id)

        self.current_episode_data = {
            "episode_id": self.episode_id,
            "prompt": prompt,
            "start_time": time.time(),
            "trajectory_data": [],
            "camera_data": [],
            "actions": [],
        }

        # Clear synchronized buffers for new episode
        self.joint_state_buffer.clear()
        self.rgb_buffer.clear()
        self.depth_buffer.clear()
        self.pose_buffer.clear()
        
        self.is_collecting = True

        self.logger.info(f"Started RL data collection for episode: {self.episode_id}")
        return self.episode_id

    def stop_episode(self) -> Dict[str, Any]:
        """
        Stop the current data collection episode and return collected data.

        Returns:
            Dictionary containing all collected episode data
        """
        if not self.is_collecting:
            self.logger.warn("No episode in progress to stop.")
            return {}

        self.is_collecting = False
        self.current_episode_data["end_time"] = time.time()
        self.current_episode_data["duration"] = (
            self.current_episode_data["end_time"]
            - self.current_episode_data["start_time"]
        )
        
        # Synchronize all collected data before saving
        self.current_episode_data["trajectory_data"] = self._synchronize_all_data()

        # Save episode data
        episode_file = self.save_episode_data()

        self.logger.info(
            f"Stopped RL data collection for episode: {self.episode_id}. "
            f"Collected {len(self.current_episode_data['trajectory_data'])} trajectory points. "
            f"Saved to: {episode_file}"
        )

        return self.current_episode_data.copy()

    def record_joint_state(self, joint_state: JointState) -> None:
        """
        Record a joint state data point during trajectory execution.

        Args:
            joint_state: Current joint state message
        """
        if not self.is_collecting:
            return

        # Extract arm joint data
        arm_positions = []
        arm_velocities = []

        for joint_name in self.arm_joint_names:
            if joint_name in joint_state.name:
                idx = joint_state.name.index(joint_name)
                arm_positions.append(joint_state.position[idx])
                arm_velocities.append(
                    joint_state.velocity[idx] if joint_state.velocity else 0.0
                )

        # Extract gripper joint data
        gripper_positions = []
        gripper_velocities = []

        for joint_name in self.gripper_joint_names:
            if joint_name in joint_state.name:
                idx = joint_state.name.index(joint_name)
                gripper_positions.append(joint_state.position[idx])
                gripper_velocities.append(
                    joint_state.velocity[idx] if joint_state.velocity else 0.0
                )

        # Only record if we have complete arm data
        if len(arm_positions) == len(self.arm_joint_names):
            ros_timestamp = joint_state.header.stamp.sec + joint_state.header.stamp.nanosec * 1e-9
            
            data_point = {
                "timestamp": time.time(),
                "ros_timestamp": ros_timestamp,
                "arm_joint_positions": arm_positions,
                "arm_joint_velocities": arm_velocities,
                "gripper_joint_positions": gripper_positions,
                "gripper_joint_velocities": gripper_velocities,
            }
            
            # Store in synchronized buffer using ROS timestamp as key
            self.joint_state_buffer[ros_timestamp] = data_point

    def record_rgb_image(self, rgb_image: Image) -> None:
        """
        Record RGB camera image at current timestamp.

        Args:
            rgb_image: RGB camera image message
        """
        if not self.is_collecting:
            return

        try:
            timestamp = time.time()
            ros_timestamp = rgb_image.header.stamp.sec + rgb_image.header.stamp.nanosec * 1e-9

            # Convert RGB image to bytes
            rgb_cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_image, "bgr8")
            _, rgb_encoded = cv2.imencode(".jpg", rgb_cv_image)
            rgb_bytes = rgb_encoded.tobytes()

            rgb_data_point = {
                "timestamp": timestamp,
                "ros_timestamp": ros_timestamp,
                "rgb_image_bytes": rgb_bytes.hex(),  # Convert to hex string for JSON serialization
                "image_width": rgb_image.width,
                "image_height": rgb_image.height,
                "rgb_encoding": rgb_image.encoding,
                "data_type": "rgb",
            }

            # Store in synchronized buffer using ROS timestamp as key
            self.rgb_buffer[ros_timestamp] = rgb_data_point

        except Exception as e:
            self.logger.error(f"Failed to record RGB image data: {e}")

    def record_depth_image(self, depth_image: Image) -> None:
        """
        Record depth camera image at current timestamp.

        Args:
            depth_image: Depth camera image message
        """
        if not self.is_collecting:
            return

        try:
            timestamp = time.time()
            ros_timestamp = depth_image.header.stamp.sec + depth_image.header.stamp.nanosec * 1e-9

            # Convert depth image to bytes
            depth_cv_image = self.cv_bridge.imgmsg_to_cv2(depth_image, "passthrough")
            _, depth_encoded = cv2.imencode(".png", depth_cv_image)
            depth_bytes = depth_encoded.tobytes()

            depth_data_point = {
                "timestamp": timestamp,
                "ros_timestamp": ros_timestamp,
                "depth_image_bytes": depth_bytes.hex(),
                "image_width": depth_image.width,
                "image_height": depth_image.height,
                "depth_encoding": depth_image.encoding,
                "data_type": "depth",
            }

            # Store in synchronized buffer using ROS timestamp as key
            self.depth_buffer[ros_timestamp] = depth_data_point

        except Exception as e:
            self.logger.error(f"Failed to record depth image data: {e}")

    def record_action(self, action_type: str, object_name: str, **kwargs) -> None:
        """
        Record an action being executed.

        Args:
            action_type: Type of action (e.g., 'pick_object', 'release_above')
            object_name: Name of target object
            **kwargs: Additional action-specific parameters
        """
        if not self.is_collecting:
            return

        action_data = {
            "timestamp": time.time(),
            "action_type": action_type,
            "object_name": object_name,
            "parameters": kwargs,
        }

        self.current_episode_data["actions"].append(action_data)
        self.logger.info(f"Recorded action: {action_type} on {object_name}")

    def record_pose(self, joint_state: JointState) -> None:
        """
        Record end effector pose calculated from joint and gripper states.

        Args:
            joint_state: Current joint state message containing arm and gripper positions
        """
        if not self.is_collecting or not self.robot_controller:
            return

        # Extract arm joint positions for FK computation
        arm_positions = []
        for joint_name in self.arm_joint_names:
            if joint_name in joint_state.name:
                idx = joint_state.name.index(joint_name)
                arm_positions.append(joint_state.position[idx])

        # Extract gripper joint positions
        gripper_positions = []
        for joint_name in self.gripper_joint_names:
            if joint_name in joint_state.name:
                idx = joint_state.name.index(joint_name)
                gripper_positions.append(joint_state.position[idx])

        # Only compute pose if we have complete arm joint data
        if len(arm_positions) == len(self.arm_joint_names):
            try:
                # Compute end effector pose using forward kinematics
                end_effector_pose = self.robot_controller.compute_end_effector_pose(
                    arm_positions
                )

                if end_effector_pose is not None:
                    pose_dict = {
                        "timestamp": time.time(),
                        "ros_timestamp": joint_state.header.stamp.sec
                        + joint_state.header.stamp.nanosec * 1e-9,
                        "pose_type": "end_effector",
                        "position": {
                            "x": end_effector_pose.position.x,
                            "y": end_effector_pose.position.y,
                            "z": end_effector_pose.position.z,
                        },
                        "orientation": {
                            "x": end_effector_pose.orientation.x,
                            "y": end_effector_pose.orientation.y,
                            "z": end_effector_pose.orientation.z,
                            "w": end_effector_pose.orientation.w,
                        },
                        "arm_joint_positions": arm_positions,
                        "gripper_joint_positions": gripper_positions,
                    }

                    # Store in synchronized buffer using ROS timestamp as key
                    ros_timestamp = joint_state.header.stamp.sec + joint_state.header.stamp.nanosec * 1e-9
                    self.pose_buffer[ros_timestamp] = pose_dict

            except Exception as e:
                self.logger.error(f"Failed to compute end effector pose: {e}")

    def save_episode_data(self) -> str:
        """
        Save the current episode data to disk.

        Returns:
            Path to the saved data file
        """
        if not self.episode_directory or not self.current_episode_data:
            self.logger.error("No episode data to save")
            return ""

        # Save main episode data as JSON
        episode_file = os.path.join(self.episode_directory, "episode_data.json")

        try:
            with open(episode_file, "w") as f:
                json.dump(self.current_episode_data, f, indent=2, default=str)

            # Save trajectory data separately for easier loading
            trajectory_file = os.path.join(
                self.episode_directory, "trajectory_data.json"
            )
            with open(trajectory_file, "w") as f:
                json.dump(self.current_episode_data["trajectory_data"], f, indent=2, default=str)

            return episode_file

        except Exception as e:
            self.logger.error(f"Failed to save episode data: {e}")
            return ""

    def get_episode_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current episode data.

        Returns:
            Summary statistics and metadata
        """
        if not self.current_episode_data:
            return {}

        trajectory_data = self.current_episode_data.get("trajectory_data", [])
        summary = {
            "episode_id": self.episode_id,
            "prompt": self.current_episode_data.get("prompt", ""),
            "num_trajectory_points": len(trajectory_data),
            "num_camera_frames": len(self.current_episode_data.get("camera_data", [])),
            "num_actions": len(self.current_episode_data.get("actions", [])),
            "duration": self.current_episode_data.get("duration", 0),
            "is_collecting": self.is_collecting,
        }

        if trajectory_data:
            summary["trajectory_start_time"] = trajectory_data[0]["timestamp"]
            summary["trajectory_end_time"] = trajectory_data[-1]["timestamp"]

        return summary

    def _generate_episode_id(self) -> str:
        """Generate a unique episode ID based on timestamp."""
        return f"episode_{int(time.time() * 1000)}"

    def _create_episode_directory(self, episode_id: str) -> str:
        """Create directory for episode data."""
        episode_dir = os.path.join(self.save_directory, episode_id)
        os.makedirs(episode_dir, exist_ok=True)
        return episode_dir

    def _find_closest_in_buffer(self, target_timestamp: float, 
                               sorted_buffer: SortedDict) -> Optional[dict]:
        """
        Find closest data point in O(log n) time using SortedDict.
        
        Args:
            target_timestamp: Target ROS timestamp to find closest match for
            sorted_buffer: SortedDict containing timestamped data
            
        Returns:
            Closest data point within sync_tolerance, or None if no match
        """
        if not sorted_buffer:
            return None
            
        # Find insertion index using binary search
        idx = sorted_buffer.bisect_left(target_timestamp)
        
        candidates = []
        
        # Check timestamp at/after target
        if idx < len(sorted_buffer):
            candidates.append(sorted_buffer.peekitem(idx)[0])
            
        # Check timestamp before target
        if idx > 0:
            candidates.append(sorted_buffer.peekitem(idx - 1)[0])
        
        if not candidates:
            return None
            
        # Find closest among candidates
        closest_timestamp = min(candidates, 
                              key=lambda t: abs(t - target_timestamp))
        
        # Check if within tolerance
        if abs(closest_timestamp - target_timestamp) <= self.sync_tolerance:
            return sorted_buffer[closest_timestamp]
        
        return None

    def _synchronize_all_data(self) -> List[Dict]:
        """
        Synchronize all buffered data by ROS timestamp.
        
        Uses joint states as the primary timeline and finds closest
        RGB, depth, and pose data within sync_tolerance.
        
        Returns:
            List of synchronized data points
        """
        synchronized_data = []
        
        # Use joint states as the primary timeline
        for joint_timestamp in self.joint_state_buffer.keys():
            # Start with joint state data
            data_point = self.joint_state_buffer[joint_timestamp].copy()
            
            # Find closest RGB image within tolerance (O(log n))
            rgb_data = self._find_closest_in_buffer(joint_timestamp, self.rgb_buffer)
            if rgb_data:
                data_point['rgb_data'] = rgb_data
                
            # Find closest depth image within tolerance (O(log n))
            depth_data = self._find_closest_in_buffer(joint_timestamp, self.depth_buffer)
            if depth_data:
                data_point['depth_data'] = depth_data
                
            # Find closest pose within tolerance (O(log n))
            pose_data = self._find_closest_in_buffer(joint_timestamp, self.pose_buffer)
            if pose_data:
                data_point['pose_data'] = pose_data
                
            synchronized_data.append(data_point)
            
        self.logger.info(f"Synchronized {len(synchronized_data)} data points from buffers: "
                        f"joint_states={len(self.joint_state_buffer)}, "
                        f"rgb={len(self.rgb_buffer)}, "
                        f"depth={len(self.depth_buffer)}, "
                        f"poses={len(self.pose_buffer)}")
            
        return synchronized_data
