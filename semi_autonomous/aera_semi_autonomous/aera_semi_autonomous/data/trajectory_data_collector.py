import time
import json
import os

import cv2
from typing import List, Dict, Any, Optional
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge
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
        save_directory: str = "rl_training_data",
        sync_tolerance: float = 0.05,
    ):
        """
        Initialize the trajectory data collector.

        Args:
            logger: ROS logger instance
            arm_joint_names: List of arm joint names to track
            gripper_joint_names: List of gripper joint names to track
            save_directory: Directory to save collected data
            sync_tolerance: Time tolerance in seconds for data synchronization
        """
        self.logger = logger
        self.arm_joint_names = arm_joint_names
        self.gripper_joint_names = gripper_joint_names
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
        self.sync_tolerance = sync_tolerance

        # Synchronization statistics
        self.sync_stats = {
            "rgb_discrepancies": [],
            "depth_discrepancies": [],
            "pose_discrepancies": [],
            "failed_syncs": {"rgb": 0, "depth": 0, "current_pose": 0, "next_pose": 0},
        }

        # Create save directory
        os.makedirs(self.save_directory, exist_ok=True)

    def start_episode(
        self, input_message: str, episode_id: Optional[str] = None
    ) -> str:
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
        self.current_prompt = None

        self.current_episode_data = {
            "episode_id": self.episode_id,
            "input_message": input_message,
            "start_time": time.time(),
            "trajectory_data": [],
            "metadata": {
                "image_width": None,
                "image_height": None,
                "rgb_encoding": None,
                "depth_encoding": None,
            },
        }

        # Clear synchronized buffers for new episode
        self.joint_state_buffer.clear()
        self.rgb_buffer.clear()
        self.depth_buffer.clear()
        self.pose_buffer.clear()

        # Reset synchronization statistics
        self.sync_stats = {
            "rgb_discrepancies": [],
            "depth_discrepancies": [],
            "pose_discrepancies": [],
            "failed_syncs": {"rgb": 0, "depth": 0, "current_pose": 0, "next_pose": 0},
        }

        self.is_collecting = True

        self.logger.info(f"Started RL data collection for episode: {self.episode_id}")
        return self.episode_id

    def stop_episode(self, save_data: bool = True) -> None:
        """
        Stop the current data collection episode and return collected data.

        Returns:
            Dictionary containing all collected episode data
        """
        if not self.is_collecting:
            self.logger.warn("No episode in progress to stop.")
            return

        self.is_collecting = False
        self.current_episode_data["end_time"] = time.time()
        self.current_episode_data["duration"] = (
            self.current_episode_data["end_time"]
            - self.current_episode_data["start_time"]
        )

        # Synchronize all collected data before saving
        self.current_episode_data["trajectory_data"] = self._synchronize_all_data()

        # Add synchronization statistics to episode data
        self.current_episode_data["synchronization_stats"] = (
            self._compute_sync_statistics()
        )

        self.logger.info(f"Stopped RL data collection for episode: {self.episode_id}.")
        if save_data:
            episode_file = self.save_episode_data()
            self.logger.info(
                f"Saved episode data to: {episode_file}"
                f"Collected {len(self.current_episode_data['trajectory_data'])} trajectory points. "
            )
        else:
            self.logger.info(f"Eposied data for episode: {self.episode_id} not saved")

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
                idx = list(joint_state.name).index(joint_name)
                arm_positions.append(joint_state.position[idx])
                arm_velocities.append(
                    joint_state.velocity[idx] if joint_state.velocity else 0.0
                )

        # Extract gripper joint data
        gripper_positions = []
        gripper_velocities = []

        for joint_name in self.gripper_joint_names:
            if joint_name in joint_state.name:
                idx = list(joint_state.name).index(joint_name)
                gripper_positions.append(joint_state.position[idx])
                gripper_velocities.append(
                    joint_state.velocity[idx] if joint_state.velocity else 0.0
                )

        # Only record if we have complete arm data
        if len(arm_positions) == len(self.arm_joint_names):
            ros_timestamp = (
                joint_state.header.stamp.sec + joint_state.header.stamp.nanosec * 1e-9
            )

            data_point = {
                "timestamp": time.time(),
                "ros_timestamp": ros_timestamp,
                "arm_joint_positions": arm_positions,
                "arm_joint_velocities": arm_velocities,
                "gripper_joint_positions": gripper_positions,
                "gripper_joint_velocities": gripper_velocities,
                "prompt": self.current_prompt,
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
            ros_timestamp = (
                rgb_image.header.stamp.sec + rgb_image.header.stamp.nanosec * 1e-9
            )

            # Convert RGB image to bytes
            rgb_cv_image = self.cv_bridge.imgmsg_to_cv2(rgb_image, "bgr8")
            _, rgb_encoded = cv2.imencode(".jpg", rgb_cv_image)
            rgb_bytes = rgb_encoded.tobytes()

            # Store metadata at episode level if not already set
            if self.current_episode_data["metadata"]["image_width"] is None:
                self.current_episode_data["metadata"]["image_width"] = rgb_image.width
                self.current_episode_data["metadata"]["image_height"] = rgb_image.height
                self.current_episode_data["metadata"]["rgb_encoding"] = (
                    rgb_image.encoding
                )

            rgb_data_point = {
                "timestamp": timestamp,
                "ros_timestamp": ros_timestamp,
                "rgb_image_bytes": rgb_bytes.hex(),  # Convert to hex string for JSON serialization
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
            ros_timestamp = (
                depth_image.header.stamp.sec + depth_image.header.stamp.nanosec * 1e-9
            )

            # Convert depth image to bytes
            depth_cv_image = self.cv_bridge.imgmsg_to_cv2(depth_image, "passthrough")
            _, depth_encoded = cv2.imencode(".png", depth_cv_image)
            depth_bytes = depth_encoded.tobytes()

            # Store metadata at episode level if not already set
            if self.current_episode_data["metadata"]["depth_encoding"] is None:
                self.current_episode_data["metadata"]["depth_encoding"] = (
                    depth_image.encoding
                )

            depth_data_point = {
                "timestamp": timestamp,
                "ros_timestamp": ros_timestamp,
                "depth_image_bytes": depth_bytes.hex(),
                "data_type": "depth",
            }

            # Store in synchronized buffer using ROS timestamp as key
            self.depth_buffer[ros_timestamp] = depth_data_point

        except Exception as e:
            self.logger.error(f"Failed to record depth image data: {e}")

    def record_pose(self, end_effector_pose, ros_timestamp: float) -> None:
        """
        Record end effector pose.

        Args:
            end_effector_pose: The computed end effector pose.
            ros_timestamp: The ROS timestamp for the pose.
        """
        if not self.is_collecting:
            return

        if end_effector_pose is not None:
            try:
                pose_dict = {
                    "timestamp": time.time(),
                    "ros_timestamp": ros_timestamp,
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
                }

                # Store in synchronized buffer using ROS timestamp as key
                self.pose_buffer[ros_timestamp] = pose_dict

            except Exception as e:
                self.logger.error(f"Failed to record end effector pose: {e}")

    def record_current_prompt(self, prompt: str) -> None:
        """
        Record the current prompt for the episode.

        Args:
            prompt: Current prompt for the episode
        """
        if not self.is_collecting:
            return
        self.current_prompt = prompt

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

            return episode_file

        except Exception as e:
            self.logger.error(f"Failed to save episode data: {e}")
            return ""

    def _generate_episode_id(self) -> str:
        """Generate a unique episode ID based on timestamp."""
        return f"episode_{int(time.time() * 1000)}"

    def _create_episode_directory(self, episode_id: str) -> str:
        """Create directory for episode data."""
        episode_dir = os.path.join(self.save_directory, episode_id)
        os.makedirs(episode_dir, exist_ok=True)
        return episode_dir

    def _find_closest_in_buffer(
        self,
        target_timestamp: float,
        sorted_buffer: SortedDict,
        data_type: str = "unknown",
    ) -> Optional[dict]:
        """
        Find closest data point in O(log n) time using SortedDict.

        Args:
            target_timestamp: Target ROS timestamp to find closest match for
            sorted_buffer: SortedDict containing timestamped data
            data_type: Type of data for statistics tracking

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
        closest_timestamp = min(candidates, key=lambda t: abs(t - target_timestamp))
        discrepancy = abs(closest_timestamp - target_timestamp)

        # Check if within tolerance
        if discrepancy <= self.sync_tolerance:
            # Record synchronization statistics
            if data_type == "rgb":
                self.sync_stats["rgb_discrepancies"].append(discrepancy)
            elif data_type == "depth":
                self.sync_stats["depth_discrepancies"].append(discrepancy)
            elif data_type == "pose":
                self.sync_stats["pose_discrepancies"].append(discrepancy)

            return sorted_buffer[closest_timestamp]

        # Record failed synchronization
        if data_type in self.sync_stats["failed_syncs"]:
            self.sync_stats["failed_syncs"][data_type] += 1

        self.logger.warn(
            f"Could not find {data_type} data within {self.sync_tolerance} tolerance for timestamp: {target_timestamp}, the closest was: {closest_timestamp} (discrepancy: {discrepancy:.4f}s)"
        )
        return None

    def _synchronize_all_data(self) -> List[Dict]:
        """
        Synchronize all buffered data by ROS timestamp and format for RL training.

        Uses joint states as the primary timeline and finds closest
        RGB, depth, and pose data within sync_tolerance.

        Formats data as observation-action pairs where:
        - observations: current state (joint positions, gripper state, cartesian pose, images)
        - action: next state (joint positions/velocities, gripper state, cartesian pose/velocities)

        Returns:
            List of RL-formatted data points with observations and actions
        """
        synchronized_data = []
        joint_timestamps = list(self.joint_state_buffer.keys())

        # Group timestamps by prompt to determine episode boundaries
        prompt_groups = self._group_timestamps_by_prompt(joint_timestamps)

        # Process all timestamps except the last one (since we need next state for action)
        for i in range(len(joint_timestamps) - 1):
            current_timestamp = joint_timestamps[i]
            next_timestamp = joint_timestamps[i + 1]

            # Get current state data for observations
            current_joint_data = self.joint_state_buffer[current_timestamp]

            # Get next state data for actions
            next_joint_data = self.joint_state_buffer[next_timestamp]

            # Find closest RGB image within tolerance for current timestamp
            rgb_data = self._find_closest_in_buffer(
                current_timestamp, self.rgb_buffer, "rgb"
            )

            # Find closest depth image within tolerance for current timestamp
            depth_data = self._find_closest_in_buffer(
                current_timestamp, self.depth_buffer, "depth"
            )

            # Find closest pose within tolerance for current and next timestamps
            current_pose_data = self._find_closest_in_buffer(
                current_timestamp, self.pose_buffer, "pose"
            )
            next_pose_data = self._find_closest_in_buffer(
                next_timestamp, self.pose_buffer, "pose"
            )

            # Skip if we don't have essential data
            if (
                not rgb_data
                or not depth_data
                or not current_pose_data
                or not next_pose_data
            ):
                continue

            # Calculate cartesian velocities
            dt = next_timestamp - current_timestamp
            cartesian_velocity = self._calculate_cartesian_velocity(
                current_pose_data, next_pose_data, dt
            )

            # Determine episode flags
            current_prompt = current_joint_data.get("prompt")
            is_first = self._is_first_in_prompt_group(
                current_timestamp, current_prompt, prompt_groups
            )
            is_last = self._is_last_in_prompt_group(
                current_timestamp, current_prompt, prompt_groups
            )
            is_terminal = i == len(joint_timestamps) - 2  # Last possible data point
            default_reward = 1.0 if is_last else 0.0

            # Format as RL observation-action pair
            rl_data_point = {
                "prompt": current_prompt,
                "is_first": is_first,
                "is_last": is_last,
                "is_terminal": is_terminal,
                "default_reward": default_reward,
                "observations": {
                    "joint_state": current_joint_data["arm_joint_positions"],
                    "gripper_state": current_joint_data["gripper_joint_positions"],
                    "cartesian_position": {
                        "position": current_pose_data["position"],
                        "orientation": current_pose_data["orientation"],
                    },
                    "rgb_image": rgb_data["rgb_image_bytes"],
                    "depth_image": depth_data["depth_image_bytes"],
                    "timestamp": current_timestamp,
                },
                "action": {
                    "joint_state": next_joint_data["arm_joint_positions"],
                    "joint_velocities": next_joint_data["arm_joint_velocities"],
                    "gripper_state": next_joint_data["gripper_joint_positions"],
                    "gripper_velocities": next_joint_data["gripper_joint_velocities"],
                    "cartesian_position": {
                        "position": next_pose_data["position"],
                        "orientation": next_pose_data["orientation"],
                    },
                    "cartesian_velocity": cartesian_velocity,
                    "timestamp": next_timestamp,
                },
            }

            synchronized_data.append(rl_data_point)

        self.logger.info(
            f"Synchronized {len(synchronized_data)} RL data points from buffers: "
            f"joint_states={len(self.joint_state_buffer)}, "
            f"rgb={len(self.rgb_buffer)}, "
            f"depth={len(self.depth_buffer)}, "
            f"poses={len(self.pose_buffer)}"
        )

        return synchronized_data

    def _calculate_cartesian_velocity(
        self, current_pose: Dict, next_pose: Dict, dt: float
    ) -> Dict:
        """
        Calculate cartesian velocity between two poses.

        Args:
            current_pose: Current pose data with position and orientation
            next_pose: Next pose data with position and orientation
            dt: Time difference between poses

        Returns:
            Dictionary with linear and angular velocities
        """
        if dt <= 0:
            return {
                "linear": {"x": 0.0, "y": 0.0, "z": 0.0},
                "angular": {"x": 0.0, "y": 0.0, "z": 0.0},
            }

        # Calculate linear velocity
        linear_vel = {
            "x": (next_pose["position"]["x"] - current_pose["position"]["x"]) / dt,
            "y": (next_pose["position"]["y"] - current_pose["position"]["y"]) / dt,
            "z": (next_pose["position"]["z"] - current_pose["position"]["z"]) / dt,
        }

        # For angular velocity, we'd need to compute quaternion difference
        # For now, using a simplified approach - could be enhanced with proper quaternion math
        angular_vel = {
            "x": (next_pose["orientation"]["x"] - current_pose["orientation"]["x"])
            / dt,
            "y": (next_pose["orientation"]["y"] - current_pose["orientation"]["y"])
            / dt,
            "z": (next_pose["orientation"]["z"] - current_pose["orientation"]["z"])
            / dt,
        }

        return {"linear": linear_vel, "angular": angular_vel}

    def _group_timestamps_by_prompt(
        self, timestamps: List[float]
    ) -> Dict[str, List[float]]:
        """
        Group timestamps by their associated prompt.

        Args:
            timestamps: List of timestamps to group

        Returns:
            Dictionary mapping prompt to list of timestamps
        """
        prompt_groups = {}

        for timestamp in timestamps:
            joint_data = self.joint_state_buffer.get(timestamp)
            if joint_data:
                prompt = joint_data.get("prompt")
                if prompt not in prompt_groups:
                    prompt_groups[prompt] = []
                prompt_groups[prompt].append(timestamp)

        return prompt_groups

    def _is_first_in_prompt_group(
        self, timestamp: float, prompt: str, prompt_groups: Dict[str, List[float]]
    ) -> bool:
        """
        Check if timestamp is the first in its prompt group.

        Args:
            timestamp: Current timestamp
            prompt: Current prompt
            prompt_groups: Dictionary of prompt groups

        Returns:
            True if this is the first timestamp for the given prompt
        """
        if prompt not in prompt_groups or not prompt_groups[prompt]:
            return False

        return timestamp == min(prompt_groups[prompt])

    def _is_last_in_prompt_group(
        self, timestamp: float, prompt: str, prompt_groups: Dict[str, List[float]]
    ) -> bool:
        """
        Check if timestamp is the last in its prompt group.

        Args:
            timestamp: Current timestamp
            prompt: Current prompt
            prompt_groups: Dictionary of prompt groups

        Returns:
            True if this is the last timestamp for the given prompt
        """
        if prompt not in prompt_groups or not prompt_groups[prompt]:
            return False

        return timestamp == max(prompt_groups[prompt])

    def _compute_sync_statistics(self) -> Dict[str, Any]:
        """
        Compute synchronization statistics for the episode.

        Returns:
            Dictionary containing synchronization statistics
        """
        stats = {
            "sync_tolerance_used": self.sync_tolerance,
            "total_failed_syncs": sum(self.sync_stats["failed_syncs"].values()),
            "failed_syncs_by_type": self.sync_stats["failed_syncs"].copy(),
        }

        # Compute RGB synchronization statistics
        if self.sync_stats["rgb_discrepancies"]:
            rgb_discrepancies = self.sync_stats["rgb_discrepancies"]
            stats["rgb_sync"] = {
                "count": len(rgb_discrepancies),
                "mean_discrepancy": sum(rgb_discrepancies) / len(rgb_discrepancies),
                "max_discrepancy": max(rgb_discrepancies),
                "min_discrepancy": min(rgb_discrepancies),
            }
        else:
            stats["rgb_sync"] = {"count": 0}

        # Compute depth synchronization statistics
        if self.sync_stats["depth_discrepancies"]:
            depth_discrepancies = self.sync_stats["depth_discrepancies"]
            stats["depth_sync"] = {
                "count": len(depth_discrepancies),
                "mean_discrepancy": sum(depth_discrepancies) / len(depth_discrepancies),
                "max_discrepancy": max(depth_discrepancies),
                "min_discrepancy": min(depth_discrepancies),
            }
        else:
            stats["depth_sync"] = {"count": 0}

        # Compute pose synchronization statistics
        if self.sync_stats["pose_discrepancies"]:
            pose_discrepancies = self.sync_stats["pose_discrepancies"]
            stats["pose_sync"] = {
                "count": len(pose_discrepancies),
                "mean_discrepancy": sum(pose_discrepancies) / len(pose_discrepancies),
                "max_discrepancy": max(pose_discrepancies),
                "min_discrepancy": min(pose_discrepancies),
            }
        else:
            stats["pose_sync"] = {"count": 0}

        return stats

    def summarize_trajectory_data(self) -> Dict[str, Any]:
        """
        Summarize the most important facts about the recorded trajectory data.

        Returns:
            Dictionary containing trajectory summary statistics
        """
        trajectory_data = self.current_episode_data.get("trajectory_data", [])

        if not trajectory_data:
            return {"error": "No trajectory data to summarize"}

        # Basic trajectory statistics
        total_points = len(trajectory_data)

        # Extract prompts and episode structure
        prompts = [
            point.get("prompt") for point in trajectory_data if point.get("prompt")
        ]
        unique_prompts = list(set(prompts))

        # Calculate trajectory duration
        if total_points >= 2:
            start_time = trajectory_data[0]["observations"]["timestamp"]
            end_time = trajectory_data[-1]["action"]["timestamp"]
            duration = end_time - start_time
        else:
            duration = 0.0

        # Analyze joint movement ranges
        joint_positions = [
            point["observations"]["joint_state"] for point in trajectory_data
        ]
        if joint_positions:
            joint_ranges = []
            for joint_idx in range(len(joint_positions[0])):
                joint_values = [pos[joint_idx] for pos in joint_positions]
                joint_range = max(joint_values) - min(joint_values)
                joint_ranges.append(joint_range)
        else:
            joint_ranges = []

        # Analyze gripper state changes
        gripper_states = [
            point["observations"]["gripper_state"] for point in trajectory_data
        ]
        gripper_changes = 0
        if len(gripper_states) > 1:
            for i in range(1, len(gripper_states)):
                if gripper_states[i] != gripper_states[i - 1]:
                    gripper_changes += 1

        # Analyze cartesian movement
        positions = [
            point["observations"]["cartesian_position"]["position"]
            for point in trajectory_data
        ]
        if positions:
            # Calculate total cartesian distance traveled
            total_distance = 0.0
            for i in range(1, len(positions)):
                dx = positions[i]["x"] - positions[i - 1]["x"]
                dy = positions[i]["y"] - positions[i - 1]["y"]
                dz = positions[i]["z"] - positions[i - 1]["z"]
                distance = (dx**2 + dy**2 + dz**2) ** 0.5
                total_distance += distance

            # Calculate workspace bounds
            x_coords = [pos["x"] for pos in positions]
            y_coords = [pos["y"] for pos in positions]
            z_coords = [pos["z"] for pos in positions]

            workspace_bounds = {
                "x_range": max(x_coords) - min(x_coords),
                "y_range": max(y_coords) - min(y_coords),
                "z_range": max(z_coords) - min(z_coords),
                "min_position": {
                    "x": min(x_coords),
                    "y": min(y_coords),
                    "z": min(z_coords),
                },
                "max_position": {
                    "x": max(x_coords),
                    "y": max(y_coords),
                    "z": max(z_coords),
                },
            }
        else:
            total_distance = 0.0
            workspace_bounds = {}

        # Count episode boundaries
        first_points = sum(
            1 for point in trajectory_data if point.get("is_first", False)
        )
        last_points = sum(1 for point in trajectory_data if point.get("is_last", False))
        terminal_points = sum(
            1 for point in trajectory_data if point.get("is_terminal", False)
        )

        # Calculate average velocities
        joint_velocities = [
            point["action"]["joint_velocities"]
            for point in trajectory_data
            if point["action"]["joint_velocities"]
        ]
        if joint_velocities:
            avg_joint_velocities = []
            for joint_idx in range(len(joint_velocities[0])):
                joint_vels = [abs(vel[joint_idx]) for vel in joint_velocities]
                avg_joint_velocities.append(sum(joint_vels) / len(joint_vels))
        else:
            avg_joint_velocities = []

        summary = {
            "trajectory_overview": {
                "total_data_points": total_points,
                "duration_seconds": round(duration, 3),
                "unique_prompts": len(unique_prompts),
                "prompts_executed": unique_prompts,
            },
            "movement_analysis": {
                "total_cartesian_distance_meters": round(total_distance, 4),
                "joint_movement_ranges_radians": [round(r, 4) for r in joint_ranges],
                "max_joint_movement_radians": round(max(joint_ranges), 4)
                if joint_ranges
                else 0.0,
                "average_joint_velocities_rad_per_sec": [
                    round(v, 4) for v in avg_joint_velocities
                ],
                "workspace_bounds_meters": workspace_bounds,
            },
            "manipulation_analysis": {
                "gripper_state_changes": gripper_changes,
                "episode_boundaries": {
                    "first_points": first_points,
                    "last_points": last_points,
                    "terminal_points": terminal_points,
                },
            },
            "data_quality": {
                "average_frequency_hz": round(total_points / duration, 2)
                if duration > 0
                else 0.0,
                "has_complete_observations": all(
                    point.get("observations", {}).get("rgb_image")
                    and point.get("observations", {}).get("depth_image")
                    and point.get("observations", {}).get("joint_state")
                    for point in trajectory_data
                ),
                "has_complete_actions": all(
                    point.get("action", {}).get("joint_state")
                    and point.get("action", {}).get("cartesian_position")
                    for point in trajectory_data
                ),
            },
        }

        return summary

    def log_trajectory_summary(self) -> None:
        """
        Log a formatted summary of the trajectory data to the logger.
        """
        if not self.current_episode_data:
            self.logger.warn("No episode data available for summary")
            return

        trajectory_summary = self.summarize_trajectory_data()

        if "error" in trajectory_summary:
            self.logger.warn(f"Trajectory summary error: {trajectory_summary['error']}")
            return

        self.logger.info("=== TRAJECTORY SUMMARY ===")
        self.logger.info(
            f"Episode: {self.current_episode_data.get('episode_id', 'unknown')}"
        )
        self.logger.info(
            f"Input: {self.current_episode_data.get('input_message', 'unknown')}"
        )

        # Log overview
        overview = trajectory_summary.get("trajectory_overview", {})
        self.logger.info(f"Data Points: {overview.get('total_data_points', 0)}")
        self.logger.info(f"Duration: {overview.get('duration_seconds', 0)}s")
        self.logger.info(f"Prompts: {overview.get('unique_prompts', 0)} unique")

        # Log movement analysis
        movement = trajectory_summary.get("movement_analysis", {})
        self.logger.info(
            f"Distance Traveled: {movement.get('total_cartesian_distance_meters', 0)}m"
        )
        self.logger.info(
            f"Max Joint Movement: {movement.get('max_joint_movement_radians', 0)} rad"
        )

        # Log manipulation analysis
        manipulation = trajectory_summary.get("manipulation_analysis", {})
        self.logger.info(
            f"Gripper Changes: {manipulation.get('gripper_state_changes', 0)}"
        )

        # Log data quality
        quality = trajectory_summary.get("data_quality", {})
        self.logger.info(f"Frequency: {quality.get('average_frequency_hz', 0)} Hz")
        self.logger.info(
            f"Complete Data: {quality.get('has_complete_observations', False) and quality.get('has_complete_actions', False)}"
        )
        self.logger.info("=== END SUMMARY ===")
