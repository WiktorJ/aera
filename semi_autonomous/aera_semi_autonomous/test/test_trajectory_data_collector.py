import unittest
from unittest.mock import patch, Mock
import time
import json
import os
import tempfile
import shutil
import numpy as np

# ROS message mocks
from sensor_msgs.msg import Image, JointState
from geometry_msgs.msg import Pose, Point, Quaternion
from std_msgs.msg import Header
from builtin_interfaces.msg import Time as ROSTime

from aera_semi_autonomous.data.trajectory_data_collector import TrajectoryDataCollector


class TestTrajectoryDataCollector(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.arm_joint_names = [
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
        ]
        self.gripper_joint_names = [
            "gripper_left_finger_joint",
            "gripper_right_finger_joint",
        ]

        # Create temporary directory for testing
        self.test_dir = tempfile.mkdtemp()

        self.collector = TrajectoryDataCollector(
            self.mock_logger,
            self.arm_joint_names,
            self.gripper_joint_names,
            save_directory=self.test_dir,
            sync_tolerance=0.05,
        )

    def tearDown(self):
        """Clean up test fixtures."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)

    def _create_mock_joint_state(self, timestamp_sec=10.0, timestamp_nanosec=0):
        """Create a realistic mock JointState message."""
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = ROSTime()
        joint_state.header.stamp.sec = int(timestamp_sec)
        joint_state.header.stamp.nanosec = int(
            (timestamp_sec % 1) * 1e9 + timestamp_nanosec
        )

        # Include all joints (arm + gripper + some extras)
        joint_state.name = (
            self.arm_joint_names + self.gripper_joint_names + ["extra_joint"]
        )
        joint_state.position = [0.1, 0.2, 0.3, 0.01, 0.01, 0.5]  # 6 joints total
        joint_state.velocity = [0.05, 0.1, 0.15, 0.001, 0.001, 0.02]

        return joint_state

    def _create_mock_image(
        self,
        timestamp_sec=10.0,
        timestamp_nanosec=0,
        encoding="bgr8",
        width=640,
        height=480,
    ):
        """Create a realistic mock Image message."""
        image = Image()
        image.header = Header()
        image.header.stamp = ROSTime()
        image.header.stamp.sec = int(timestamp_sec)
        image.header.stamp.nanosec = int(timestamp_nanosec)
        image.width = width
        image.height = height
        image.encoding = encoding
        image.step = width * 3 if encoding == "bgr8" else width * 2
        image.data = bytes(width * height * (3 if encoding == "bgr8" else 2))

        return image

    def _create_mock_pose(self, x=0.5, y=0.0, z=0.3):
        """Create a realistic mock Pose."""
        pose = Pose()
        pose.position = Point(x=x, y=y, z=z)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        return pose

    def test_episode_lifecycle(self):
        """Test the complete lifecycle of an episode."""
        # Test initial state
        self.assertFalse(self.collector.is_collecting)
        self.assertEqual(len(self.collector.current_episode_data), 0)
        # episode_id is initialized in __init__, not None
        self.assertIsNotNone(self.collector.episode_id)

        # Test starting an episode
        input_message = "pick up the red block"
        episode_id = self.collector.start_episode(input_message)

        self.assertTrue(self.collector.is_collecting)
        self.assertIsNotNone(episode_id)
        self.assertEqual(self.collector.episode_id, episode_id)
        self.assertEqual(
            self.collector.current_episode_data["input_message"], input_message
        )
        self.assertIn("start_time", self.collector.current_episode_data)

        # Test stopping an episode
        with patch.object(
            self.collector, "save_episode_data", return_value="test_file.json"
        ):
            self.collector.stop_episode()

        self.assertFalse(self.collector.is_collecting)
        self.assertIn("end_time", self.collector.current_episode_data)
        self.assertIn("duration", self.collector.current_episode_data)
        # episode_id is not cleared after stopping in the implementation
        self.assertIsNotNone(self.collector.episode_id)

        # Verify buffers are cleared
        self.assertEqual(len(self.collector.joint_state_buffer), 0)
        self.assertEqual(len(self.collector.rgb_buffer), 0)
        self.assertEqual(len(self.collector.depth_buffer), 0)
        self.assertEqual(len(self.collector.pose_buffer), 0)

    def test_start_episode_while_collecting(self):
        """Test starting a new episode while one is already in progress."""
        # Start first episode
        self.collector.start_episode("first episode")
        first_episode_id = self.collector.episode_id

        # Wait a small amount to ensure different timestamp
        import time

        time.sleep(0.001)

        # Start second episode (should stop first one)
        with patch.object(self.collector, "stop_episode") as mock_stop:
            second_episode_id = self.collector.start_episode("second episode")

        mock_stop.assert_called_once()
        self.assertNotEqual(first_episode_id, second_episode_id)
        self.assertEqual(
            self.collector.current_episode_data["input_message"], "second episode"
        )

    def test_stop_episode_when_not_collecting(self):
        """Test stopping an episode when none is in progress."""
        self.collector.stop_episode()
        self.mock_logger.warn.assert_called_with("No episode in progress to stop.")

    @patch("cv2.imencode")
    def test_record_joint_state(self, _):
        """Test recording joint state data."""
        self.collector.start_episode("test episode")

        joint_state = self._create_mock_joint_state(timestamp_sec=10.5)
        self.collector.record_joint_state(joint_state)

        # Verify data was recorded
        self.assertEqual(len(self.collector.joint_state_buffer), 1)

        # Check recorded data structure
        recorded_data = list(self.collector.joint_state_buffer.values())[0]
        self.assertEqual(
            len(recorded_data["arm_joint_positions"]), len(self.arm_joint_names)
        )
        self.assertEqual(
            len(recorded_data["gripper_joint_positions"]), len(self.gripper_joint_names)
        )
        self.assertEqual(recorded_data["ros_timestamp"], 10.5)

    def test_record_joint_state_incomplete_data(self):
        """Test recording joint state with missing arm joints."""
        self.collector.start_episode("test episode")

        # Create joint state with missing arm joints
        joint_state = JointState()
        joint_state.header = Header()
        joint_state.header.stamp = ROSTime()
        joint_state.header.stamp.sec = 10
        joint_state.header.stamp.nanosec = 0
        joint_state.name = ["unknown_joint"]
        joint_state.position = [0.1]
        joint_state.velocity = [0.05]

        self.collector.record_joint_state(joint_state)

        # Should not record incomplete data
        self.assertEqual(len(self.collector.joint_state_buffer), 0)

    @patch("aera_semi_autonomous.data.trajectory_data_collector.CvBridge")
    @patch("cv2.imencode")
    def test_record_rgb_image(self, mock_imencode, mock_cv_bridge):
        """Test recording RGB image data."""
        self.collector.start_episode("test episode")

        # Mock cv2.imencode to return fake encoded data
        mock_encoded = Mock()
        mock_encoded.tobytes.return_value = b"fake_image_data"
        mock_imencode.return_value = (True, mock_encoded)

        # Mock CvBridge
        mock_bridge_instance = Mock()
        mock_bridge_instance.imgmsg_to_cv2.return_value = np.zeros(
            (480, 640, 3), dtype=np.uint8
        )
        mock_cv_bridge.return_value = mock_bridge_instance

        rgb_image = self._create_mock_image(timestamp_sec=10.5, encoding="bgr8")
        self.collector.record_rgb_image(rgb_image)

        # Verify data was recorded
        self.assertEqual(len(self.collector.rgb_buffer), 1)

        # Check metadata was set
        metadata = self.collector.current_episode_data["metadata"]
        self.assertEqual(metadata["image_width"], 640)
        self.assertEqual(metadata["image_height"], 480)
        self.assertEqual(metadata["rgb_encoding"], "bgr8")

    @patch("aera_semi_autonomous.data.trajectory_data_collector.CvBridge")
    @patch("cv2.imencode")
    def test_record_depth_image(self, mock_imencode, mock_cv_bridge):
        """Test recording depth image data."""
        self.collector.start_episode("test episode")

        # Mock cv2.imencode to return fake encoded data
        mock_encoded = Mock()
        mock_encoded.tobytes.return_value = b"fake_depth_data"
        mock_imencode.return_value = (True, mock_encoded)

        # Mock CvBridge
        mock_bridge_instance = Mock()
        mock_bridge_instance.imgmsg_to_cv2.return_value = np.zeros(
            (480, 640), dtype=np.uint16
        )
        mock_cv_bridge.return_value = mock_bridge_instance

        depth_image = self._create_mock_image(timestamp_sec=10.5, encoding="16UC1")
        self.collector.record_depth_image(depth_image)

        # Verify data was recorded
        self.assertEqual(len(self.collector.depth_buffer), 1)

        # Check metadata was set
        metadata = self.collector.current_episode_data["metadata"]
        self.assertEqual(metadata["depth_encoding"], "16UC1")

    def test_record_pose(self):
        """Test recording end effector pose data."""
        self.collector.start_episode("test episode")

        pose = self._create_mock_pose(x=0.5, y=0.2, z=0.3)
        timestamp = 10.5

        self.collector.record_pose(pose, timestamp)

        # Verify data was recorded
        self.assertEqual(len(self.collector.pose_buffer), 1)

        # Check recorded data structure
        recorded_data = list(self.collector.pose_buffer.values())[0]
        self.assertEqual(recorded_data["position"]["x"], 0.5)
        self.assertEqual(recorded_data["position"]["y"], 0.2)
        self.assertEqual(recorded_data["position"]["z"], 0.3)
        self.assertEqual(recorded_data["ros_timestamp"], timestamp)

    def test_record_current_prompt(self):
        """Test recording current prompt."""
        self.collector.start_episode("test episode")

        prompt = "pick up the blue cube"
        self.collector.record_current_prompt(prompt)

        self.assertEqual(self.collector.current_prompt, prompt)

    def test_recording_when_not_collecting(self):
        """Test that recording methods do nothing when not collecting."""
        joint_state = self._create_mock_joint_state()
        rgb_image = self._create_mock_image()
        depth_image = self._create_mock_image(encoding="16UC1")
        pose = self._create_mock_pose()

        # Try recording without starting episode
        self.collector.record_joint_state(joint_state)
        self.collector.record_rgb_image(rgb_image)
        self.collector.record_depth_image(depth_image)
        self.collector.record_pose(pose, 10.0)
        self.collector.record_current_prompt("test prompt")

        # Verify nothing was recorded
        self.assertEqual(len(self.collector.joint_state_buffer), 0)
        self.assertEqual(len(self.collector.rgb_buffer), 0)
        self.assertEqual(len(self.collector.depth_buffer), 0)
        self.assertEqual(len(self.collector.pose_buffer), 0)
        self.assertIsNone(getattr(self.collector, "current_prompt", None))

    def test_find_closest_in_buffer(self):
        """Test the internal _find_closest_in_buffer helper method."""
        from sortedcontainers import SortedDict

        # Test with empty buffer
        empty_buffer = SortedDict()
        result = self.collector._find_closest_in_buffer(10.0, empty_buffer, "test")
        self.assertIsNone(result)

        # Test with populated buffer
        buffer = SortedDict(
            [
                (10.0, {"data": "a", "timestamp": 10.0}),
                (20.0, {"data": "b", "timestamp": 20.0}),
                (30.0, {"data": "c", "timestamp": 30.0}),
            ]
        )

        # Test exact match
        result = self.collector._find_closest_in_buffer(20.0, buffer, "test")
        self.assertIsNotNone(result)
        self.assertEqual(result["data"], "b")  # type: ignore

        # Test close match within tolerance (default 0.05s)
        result = self.collector._find_closest_in_buffer(20.03, buffer, "test")
        self.assertIsNotNone(result)
        self.assertEqual(result["data"], "b")  # type: ignore

        # Test match outside tolerance
        result = self.collector._find_closest_in_buffer(25.0, buffer, "test")
        self.assertIsNone(result)

        # Test edge cases
        result = self.collector._find_closest_in_buffer(
            5.0, buffer, "test"
        )  # Before first
        self.assertIsNone(result)

        result = self.collector._find_closest_in_buffer(
            35.0, buffer, "test"
        )  # After last
        self.assertIsNone(result)

        # Test closest selection between two candidates (within tolerance)
        result = self.collector._find_closest_in_buffer(10.04, buffer, "test")
        # Should pick 10.0 since it's within tolerance
        self.assertIsNotNone(result)
        self.assertEqual(result["data"], "a")  # type: ignore

        # Test case outside tolerance
        result = self.collector._find_closest_in_buffer(15.0, buffer, "test")
        # Should return None since 15.0 is 5 seconds away from nearest (outside 0.05s tolerance)
        self.assertIsNone(result)

    def test_find_closest_in_buffer_statistics(self):
        """Test that synchronization statistics are properly recorded."""
        from sortedcontainers import SortedDict

        buffer = SortedDict([(10.0, {"data": "a"}), (20.0, {"data": "b"})])

        # Test successful sync records discrepancy
        initial_rgb_count = len(self.collector.sync_stats["rgb_discrepancies"])
        result = self.collector._find_closest_in_buffer(20.02, buffer, "rgb")
        self.assertIsNotNone(result)
        self.assertEqual(
            len(self.collector.sync_stats["rgb_discrepancies"]), initial_rgb_count + 1
        )
        self.assertAlmostEqual(
            self.collector.sync_stats["rgb_discrepancies"][-1], 0.02, places=3
        )

        # Test failed sync records failure
        initial_fail_count = self.collector.sync_stats["failed_syncs"]["depth"]
        result = self.collector._find_closest_in_buffer(25.0, buffer, "depth")
        self.assertIsNone(result)
        self.assertEqual(
            self.collector.sync_stats["failed_syncs"]["depth"], initial_fail_count + 1
        )

    def test_synchronize_all_data_complete(self):
        """Test synchronization with complete data sets."""
        self.collector.start_episode("test sync")

        # Add synchronized data to all buffers
        timestamps = [10.0, 11.0, 12.0]

        for i, ts in enumerate(timestamps):
            # Add joint state data
            joint_data = {
                "timestamp": time.time(),
                "ros_timestamp": ts,
                "arm_joint_positions": [0.1 * i, 0.2 * i, 0.3 * i],
                "arm_joint_velocities": [0.05, 0.1, 0.15],
                "gripper_joint_positions": [0.01 * i, 0.01 * i],
                "gripper_joint_velocities": [0.001, 0.001],
                "prompt": f"prompt_{i}",
            }
            self.collector.joint_state_buffer[ts] = joint_data

            # Add RGB data
            rgb_data = {
                "timestamp": time.time(),
                "ros_timestamp": ts,
                "rgb_image_bytes": f"rgb_data_{i}".encode().hex(),
                "data_type": "rgb",
            }
            self.collector.rgb_buffer[ts] = rgb_data

            # Add depth data
            depth_data = {
                "timestamp": time.time(),
                "ros_timestamp": ts,
                "depth_image_bytes": f"depth_data_{i}".encode().hex(),
                "data_type": "depth",
            }
            self.collector.depth_buffer[ts] = depth_data

            # Add pose data
            pose_data = {
                "timestamp": time.time(),
                "ros_timestamp": ts,
                "pose_type": "end_effector",
                "position": {"x": 0.5 + 0.1 * i, "y": 0.0, "z": 0.3},
                "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
            }
            self.collector.pose_buffer[ts] = pose_data

        synced_data = self.collector._synchronize_all_data()

        # Should have 2 data points (n-1 for n timestamps)
        self.assertEqual(len(synced_data), 2)

        # Check structure of first data point
        first_point = synced_data[0]
        self.assertIn("observations", first_point)
        self.assertIn("action", first_point)
        self.assertIn("prompt", first_point)
        self.assertIn("is_first", first_point)
        self.assertIn("is_last", first_point)

        # Check observations structure
        obs = first_point["observations"]
        self.assertIn("joint_state", obs)
        self.assertIn("gripper_state", obs)
        self.assertIn("cartesian_position", obs)
        self.assertIn("rgb_image", obs)
        self.assertIn("depth_image", obs)

        # Check action structure
        action = first_point["action"]
        self.assertIn("joint_state", action)
        self.assertIn("joint_velocities", action)
        self.assertIn("cartesian_velocity", action)

    def test_synchronize_all_data_missing_data(self):
        """Test synchronization with missing data."""
        self.collector.start_episode("test sync")

        # Add only joint state data (missing images and poses)
        joint_data = {
            "timestamp": time.time(),
            "ros_timestamp": 10.0,
            "arm_joint_positions": [0.1, 0.2, 0.3],
            "arm_joint_velocities": [0.05, 0.1, 0.15],
            "gripper_joint_positions": [0.01, 0.01],
            "gripper_joint_velocities": [0.001, 0.001],
            "prompt": "test_prompt",
        }
        self.collector.joint_state_buffer[10.0] = joint_data
        self.collector.joint_state_buffer[11.0] = joint_data.copy()

        synced_data = self.collector._synchronize_all_data()

        # Should have no synchronized data due to missing images/poses
        self.assertEqual(len(synced_data), 0)

    def test_calculate_cartesian_velocity(self):
        """Test cartesian velocity calculation."""
        current_pose = {
            "position": {"x": 0.5, "y": 0.0, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        next_pose = {
            "position": {"x": 0.6, "y": 0.1, "z": 0.4},
            "orientation": {"x": 0.1, "y": 0.0, "z": 0.0, "w": 0.995},
        }

        dt = 1.0  # 1 second

        velocity = self.collector._calculate_cartesian_velocity(
            current_pose, next_pose, dt
        )

        # Check linear velocity
        self.assertAlmostEqual(velocity["linear"]["x"], 0.1, places=3)
        self.assertAlmostEqual(velocity["linear"]["y"], 0.1, places=3)
        self.assertAlmostEqual(velocity["linear"]["z"], 0.1, places=3)

        # Check angular velocity (simplified calculation)
        self.assertAlmostEqual(velocity["angular"]["x"], 0.1, places=3)

    def test_calculate_cartesian_velocity_zero_dt(self):
        """Test cartesian velocity calculation with zero time difference."""
        current_pose = {
            "position": {"x": 0.5, "y": 0.0, "z": 0.3},
            "orientation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        }

        velocity = self.collector._calculate_cartesian_velocity(
            current_pose, current_pose, 0.0
        )

        # Should return zero velocities
        self.assertEqual(velocity["linear"]["x"], 0.0)
        self.assertEqual(velocity["linear"]["y"], 0.0)
        self.assertEqual(velocity["linear"]["z"], 0.0)
        self.assertEqual(velocity["angular"]["x"], 0.0)
        self.assertEqual(velocity["angular"]["y"], 0.0)
        self.assertEqual(velocity["angular"]["z"], 0.0)

    def test_save_episode_data(self):
        """Test saving episode data to disk."""
        self.collector.start_episode("test episode")

        # Add some mock trajectory data
        self.collector.current_episode_data["trajectory_data"] = [
            {
                "prompt": "test_prompt",
                "observations": {"joint_state": [0.1, 0.2, 0.3]},
                "action": {"joint_state": [0.2, 0.3, 0.4]},
            }
        ]

        # Set episode directory (normally done in stop_episode)
        self.collector.episode_directory = self.collector._create_episode_directory(
            self.collector.episode_id
        )

        # Mock synchronization to avoid complex setup
        with patch.object(self.collector, "_synchronize_all_data", return_value=[]):
            result_path = self.collector.save_episode_data()

        # Verify file was created
        self.assertIsNotNone(result_path)
        self.assertNotEqual(result_path, "")
        self.assertTrue(os.path.exists(result_path))
        self.assertTrue(result_path.endswith("episode_data.json"))

        # Verify file contents
        with open(result_path, "r") as f:
            saved_data = json.load(f)

        self.assertEqual(saved_data["input_message"], "test episode")
        self.assertIn("episode_id", saved_data)
        self.assertIn("start_time", saved_data)

    def test_save_episode_data_no_episode(self):
        """Test saving when no episode is active."""
        # Start with a fresh collector that has no episode data
        fresh_collector = TrajectoryDataCollector(
            self.mock_logger,
            self.arm_joint_names,
            self.gripper_joint_names,
            save_directory=self.test_dir,
        )

        result = fresh_collector.save_episode_data()
        self.assertEqual(result, "")
        self.mock_logger.error.assert_called_with("No episode data to save")

    def test_summarize_trajectory_data(self):
        """Test trajectory data summarization."""
        self.collector.start_episode("test episode")

        # Create mock trajectory data
        trajectory_data = [
            {
                "prompt": "pick object",
                "is_first": True,
                "is_last": False,
                "observations": {
                    "joint_state": [0.1, 0.2, 0.3],
                    "gripper_state": [0.01, 0.01],
                    "cartesian_position": {"position": {"x": 0.5, "y": 0.0, "z": 0.3}},
                    "timestamp": 10.0,
                },
                "action": {
                    "joint_state": [0.2, 0.3, 0.4],
                    "joint_velocities": [0.1, 0.1, 0.1],
                    "gripper_state": [0.02, 0.02],
                    "cartesian_position": {"position": {"x": 0.6, "y": 0.1, "z": 0.4}},
                    "timestamp": 11.0,
                },
            },
            {
                "prompt": "pick object",
                "is_first": False,
                "is_last": True,
                "observations": {
                    "joint_state": [0.2, 0.3, 0.4],
                    "gripper_state": [0.02, 0.02],
                    "cartesian_position": {"position": {"x": 0.6, "y": 0.1, "z": 0.4}},
                    "timestamp": 11.0,
                },
                "action": {
                    "joint_state": [0.3, 0.4, 0.5],
                    "joint_velocities": [0.1, 0.1, 0.1],
                    "gripper_state": [0.01, 0.01],  # Gripper state change
                    "cartesian_position": {"position": {"x": 0.7, "y": 0.2, "z": 0.5}},
                    "timestamp": 12.0,
                },
            },
        ]

        self.collector.current_episode_data["trajectory_data"] = trajectory_data

        summary = self.collector.summarize_trajectory_data()

        # Check basic statistics
        self.assertEqual(summary["trajectory_overview"]["total_data_points"], 2)
        self.assertEqual(summary["trajectory_overview"]["unique_prompts"], 1)
        self.assertGreater(summary["trajectory_overview"]["duration_seconds"], 0)

        # Check movement analysis
        self.assertGreater(
            summary["movement_analysis"]["total_cartesian_distance_meters"], 0
        )
        self.assertEqual(
            len(summary["movement_analysis"]["joint_movement_ranges_radians"]), 3
        )

        # Check manipulation analysis
        self.assertEqual(summary["manipulation_analysis"]["gripper_state_changes"], 1)
        self.assertEqual(
            summary["manipulation_analysis"]["episode_boundaries"]["first_points"], 1
        )
        self.assertEqual(
            summary["manipulation_analysis"]["episode_boundaries"]["last_points"], 1
        )

    def test_summarize_trajectory_data_empty(self):
        """Test summarization with no trajectory data."""
        self.collector.start_episode("test episode")

        summary = self.collector.summarize_trajectory_data()

        self.assertIn("error", summary)
        self.assertEqual(summary["error"], "No trajectory data to summarize")

    def test_log_trajectory_summary(self):
        """Test logging of trajectory summary."""
        self.collector.start_episode("test episode")

        # Mock summarize_trajectory_data to return predictable data
        mock_summary = {
            "trajectory_overview": {
                "total_data_points": 5,
                "duration_seconds": 2.5,
                "unique_prompts": 1,
            },
            "movement_analysis": {
                "total_cartesian_distance_meters": 0.15,
                "max_joint_movement_radians": 0.3,
            },
            "manipulation_analysis": {"gripper_state_changes": 2},
            "data_quality": {
                "average_frequency_hz": 2.0,
                "has_complete_observations": True,
                "has_complete_actions": True,
            },
        }

        with patch.object(
            self.collector, "summarize_trajectory_data", return_value=mock_summary
        ):
            self.collector.log_trajectory_summary()

        # Verify logger was called with summary information
        self.mock_logger.info.assert_called()
        info_calls = [call.args[0] for call in self.mock_logger.info.call_args_list]

        # Check that key information was logged
        summary_start = any("=== TRAJECTORY SUMMARY ===" in call for call in info_calls)
        self.assertTrue(summary_start)

    def test_compute_sync_statistics(self):
        """Test computation of synchronization statistics."""
        # Add some mock discrepancies
        self.collector.sync_stats["rgb_discrepancies"] = [0.01, 0.02, 0.03]
        self.collector.sync_stats["depth_discrepancies"] = [0.015, 0.025]
        self.collector.sync_stats["failed_syncs"]["rgb"] = 2
        self.collector.sync_stats["failed_syncs"]["depth"] = 1

        stats = self.collector._compute_sync_statistics()

        # Check overall statistics
        self.assertEqual(stats["total_failed_syncs"], 3)
        self.assertEqual(stats["sync_tolerance_used"], self.collector.sync_tolerance)

        # Check RGB statistics
        self.assertEqual(stats["rgb_sync"]["count"], 3)
        self.assertAlmostEqual(stats["rgb_sync"]["mean_discrepancy"], 0.02, places=3)
        self.assertEqual(stats["rgb_sync"]["max_discrepancy"], 0.03)
        self.assertEqual(stats["rgb_sync"]["min_discrepancy"], 0.01)

        # Check depth statistics
        self.assertEqual(stats["depth_sync"]["count"], 2)
        self.assertAlmostEqual(stats["depth_sync"]["mean_discrepancy"], 0.02, places=3)

    def test_group_timestamps_by_prompt(self):
        """Test grouping of timestamps by prompt."""
        # Add mock joint state data with different prompts
        self.collector.joint_state_buffer[10.0] = {"prompt": "prompt_a"}
        self.collector.joint_state_buffer[11.0] = {"prompt": "prompt_a"}
        self.collector.joint_state_buffer[12.0] = {"prompt": "prompt_b"}
        self.collector.joint_state_buffer[13.0] = {"prompt": "prompt_b"}
        self.collector.joint_state_buffer[14.0] = {"prompt": "prompt_a"}

        timestamps = [10.0, 11.0, 12.0, 13.0, 14.0]
        groups = self.collector._group_timestamps_by_prompt(timestamps)

        self.assertEqual(len(groups), 2)
        self.assertIn("prompt_a", groups)
        self.assertIn("prompt_b", groups)
        self.assertEqual(len(groups["prompt_a"]), 3)
        self.assertEqual(len(groups["prompt_b"]), 2)

    def test_is_first_and_last_in_prompt_group(self):
        """Test identification of first and last timestamps in prompt groups."""
        prompt_groups = {"prompt_a": [10.0, 11.0, 14.0], "prompt_b": [12.0, 13.0]}

        # Test first in group
        self.assertTrue(
            self.collector._is_first_in_prompt_group(10.0, "prompt_a", prompt_groups)
        )
        self.assertFalse(
            self.collector._is_first_in_prompt_group(11.0, "prompt_a", prompt_groups)
        )

        # Test last in group
        self.assertTrue(
            self.collector._is_last_in_prompt_group(14.0, "prompt_a", prompt_groups)
        )
        self.assertFalse(
            self.collector._is_last_in_prompt_group(11.0, "prompt_a", prompt_groups)
        )

        # Test with non-existent prompt
        self.assertFalse(
            self.collector._is_first_in_prompt_group(10.0, "nonexistent", prompt_groups)
        )
        self.assertFalse(
            self.collector._is_last_in_prompt_group(10.0, "nonexistent", prompt_groups)
        )


if __name__ == "__main__":
    unittest.main()
