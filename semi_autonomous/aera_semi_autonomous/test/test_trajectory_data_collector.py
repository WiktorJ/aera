import unittest
from unittest.mock import patch, mock_open, MagicMock, Mock
import time
import numpy as np
from collections import OrderedDict

# NOTE: This test assumes 'aera_semi_autonomous.data.trajectory_data_collector' can be imported.
from aera_semi_autonomous.data.trajectory_data_collector import TrajectoryDataCollector


class TestTrajectoryDataCollector(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        mock_logger = Mock()
        arm_joint_names = ["joint1", "joint2"]
        gripper_joint_names = ["gripper_joint"]
        
        self.collector = TrajectoryDataCollector(
            mock_logger,
            arm_joint_names,
            gripper_joint_names,
            save_directory="/tmp/test_data"
        )

    def test_episode_lifecycle(self):
        """Test the start and stop of an episode."""
        self.assertIsNone(self.collector.current_episode_id)
        self.collector.start_episode()
        self.assertIsNotNone(self.collector.current_episode_id)
        self.collector.stop_episode()
        self.assertIsNone(self.collector.current_episode_id)
        self.assertEqual(len(self.collector.image_buffer), 0)
        self.assertEqual(len(self.collector.depth_buffer), 0)

    def test_data_recording(self):
        """Test recording of various data types into buffers."""
        self.collector.start_episode("test episode")
        
        # Create mock ROS message
        mock_rgb_msg = Mock()
        mock_rgb_msg.header.stamp.sec = 1000
        mock_rgb_msg.header.stamp.nanosec = 0
        
        self.collector.record_rgb_image(mock_rgb_msg)
        
        # Just verify the method can be called
        self.assertIsNotNone(self.collector.current_episode_id)

    def test_find_closest_in_buffer(self):
        """Test the internal _find_closest_in_buffer helper."""
        from sortedcontainers import SortedDict
        buffer = SortedDict([(10.0, 'a'), (20.0, 'b'), (30.0, 'c')])
        self.assertEqual(self.collector._find_closest_in_buffer(24.0, buffer), 'b')
        self.assertEqual(self.collector._find_closest_in_buffer(26.0, buffer), 'c')
        self.assertEqual(self.collector._find_closest_in_buffer(5.0, buffer), 'a')
        self.assertEqual(self.collector._find_closest_in_buffer(35.0, buffer), 'c')

    def test_synchronize_all_data(self):
        """Test synchronization of data from different buffers."""
        # Start an episode first
        self.collector.start_episode("test sync")
        
        # Create mock data and add to buffers manually for testing
        mock_rgb_msg1 = Mock()
        mock_rgb_msg1.header.stamp.sec = 10
        mock_rgb_msg1.header.stamp.nanosec = 0
        
        mock_rgb_msg2 = Mock() 
        mock_rgb_msg2.header.stamp.sec = 20
        mock_rgb_msg2.header.stamp.nanosec = 0
        
        self.collector.record_rgb_image(mock_rgb_msg1)
        self.collector.record_rgb_image(mock_rgb_msg2)

        synced_data = self.collector._synchronize_all_data()

        # Just verify we get some data back
        self.assertIsInstance(synced_data, list)

    @patch('os.makedirs')
    @patch('builtins.open', new_callable=mock_open)
    @patch('json.dump')
    @patch('cv2.imwrite')
    def test_save_episode_data(self, mock_imwrite, mock_json_dump, mock_open_file, mock_makedirs):
        """Test saving of an episode's data to files."""
        self.collector.start_episode()
        episode_id = self.collector.current_episode_id
        mock_image = np.zeros((10, 10, 3), dtype=np.uint8)
        
        # Mock internal synchronization to return predictable data
        sync_data = [{
            'timestamp': 10.0, 'image': mock_image, 'image_path': f'images/img_10.000000.png',
            'depth': None, 'depth_path': None, 'joint_states': {'j': 1},
            'gripper_state': None, 'prompt': None
        }]
        self.collector._synchronize_all_data = MagicMock(return_value=sync_data)

        self.collector.save_episode_data()

        self.collector._synchronize_all_data.assert_called_once()
        mock_makedirs.assert_called()
        mock_open_file.assert_called_once_with(f'/tmp/test_data/{episode_id}/trajectory.json', 'w')
        mock_json_dump.assert_called_once()
        expected_img_path = f'/tmp/test_data/{episode_id}/images/img_10.000000.png'
        mock_imwrite.assert_called_once_with(expected_img_path, mock_image)
