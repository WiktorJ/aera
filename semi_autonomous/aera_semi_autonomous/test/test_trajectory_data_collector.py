import unittest
from unittest.mock import patch, mock_open, MagicMock
import time
import numpy as np
from collections import OrderedDict

# NOTE: This test assumes 'aera_semi_autonomous.trajectory_data_collector' can be imported.
from aera_semi_autonomous.trajectory_data_collector import TrajectoryDataCollector


class TestTrajectoryDataCollector(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.collector = TrajectoryDataCollector(output_dir="/tmp/test_data")

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
        self.collector.start_episode()
        ts = time.time()
        self.collector.record_image(ts, np.zeros((1, 1, 3)))
        self.collector.record_joint_states(ts, {'j1': 0})
        self.assertEqual(len(self.collector.image_buffer), 1)
        self.assertEqual(len(self.collector.joint_states_buffer), 1)

    def test_find_closest_in_buffer(self):
        """Test the internal _find_closest_in_buffer helper."""
        buffer = OrderedDict([(10.0, 'a'), (20.0, 'b'), (30.0, 'c')])
        self.assertEqual(self.collector._find_closest_in_buffer(buffer, 24.0), 'b')
        self.assertEqual(self.collector._find_closest_in_buffer(buffer, 26.0), 'c')
        self.assertEqual(self.collector._find_closest_in_buffer(buffer, 5.0), 'a')
        self.assertEqual(self.collector._find_closest_in_buffer(buffer, 35.0), 'c')

    def test_synchronize_all_data(self):
        """Test synchronization of data from different buffers."""
        self.collector.record_image(10.0, "img1")
        self.collector.record_joint_states(10.1, "js1")
        self.collector.record_gripper_state(9.9, "gs1")
        self.collector.record_image(20.0, "img2")
        self.collector.record_joint_states(19.8, "js2")

        synced_data = self.collector._synchronize_all_data()

        self.assertEqual(len(synced_data), 2)
        self.assertEqual(synced_data[0]['image'], "img1")
        self.assertEqual(synced_data[0]['joint_states'], "js1")
        self.assertEqual(synced_data[0]['gripper_state'], "gs1")
        self.assertEqual(synced_data[1]['image'], "img2")
        self.assertEqual(synced_data[1]['joint_states'], "js2")

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
