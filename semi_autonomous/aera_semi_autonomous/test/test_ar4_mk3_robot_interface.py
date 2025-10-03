import sys
import os
import unittest
from unittest.mock import Mock, MagicMock, patch, call
import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import Image

# Add the project root to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

from semi_autonomous.aera_semi_autonomous.aera_semi_autonomous.control.ar4_mk3_robot_interface import (
    Ar4Mk3RobotInterface,
)
from aera.autonomous.envs.ar4_mk3_base import Ar4Mk3Env


class TestAr4Mk3RobotInterface(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a mock environment
        self.mock_env = Mock(spec=Ar4Mk3Env)
        self.mock_env.use_eef_control = True
        self.mock_env.data = Mock()
        self.mock_env.model = Mock()
        self.mock_env._utils = Mock()
        
        # Mock data attributes
        self.mock_env.data.qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0])
        
        # Create the robot interface
        self.robot_interface = Ar4Mk3RobotInterface(self.mock_env)

    def test_init_default_camera_config(self):
        """Test initialization with default camera configuration."""
        self.assertEqual(self.robot_interface.camera_config["width"], 640)
        self.assertEqual(self.robot_interface.camera_config["height"], 480)
        self.assertEqual(self.robot_interface.camera_config["fx"], 525.0)
        self.assertEqual(self.robot_interface.camera_config["fy"], 525.0)
        self.assertEqual(self.robot_interface.camera_config["cx"], 320.0)
        self.assertEqual(self.robot_interface.camera_config["cy"], 240.0)

    def test_init_custom_camera_config(self):
        """Test initialization with custom camera configuration."""
        custom_config = {
            "width": 1280,
            "height": 720,
            "fx": 600.0,
            "fy": 600.0,
            "cx": 640.0,
            "cy": 360.0,
        }
        robot_interface = Ar4Mk3RobotInterface(self.mock_env, custom_config)
        
        self.assertEqual(robot_interface.camera_config["width"], 1280)
        self.assertEqual(robot_interface.camera_config["height"], 720)
        self.assertEqual(robot_interface.camera_config["fx"], 600.0)

    def test_get_logger(self):
        """Test that get_logger returns a logger instance."""
        logger = self.robot_interface.get_logger()
        self.assertIsNotNone(logger)
        self.assertEqual(logger.name, "semi_autonomous.aera_semi_autonomous.aera_semi_autonomous.control.ar4_mk3_robot_interface")

    def test_go_home_eef_control_success(self):
        """Test go_home with end-effector control mode."""
        self.mock_env.use_eef_control = True
        self.mock_env.step.return_value = (None, None, None, None, None)
        
        result = self.robot_interface.go_home()
        
        self.assertTrue(result)
        self.mock_env.step.assert_called()

    def test_go_home_joint_control_success(self):
        """Test go_home with joint control mode."""
        self.mock_env.use_eef_control = False
        self.mock_env.step.return_value = (None, None, None, None, None)
        
        result = self.robot_interface.go_home()
        
        self.assertTrue(result)
        self.mock_env.step.assert_called()

    def test_go_home_exception(self):
        """Test go_home when an exception occurs."""
        self.mock_env.step.side_effect = Exception("Test exception")
        
        result = self.robot_interface.go_home()
        
        self.assertFalse(result)

    def test_move_to_eef_control_success(self):
        """Test move_to with end-effector control mode."""
        self.mock_env.use_eef_control = True
        self.mock_env.step.return_value = (None, None, None, None, None)
        # Mock the gripper position to be close to target after a few steps
        positions = [
            np.array([0.0, 0.0, 0.0]),  # Initial position
            np.array([0.05, 0.05, 0.05]),  # After first step
            np.array([0.1, 0.1, 0.1]),  # Final position (close to target)
        ]
        self.mock_env._utils.get_site_xpos.side_effect = positions
        
        pose = Pose()
        pose.position = Point(x=0.1, y=0.1, z=0.1)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        result = self.robot_interface.move_to(pose)
        
        self.assertTrue(result)
        self.mock_env.step.assert_called()

    def test_move_to_joint_control_not_implemented(self):
        """Test move_to with joint control mode (not fully implemented)."""
        self.mock_env.use_eef_control = False
        
        pose = Pose()
        pose.position = Point(x=0.1, y=0.1, z=0.1)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        result = self.robot_interface.move_to(pose)
        
        self.assertFalse(result)

    def test_move_to_exception(self):
        """Test move_to when an exception occurs."""
        self.mock_env.use_eef_control = True
        self.mock_env._utils.get_site_xpos.side_effect = Exception("Test exception")
        
        pose = Pose()
        pose.position = Point(x=0.1, y=0.1, z=0.1)
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        result = self.robot_interface.move_to(pose)
        
        self.assertFalse(result)

    def test_release_gripper_eef_control_success(self):
        """Test release_gripper with end-effector control mode."""
        self.mock_env.use_eef_control = True
        self.mock_env.step.return_value = (None, None, None, None, None)
        
        result = self.robot_interface.release_gripper()
        
        self.assertTrue(result)
        # Check that step was called with the correct action
        self.mock_env.step.assert_called_once()
        call_args = self.mock_env.step.call_args[0][0]
        expected_action = np.array([0.0, 0.0, 0.0, -1.0])
        np.testing.assert_array_equal(call_args, expected_action)

    def test_release_gripper_joint_control_success(self):
        """Test release_gripper with joint control mode."""
        self.mock_env.use_eef_control = False
        self.mock_env.step.return_value = (None, None, None, None, None)
        
        result = self.robot_interface.release_gripper()
        
        self.assertTrue(result)
        # Check that step was called with the correct action
        self.mock_env.step.assert_called_once()
        call_args = self.mock_env.step.call_args[0][0]
        expected_action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0])
        np.testing.assert_array_equal(call_args, expected_action)

    def test_release_gripper_exception(self):
        """Test release_gripper when an exception occurs."""
        self.mock_env.step.side_effect = Exception("Test exception")
        
        result = self.robot_interface.release_gripper()
        
        self.assertFalse(result)

    def test_grasp_at_success(self):
        """Test grasp_at successful execution."""
        with patch.object(self.robot_interface, 'move_to', return_value=True) as mock_move_to:
            self.mock_env.step.return_value = (None, None, None, None, None)
            
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            gripper_pos = 0.5
            
            result = self.robot_interface.grasp_at(pose, gripper_pos)
            
            self.assertTrue(result)
            self.assertEqual(mock_move_to.call_count, 3)  # above, grasp, lift
            self.mock_env.step.assert_called_once()  # For gripper action

    def test_grasp_at_move_failure(self):
        """Test grasp_at when move_to fails."""
        with patch.object(self.robot_interface, 'move_to', return_value=False) as mock_move_to:
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            gripper_pos = 0.5
            
            result = self.robot_interface.grasp_at(pose, gripper_pos)
            
            self.assertFalse(result)
            mock_move_to.assert_called_once()  # Should fail on first move_to call

    def test_release_at_success(self):
        """Test release_at successful execution."""
        with patch.object(self.robot_interface, 'move_to', return_value=True) as mock_move_to, \
             patch.object(self.robot_interface, 'release_gripper', return_value=True) as mock_release_gripper:
            
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            result = self.robot_interface.release_at(pose)
            
            self.assertTrue(result)
            mock_move_to.assert_called_once_with(pose)
            mock_release_gripper.assert_called_once()

    def test_release_at_move_failure(self):
        """Test release_at when move_to fails."""
        with patch.object(self.robot_interface, 'move_to', return_value=False) as mock_move_to:
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            
            result = self.robot_interface.release_at(pose)
            
            self.assertFalse(result)
            mock_move_to.assert_called_once_with(pose)

    def test_get_end_effector_pose_success(self):
        """Test get_end_effector_pose successful execution."""
        # Mock the position and rotation matrix
        mock_pos = np.array([0.1, 0.2, 0.3])
        mock_rot_matrix = np.eye(3)  # Identity matrix
        
        self.mock_env._utils.get_site_xpos.return_value = mock_pos
        self.mock_env._utils.get_site_xmat.return_value = mock_rot_matrix.flatten()
        
        pose = self.robot_interface.get_end_effector_pose()
        
        self.assertIsNotNone(pose)
        self.assertAlmostEqual(pose.position.x, 0.1)
        self.assertAlmostEqual(pose.position.y, 0.2)
        self.assertAlmostEqual(pose.position.z, 0.3)
        # For identity matrix, quaternion should be [0, 0, 0, 1]
        self.assertAlmostEqual(pose.orientation.x, 0.0, places=5)
        self.assertAlmostEqual(pose.orientation.y, 0.0, places=5)
        self.assertAlmostEqual(pose.orientation.z, 0.0, places=5)
        self.assertAlmostEqual(pose.orientation.w, 1.0, places=5)

    def test_get_end_effector_pose_exception(self):
        """Test get_end_effector_pose when an exception occurs."""
        self.mock_env._utils.get_site_xpos.side_effect = Exception("Test exception")
        
        pose = self.robot_interface.get_end_effector_pose()
        
        self.assertIsNone(pose)

    def test_get_latest_rgb_image_success(self):
        """Test get_latest_rgb_image successful execution."""
        mock_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        self.mock_env.render.return_value = mock_image
        
        image = self.robot_interface.get_latest_rgb_image()
        
        self.assertIsNotNone(image)
        np.testing.assert_array_equal(image, mock_image)

    def test_get_latest_rgb_image_none_returned(self):
        """Test get_latest_rgb_image when render returns None."""
        self.mock_env.render.return_value = None
        self.robot_interface._latest_rgb_image = np.array([1, 2, 3])
        
        image = self.robot_interface.get_latest_rgb_image()
        
        np.testing.assert_array_equal(image, np.array([1, 2, 3]))

    def test_get_latest_rgb_image_exception(self):
        """Test get_latest_rgb_image when an exception occurs."""
        self.mock_env.render.side_effect = Exception("Test exception")
        
        image = self.robot_interface.get_latest_rgb_image()
        
        self.assertIsNone(image)

    def test_get_latest_depth_image_success(self):
        """Test get_latest_depth_image successful execution."""
        mock_depth = np.random.rand(480, 640).astype(np.float32)
        self.mock_env.render.return_value = mock_depth
        
        depth = self.robot_interface.get_latest_depth_image()
        
        self.assertIsNotNone(depth)
        np.testing.assert_array_equal(depth, mock_depth)
        # Verify render was called with correct mode
        self.mock_env.render.assert_called_with(mode="depth_array")

    def test_get_latest_depth_image_exception(self):
        """Test get_latest_depth_image when an exception occurs."""
        self.mock_env.render.side_effect = Exception("Test exception")
        
        depth = self.robot_interface.get_latest_depth_image()
        
        self.assertIsNone(depth)

    def test_get_camera_intrinsics(self):
        """Test get_camera_intrinsics returns correct intrinsics."""
        intrinsics = self.robot_interface.get_camera_intrinsics()
        
        self.assertIsInstance(intrinsics, o3d.camera.PinholeCameraIntrinsic)
        self.assertEqual(intrinsics.width, 640)
        self.assertEqual(intrinsics.height, 480)

    def test_get_cam_to_base_transform(self):
        """Test get_cam_to_base_transform returns correct transform."""
        transform = self.robot_interface.get_cam_to_base_transform()
        
        self.assertIsNotNone(transform)
        self.assertEqual(transform.shape, (4, 4))
        # Check that it's the expected transform matrix
        expected_transform = np.array([
            [0.0, -1.0, 0.0, 0.3],
            [0.0, 0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0, 1.0],
        ])
        np.testing.assert_array_equal(transform, expected_transform)

    def test_get_last_rgb_msg(self):
        """Test get_last_rgb_msg returns None (simulation doesn't use ROS messages)."""
        msg = self.robot_interface.get_last_rgb_msg()
        self.assertIsNone(msg)

    def test_update_camera_config(self):
        """Test update_camera_config updates configuration correctly."""
        new_config = {
            "width": 1920,
            "height": 1080,
            "fx": 800.0,
        }
        
        self.robot_interface.update_camera_config(new_config)
        
        self.assertEqual(self.robot_interface.camera_config["width"], 1920)
        self.assertEqual(self.robot_interface.camera_config["height"], 1080)
        self.assertEqual(self.robot_interface.camera_config["fx"], 800.0)
        # Check that intrinsics were updated
        intrinsics = self.robot_interface.get_camera_intrinsics()
        self.assertEqual(intrinsics.width, 1920)
        self.assertEqual(intrinsics.height, 1080)

    def test_set_cam_to_base_transform_valid(self):
        """Test set_cam_to_base_transform with valid 4x4 matrix."""
        new_transform = np.eye(4)
        new_transform[0, 3] = 1.0  # Set translation
        
        self.robot_interface.set_cam_to_base_transform(new_transform)
        
        np.testing.assert_array_equal(
            self.robot_interface.get_cam_to_base_transform(), new_transform
        )

    def test_set_cam_to_base_transform_invalid_shape(self):
        """Test set_cam_to_base_transform with invalid matrix shape."""
        invalid_transform = np.eye(3)  # 3x3 instead of 4x4
        
        with self.assertRaises(ValueError) as context:
            self.robot_interface.set_cam_to_base_transform(invalid_transform)
        
        self.assertIn("Transform must be a 4x4 matrix", str(context.exception))

    def test_home_joint_positions(self):
        """Test that home joint positions are correctly initialized."""
        expected_home = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        np.testing.assert_array_equal(
            self.robot_interface.home_joint_positions, expected_home
        )


if __name__ == "__main__":
    unittest.main()
