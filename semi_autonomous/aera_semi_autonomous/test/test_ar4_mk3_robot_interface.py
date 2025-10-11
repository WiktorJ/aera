import unittest
from unittest.mock import Mock, patch
import numpy as np
import open3d as o3d
from geometry_msgs.msg import Pose, Point, Quaternion

from aera_semi_autonomous.control.ar4_mk3_robot_interface import (
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
        self.mock_env.action_space = Mock()
        self.mock_env.action_space.shape = (7,)
        self.mock_env.model.nq = 8
        self.mock_env.model.nv = 7
        self.mock_env._utils = Mock()

        # Mock data attributes
        self.mock_env.data.qpos = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0, 0.0])
        self.mock_env.initial_qpos = np.array(
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.04, 0.04]
        )
        self.mock_env.initial_gripper_xpos = np.array([0.1, 0.2, 0.3])
        self.mock_env._utils.get_site_xmat.return_value = np.eye(3).flatten()

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

    @patch("aera_semi_autonomous.control.ar4_mk3_robot_interface.time")
    @patch("aera_semi_autonomous.control.ar4_mk3_robot_interface.mujoco")
    def test_go_home_success(self, mock_mujoco, mock_time):
        """Test go_home successful execution."""
        self.mock_env.initial_qpos = np.array([0.0] * 8)
        with patch.object(
            self.robot_interface, "_get_qpos_indices", return_value=np.arange(6)
        ):
            result = self.robot_interface.go_home()

        self.assertTrue(result)
        np.testing.assert_array_equal(
            self.mock_env.data.qpos, self.mock_env.initial_qpos
        )
        self.assertEqual(mock_mujoco.mj_forward.call_count, 101)
        mock_mujoco.mj_forward.assert_called_with(
            self.mock_env.model, self.mock_env.data
        )
        self.assertEqual(mock_time.sleep.call_count, 101)

    def test_move_to_ik_success(self):
        """Test move_to with inverse kinematics."""
        # Mock IK solver to return success
        with patch.object(
            self.robot_interface, "_solve_ik_for_site_pose", return_value=True
        ) as mock_ik:
            # Mock final position check
            self.mock_env._utils.get_site_xpos.return_value = np.array([0.1, 0.1, 0.1])

            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            result = self.robot_interface.move_to(pose)

            self.assertTrue(result)
            mock_ik.assert_called_once()

    def test_move_to_ik_failure(self):
        """Test move_to when IK fails to converge."""
        # Mock IK solver to return failure
        with patch.object(
            self.robot_interface, "_solve_ik_for_site_pose", return_value=False
        ) as mock_ik:
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            result = self.robot_interface.move_to(pose)

            self.assertFalse(result)
            mock_ik.assert_called_once()

    def test_move_to_exception(self):
        """Test move_to when an exception occurs."""
        # Mock IK solver to raise exception
        with patch.object(
            self.robot_interface,
            "_solve_ik_for_site_pose",
            side_effect=Exception("Test exception"),
        ):
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            result = self.robot_interface.move_to(pose)

            self.assertFalse(result)

    def test_release_gripper_success(self):
        """Test release_gripper successful execution."""
        with patch.object(
            self.robot_interface, "_interpolate_gripper", return_value=True
        ) as mock_interpolate:
            result = self.robot_interface.release_gripper()

            self.assertTrue(result)
            mock_interpolate.assert_called_once()
            # Check that it was called with the open gripper qpos from initial_qpos
            expected_qpos = self.mock_env.initial_qpos[-2:]
            called_qpos = mock_interpolate.call_args[0][0]
            np.testing.assert_array_equal(called_qpos, expected_qpos)
    
    def test_release_gripper_with_invalid_initial_qpos(self):
        """Test release_gripper when initial_qpos has invalid gripper values."""
        # Set initial_qpos to have non-positive gripper values
        self.mock_env.initial_qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.01, -0.01])
        
        with patch.object(
            self.robot_interface, "_interpolate_gripper", return_value=True
        ) as mock_interpolate:
            result = self.robot_interface.release_gripper()

            self.assertTrue(result)
            mock_interpolate.assert_called_once()
            # Should use default open values when initial_qpos is invalid
            called_qpos = mock_interpolate.call_args[0][0]
            np.testing.assert_array_equal(called_qpos, np.array([0.04, 0.04]))

    def test_release_gripper_exception(self):
        """Test release_gripper when an exception occurs."""
        self.mock_env.step.side_effect = Exception("Test exception")
        # Mock gripper state to trigger movement
        self.mock_env.data.qpos = np.array(
            [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, -0.007, -0.007]
        )

        result = self.robot_interface.release_gripper()

        self.assertFalse(result)

    def test_grasp_at_success(self):
        """Test grasp_at successful execution."""
        with patch.object(
            self.robot_interface, "move_to", return_value=True
        ) as mock_move_to, patch.object(
            self.robot_interface, "_interpolate_gripper", return_value=True
        ) as mock_interpolate_gripper:
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            gripper_pos = 0.5

            result = self.robot_interface.grasp_at(pose, gripper_pos)

            self.assertTrue(result)
            self.assertEqual(mock_move_to.call_count, 3)  # above, grasp, lift
            mock_interpolate_gripper.assert_called_once()
            # Check the argument passed to _interpolate_gripper
            called_args = mock_interpolate_gripper.call_args[0][0]
            np.testing.assert_array_equal(called_args, np.zeros(2))

    def test_grasp_at_move_failure(self):
        """Test grasp_at when move_to fails."""
        with patch.object(
            self.robot_interface, "move_to", return_value=False
        ) as mock_move_to:
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            gripper_pos = 0.5

            result = self.robot_interface.grasp_at(pose, gripper_pos)

            self.assertFalse(result)
            mock_move_to.assert_called_once()  # Should fail on first move_to call

    def test_release_at_success(self):
        """Test release_at successful execution."""
        with (
            patch.object(
                self.robot_interface, "move_to", return_value=True
            ) as mock_move_to,
            patch.object(
                self.robot_interface, "release_gripper", return_value=True
            ) as mock_release_gripper,
        ):
            pose = Pose()
            pose.position = Point(x=0.1, y=0.1, z=0.1)
            pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

            result = self.robot_interface.release_at(pose)

            self.assertTrue(result)
            mock_move_to.assert_called_once_with(pose)
            mock_release_gripper.assert_called_once()

    def test_release_at_move_failure(self):
        """Test release_at when move_to fails."""
        with patch.object(
            self.robot_interface, "move_to", return_value=False
        ) as mock_move_to:
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
        self.assertAlmostEqual(pose.position.x, 0.1)  # type: ignore
        self.assertAlmostEqual(pose.position.y, 0.2)  # type: ignore
        self.assertAlmostEqual(pose.position.z, 0.3)  # type: ignore
        # For identity matrix, quaternion should be [0, 0, 0, 1]
        self.assertAlmostEqual(pose.orientation.x, 0.0, places=5)  # type: ignore
        self.assertAlmostEqual(pose.orientation.y, 0.0, places=5)  # type: ignore
        self.assertAlmostEqual(pose.orientation.z, 0.0, places=5)  # type: ignore
        self.assertAlmostEqual(pose.orientation.w, 1.0, places=5)  # type: ignore

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
        np.testing.assert_array_equal(image, mock_image)  # type: ignore

    def test_get_latest_rgb_image_none_returned(self):
        """Test get_latest_rgb_image when render returns None."""
        self.mock_env.render.return_value = None
        self.robot_interface._latest_rgb_image = np.array([1, 2, 3])

        image = self.robot_interface.get_latest_rgb_image()

        np.testing.assert_array_equal(image, np.array([1, 2, 3]))  # type: ignore

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
        np.testing.assert_array_equal(depth, mock_depth)  # type: ignore
        # Verify render was called
        self.mock_env.render.assert_called_once()

    def test_get_latest_depth_image_exception(self):
        """Test get_latest_depth_image when an exception occurs."""
        self.mock_env.render.side_effect = Exception("Test exception")

        depth = self.robot_interface.get_latest_depth_image()

        self.assertIsNone(depth)

    def test_get_camera_intrinsics(self):
        """Test get_camera_intrinsics returns correct intrinsics."""
        intrinsics = self.robot_interface.get_camera_intrinsics()

        self.assertIsInstance(intrinsics, o3d.camera.PinholeCameraIntrinsic)
        self.assertEqual(intrinsics.width, 640)  # type: ignore
        self.assertEqual(intrinsics.height, 480)  # type: ignore

    def test_get_cam_to_base_transform(self):
        """Test get_cam_to_base_transform returns correct transform."""
        transform = self.robot_interface.get_cam_to_base_transform()

        self.assertIsNotNone(transform)
        self.assertEqual(transform.shape, (4, 4))  # type: ignore
        # Check that it's the expected transform matrix
        expected_transform = np.array(
            [
                [0.0, -1.0, 0.0, 0.3],
                [0.0, 0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0, 0.5],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        np.testing.assert_array_equal(transform, expected_transform)  # type: ignore

    def test_home_pose_initialization(self):
        """Test that home pose is correctly initialized."""
        # Initialize home pose
        self.robot_interface._initialize_home_pose()

        self.assertIsNotNone(self.robot_interface.home_pose)
        self.assertAlmostEqual(self.robot_interface.home_pose.position.x, 0.1)
        self.assertAlmostEqual(self.robot_interface.home_pose.position.y, 0.2)
        self.assertAlmostEqual(self.robot_interface.home_pose.position.z, 0.3)


if __name__ == "__main__":
    unittest.main()
