import unittest
import numpy as np
import open3d as o3d
from unittest.mock import Mock

from aera_semi_autonomous.vision.point_cloud_processor import PointCloudProcessor


class TestPointCloudProcessor(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mock_logger = Mock()
        self.processor = PointCloudProcessor(self.mock_logger)

    def test_create_point_cloud_from_depth_basic(self):
        """Test creating a point cloud from a depth image."""
        depth_image = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        width, height = 2, 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx=500, fy=500, cx=1, cy=1
        )

        points = self.processor.create_point_cloud_from_depth(depth_image, intrinsic)

        # The method returns numpy array, not PointCloud object
        self.assertIsInstance(points, np.ndarray)
        self.assertEqual(len(points), 4)
        self.assertEqual(points.shape[1], 3)  # Should have x, y, z coordinates
        # All z-coordinates should be close to 1.0
        self.assertTrue(np.allclose(points[:, 2], 1.0))

    def test_create_point_cloud_from_depth_empty(self):
        """Test creating point cloud from empty depth image."""
        depth_image = np.zeros((2, 2), dtype=np.float32)
        width, height = 2, 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx=500, fy=500, cx=1, cy=1
        )

        points = self.processor.create_point_cloud_from_depth(depth_image, intrinsic)

        self.assertIsInstance(points, np.ndarray)
        # Zero depth should result in no valid points or points at origin
        if len(points) > 0:
            self.assertEqual(points.shape[1], 3)

    def test_create_point_cloud_from_depth_varying_depths(self):
        """Test creating point cloud with varying depth values."""
        depth_image = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
        width, height = 2, 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx=500, fy=500, cx=1, cy=1
        )

        points = self.processor.create_point_cloud_from_depth(depth_image, intrinsic)

        self.assertIsInstance(points, np.ndarray)
        if len(points) > 0:
            self.assertEqual(points.shape[1], 3)
            # Check that we have different z values
            unique_z = np.unique(np.round(points[:, 2], 1))
            self.assertGreater(len(unique_z), 1)

    def test_get_pose_and_angle_camera_base_rectangle(self):
        """Test getting pose and angle from a rectangular point cloud."""
        # A simple rectangular point cloud
        points = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 2.0, 0.0], [0.0, 2.0, 0.0]]
        )

        grasp_pose, gripper_angle, gripper_opening = (
            self.processor.get_pose_and_angle_camera_base(points)
        )

        # Check pose is at center of rectangle
        self.assertTrue(np.allclose(grasp_pose[:2], [0.5, 1.0], atol=0.1))
        self.assertAlmostEqual(grasp_pose[2], 0.0, places=5)
        self.assertAlmostEqual(grasp_pose[3], 1.0, places=5)

        # Check gripper opening is the smaller dimension
        self.assertAlmostEqual(gripper_opening, 1.0, places=1)

        # Angle should be a numeric value
        self.assertIsInstance(gripper_angle, (int, float, np.number))

    def test_get_pose_and_angle_camera_base_square(self):
        """Test getting pose and angle from a square point cloud."""
        points = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]
        )

        grasp_pose, gripper_angle, gripper_opening = (
            self.processor.get_pose_and_angle_camera_base(points)
        )

        # Check pose is at center of square
        self.assertTrue(np.allclose(grasp_pose[:2], [0.5, 0.5], atol=0.1))
        self.assertAlmostEqual(grasp_pose[2], 1.0, places=5)

        # For a square, gripper opening should be close to 1.0
        self.assertAlmostEqual(gripper_opening, 1.0, places=1)

    def test_get_pose_and_angle_camera_base_single_point(self):
        """Test getting pose and angle from a single point."""
        points = np.array([[0.5, 0.5, 1.0]])

        grasp_pose, gripper_angle, gripper_opening = (
            self.processor.get_pose_and_angle_camera_base(points)
        )

        # Should handle single point gracefully
        self.assertIsInstance(grasp_pose, np.ndarray)
        self.assertEqual(len(grasp_pose), 4)
        self.assertIsInstance(gripper_angle, (int, float, np.number))
        self.assertIsInstance(gripper_opening, (int, float, np.number))

    def test_transform_gripper_angle_to_base_frame_identity(self):
        """Test transforming gripper angle with identity transformation."""
        angle_in_cam = 45.0  # 45 degrees

        # Identity transformation
        cam_to_base_affine = np.eye(4)

        transformed_angle = self.processor.transform_gripper_angle_to_base_frame(
            angle_in_cam, cam_to_base_affine
        )

        # With identity transform, angle should remain the same
        self.assertAlmostEqual(transformed_angle, angle_in_cam, places=5)

    def test_transform_gripper_angle_to_base_frame_90_rotation(self):
        """Test transforming gripper angle with 90-degree rotation."""
        angle_in_cam = 0.0  # 0 degrees (pointing in +X direction)

        # 90-degree rotation around Z-axis
        cam_to_base_affine = np.array(
            [
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        transformed_angle = self.processor.transform_gripper_angle_to_base_frame(
            angle_in_cam, cam_to_base_affine
        )

        # 0 degrees in camera should become 90 degrees in base
        self.assertAlmostEqual(transformed_angle, 90.0, places=1)

    def test_transform_gripper_angle_to_base_frame_180_rotation(self):
        """Test transforming gripper angle with 180-degree rotation."""
        angle_in_cam = 0.0

        # 180-degree rotation around Z-axis
        cam_to_base_affine = np.array(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        transformed_angle = self.processor.transform_gripper_angle_to_base_frame(
            angle_in_cam, cam_to_base_affine
        )

        # 0 degrees in camera should become 180 degrees in base
        self.assertAlmostEqual(abs(transformed_angle), 180.0, places=1)

    def test_get_drop_pose_from_points_valid(self):
        """Test getting drop pose from valid points."""
        points = np.array(
            [[0.0, 0.0, 2.0], [1.0, 0.0, 2.0], [1.0, 1.0, 2.0], [0.0, 1.0, 2.0]]
        )

        drop_pose = self.processor.get_drop_pose_from_points(points)

        self.assertIsNotNone(drop_pose)
        self.assertIsInstance(drop_pose, np.ndarray)
        self.assertEqual(len(drop_pose), 4)  # type: ignore
        # Check center position
        self.assertTrue(np.allclose(drop_pose[:2], [0.5, 0.5], atol=0.1))  # type: ignore
        self.assertAlmostEqual(drop_pose[2], 2.0, places=5)  # type: ignore
        self.assertAlmostEqual(drop_pose[3], 1.0, places=5)  # type: ignore

    def test_get_drop_pose_from_points_insufficient_points(self):
        """Test getting drop pose with insufficient points."""
        points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])  # Only 2 points

        drop_pose = self.processor.get_drop_pose_from_points(points)

        self.assertIsNone(drop_pose)
        # Verify error was logged
        self.mock_logger.error.assert_called_once()

    def test_get_drop_pose_from_points_empty(self):
        """Test getting drop pose from empty points array."""
        points = np.array([]).reshape(0, 3)

        drop_pose = self.processor.get_drop_pose_from_points(points)

        self.assertIsNone(drop_pose)
        self.mock_logger.error.assert_called_once()

    def test_get_drop_pose_from_points_varying_z(self):
        """Test getting drop pose with varying z coordinates."""
        points = np.array(
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.5], [1.0, 1.0, 2.0], [0.0, 1.0, 1.8]]
        )

        drop_pose = self.processor.get_drop_pose_from_points(points)

        self.assertIsNotNone(drop_pose)
        # Z should be the mean of all z coordinates
        expected_z = np.mean([1.0, 1.5, 2.0, 1.8])
        self.assertAlmostEqual(drop_pose[2], expected_z, places=5)  # type: ignore


if __name__ == "__main__":
    unittest.main()
