import unittest
import numpy as np
import open3d as o3d

# NOTE: This test assumes 'aera_semi_autonomous.vision.point_cloud_processor' can be imported.
from aera_semi_autonomous.vision.point_cloud_processor import PointCloudProcessor


class TestPointCloudProcessor(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures."""
        self.processor = PointCloudProcessor()

    def test_create_point_cloud_from_depth(self):
        """Test creating a point cloud from a depth image."""
        depth_image = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32)
        width, height = 2, 2
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx=500, fy=500, cx=1, cy=1)

        pcd = self.processor.create_point_cloud_from_depth(depth_image, intrinsic)

        self.assertIsInstance(pcd, o3d.geometry.PointCloud)
        self.assertEqual(len(pcd.points), 4)
        points = np.asarray(pcd.points)
        # All z-coordinates should be close to 1.0
        self.assertTrue(np.allclose(points[:, 2], 1.0))

    def test_get_pose_and_angle_camera_base(self):
        """Test getting pose and angle from a point cloud."""
        # A simple rectangular point cloud
        points = np.array([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [1.0, 2.0, 0.0], [0.0, 2.0, 0.0]
        ])

        center, dims, angle = self.processor.get_pose_and_angle_camera_base(points)

        self.assertTrue(np.allclose(center, [0.5, 1.0, 0.0]))
        # Dimensions could be (2, 1) or (1, 2)
        self.assertTrue(
            (np.allclose(sorted(dims[:2]), [1.0, 2.0]))
        )
        # Angle should be 0 or pi/2
        self.assertTrue(np.isclose(angle, 0.0) or np.isclose(angle, np.pi / 2))

    def test_transform_gripper_angle_to_base_frame(self):
        """Test transforming gripper angle to the base frame."""
        angle_in_cam = np.pi / 4  # 45 degrees

        # Transformation representing a 90-degree rotation around Z
        cam_to_base_affine = np.array([
            [0.0, -1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])

        transformed_angle = self.processor.transform_gripper_angle_to_base_frame(
            angle_in_cam, cam_to_base_affine)

        # Expected angle is 45 + 90 = 135 degrees
        expected_angle = angle_in_cam + np.pi / 2
        self.assertAlmostEqual(transformed_angle, expected_angle)
