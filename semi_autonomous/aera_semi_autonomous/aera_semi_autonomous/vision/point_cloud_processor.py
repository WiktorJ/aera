import cv2
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


# Add return types in this file AI!
class PointCloudProcessor:
    def __init__(self, logger):
        self.logger = logger

    def get_pose_and_angle_camera_base(self, points_camera_frame: np.ndarray):
        """Calculate grasp pose and angle from camera frame points."""
        # TODO: Try with max
        grasp_z_camera = np.mean(points_camera_frame[:, 2])

        # Calculate center in camera frame (XY plane)
        xy_points_camera = points_camera_frame[:, :2].astype(np.float32)
        center_camera, dimensions, theta = cv2.minAreaRect(xy_points_camera)

        # Create grasp pose in camera frame
        grasp_pose_camera = np.array(
            [center_camera[0], center_camera[1], grasp_z_camera, 1.0]
        )

        # TODO: Try this after change of base
        gripper_angle_camera = theta
        if dimensions[0] > dimensions[1]:
            gripper_angle_camera -= 90

        gripper_opening = min(dimensions)

        return grasp_pose_camera, gripper_angle_camera, gripper_opening

    def transform_gripper_angle_to_base_frame(
        self, gripper_angle_camera, cam_to_base_affine
    ):
        """Transform gripper angle from camera frame to base frame."""
        gripper_vec_camera = np.array(
            [
                np.cos(np.radians(gripper_angle_camera)),
                np.sin(np.radians(gripper_angle_camera)),
                0.0,
            ]
        )
        cam_to_base_rotation = cam_to_base_affine[:3, :3]
        # Transform the vector to base frame
        gripper_vec_base = cam_to_base_rotation @ gripper_vec_camera

        # Convert back to angle in base frame
        gripper_rotation = np.degrees(
            np.arctan2(gripper_vec_base[1], gripper_vec_base[0])
        )

        return gripper_rotation

    def create_point_cloud_from_depth(self, masked_depth_image, camera_intrinsics):
        """Create point cloud from masked depth image."""
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image),
            camera_intrinsics,
        )
        return np.asarray(pcd.points).astype(np.float32)

    def get_drop_pose_from_points(self, points_camera_frame: np.ndarray):
        """Calculate drop pose from camera frame points."""
        xy_points = points_camera_frame[:, :2].astype(np.float32)

        if len(xy_points) < 3:  # minAreaRect needs at least 3 points
            self.logger.error(
                f"Not enough points ({len(xy_points)}) near drop_z for minAreaRect in release_above. Mask might be too small or object too thin/far."
            )
            return None

        center_camera, _, _ = cv2.minAreaRect(xy_points)
        drop_z = np.mean(points_camera_frame[:, 2])
        grasp_pose_camera = np.array([center_camera[0], center_camera[1], drop_z, 1.0])

        return grasp_pose_camera
