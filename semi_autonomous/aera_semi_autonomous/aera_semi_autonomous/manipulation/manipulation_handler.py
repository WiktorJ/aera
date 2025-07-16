import numpy as np
from geometry_msgs.msg import Pose
from scipy.spatial.transform import Rotation


class ManipulationHandler:
    def __init__(self, point_cloud_processor, robot_controller, debug_utils, 
                 camera_intrinsics, cam_to_base_affine, offset_x, offset_y, offset_z, 
                 gripper_squeeze_factor, n_frames_processed=0):
        self.point_cloud_processor = point_cloud_processor
        self.robot_controller = robot_controller
        self.debug_utils = debug_utils
        self.camera_intrinsics = camera_intrinsics
        self.cam_to_base_affine = cam_to_base_affine
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.gripper_squeeze_factor = gripper_squeeze_factor
        self.n_frames_processed = n_frames_processed
        self.logger = robot_controller.logger

    def pick_object(self, object_index: int, detections, depth_image: np.ndarray):
        """Perform a top-down grasp on the object."""
        if (
            detections is None
            or detections.mask is None
            or object_index >= len(detections.mask)
        ):
            self.logger.error(
                f"Invalid detections or object_index for pick_object. Index: {object_index}, Num Masks: {len(detections.mask) if detections.mask is not None else 'None'}"
            )
            return

        self.debug_utils.debug_visualize_selected_mask(detections, object_index, "Pick", None)

        masked_depth_image_mm = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image_mm[mask] = depth_image[mask]  # Apply mask
        masked_depth_image_mm /= 1000.0

        self.debug_utils.debug_visualize_masked_depth(masked_depth_image_mm, "Pick", self.n_frames_processed)

        # Create point cloud in camera frame
        points_camera_frame = self.point_cloud_processor.create_point_cloud_from_depth(
            masked_depth_image_mm, self.camera_intrinsics
        )

        self.debug_utils.debug_visualize_all_minarearects(points_camera_frame, "Pick_Camera", self.n_frames_processed)

        if len(points_camera_frame) < 3:  # minAreaRect needs at least 3 points
            self.logger.error(
                f"Not enough points ({len(points_camera_frame)}) near grasp_z for minAreaRect. Mask might be too small or object too thin/far."
            )
            return

        grasp_pose_camera, gripper_angle_camera, gripper_opening = (
            self.point_cloud_processor.get_pose_and_angle_camera_base(points_camera_frame)
        )
        grasp_pose_base = self.cam_to_base_affine @ grasp_pose_camera

        # Convert angle to unit vector in camera frame
        gripper_rotation = self.point_cloud_processor.transform_gripper_angle_to_base_frame(
            gripper_angle_camera, self.cam_to_base_affine
        )
        # Normalize angle to [-90, 90] range
        if gripper_rotation < -90:
            gripper_rotation += 180
        elif gripper_rotation > 90:
            gripper_rotation -= 180

        gripper_pos = -gripper_opening / 2.0 * self.gripper_squeeze_factor
        gripper_pos = min(gripper_pos, 0.0)

        # Create final grasp pose in base frame
        grasp_pose = Pose()
        grasp_pose.position.x = grasp_pose_base[0] + self.offset_x
        grasp_pose.position.y = grasp_pose_base[1] + self.offset_y
        grasp_pose.position.z = grasp_pose_base[2] + self.offset_z

        top_down_rot = Rotation.from_quat([0, 1, 0, 0])
        extra_rot = Rotation.from_euler("z", gripper_rotation, degrees=True)
        grasp_quat = (extra_rot * top_down_rot).as_quat()
        grasp_pose.orientation.x = grasp_quat[0]
        grasp_pose.orientation.y = grasp_quat[1]
        grasp_pose.orientation.z = grasp_quat[2]
        grasp_pose.orientation.w = grasp_quat[3]

        self.debug_utils.debug_log_pose_info(grasp_pose, gripper_opening, "Grasp")

        if self.debug_utils.debug_visualizations or self.debug_utils.save_debug_images:
            # Transform points to base frame for visualization
            points_base_frame = np.column_stack([
                points_camera_frame, 
                np.ones(len(points_camera_frame))
            ]) @ self.cam_to_base_affine.T
            points_base_frame = points_base_frame[:, :3]
            self.debug_utils.debug_visualize_all_minarearects(points_base_frame, "Pick_Base", self.n_frames_processed)

        self.robot_controller.grasp_at(grasp_pose, gripper_pos)

    def release_above(self, object_index: int, detections, depth_image: np.ndarray):
        """Move the robot arm above the object and release the gripper."""
        if (
            detections is None
            or detections.mask is None
            or object_index >= len(detections.mask)
        ):
            self.logger.error(
                f"Invalid detections or object_index for release_above. Index: {object_index}, Num Masks: {len(detections.mask) if detections.mask is not None else 'None'}"
            )
            return

        self.debug_utils.debug_visualize_selected_mask(detections, object_index, "Release", None)

        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image[mask] = depth_image[mask]
        masked_depth_image /= 1000.0

        self.debug_utils.debug_visualize_masked_depth(masked_depth_image, "Release", self.n_frames_processed)

        # convert the masked depth image to a point cloud
        points = self.point_cloud_processor.create_point_cloud_from_depth(
            masked_depth_image, self.camera_intrinsics
        )

        self.debug_utils.debug_visualize_all_minarearects(points, "Release_Camera", self.n_frames_processed)

        grasp_pose_camera = self.point_cloud_processor.get_drop_pose_from_points(points)
        if grasp_pose_camera is None:
            return

        drop_pose_base = self.cam_to_base_affine @ grasp_pose_camera

        drop_pose = Pose()
        drop_pose.position.x = drop_pose_base[0] + self.offset_x
        drop_pose.position.y = drop_pose_base[1] + self.offset_y
        drop_pose.position.z = drop_pose_base[2] + self.offset_z

        # Straight down pose
        drop_pose.orientation.x = 0.0
        drop_pose.orientation.y = 1.0
        drop_pose.orientation.z = 0.0
        drop_pose.orientation.w = 0.0

        self.debug_utils.debug_log_pose_info(drop_pose, operation_name="Drop")
        if self.debug_utils.debug_visualizations or self.debug_utils.save_debug_images:
            # Transform points to base frame for visualization
            points_base_frame = np.column_stack([
                points, 
                np.ones(len(points))
            ]) @ self.cam_to_base_affine.T
            points_base_frame = points_base_frame[:, :3]
            self.debug_utils.debug_visualize_all_minarearects(points_base_frame, "Release_Base", self.n_frames_processed)

        self.robot_controller.release_at(drop_pose)
