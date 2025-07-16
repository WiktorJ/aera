import os
import time
import cv2
import numpy as np
import supervision as sv
import matplotlib.pyplot as plt
from typing import List
from cv_bridge import CvBridge


class DebugUtils:
    def __init__(self, logger, save_debug_images=True, debug_visualizations=False):
        self.logger = logger
        self.save_debug_images = save_debug_images
        self.debug_visualizations = debug_visualizations
        self.cv_bridge = CvBridge()
        self.debug_img_dir = None

    def setup_debug_logging(self, input_message: str, camera_intrinsics=None):
        if self.save_debug_images:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.debug_img_dir = os.path.join(".debug_img_logs", timestamp)
            os.makedirs(self.debug_img_dir, exist_ok=True)
            with open(os.path.join(self.debug_img_dir, "log.txt"), "w") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Input Message: {input_message}\n")
                if camera_intrinsics:
                    f.write("\n--- Camera Intrinsics ---\n")
                    f.write(f"{camera_intrinsics}\n")
        else:
            self.debug_img_dir = None

    def log_debug_info(self, message: str):
        if self.save_debug_images and self.debug_img_dir:
            with open(os.path.join(self.debug_img_dir, "log.txt"), "a") as f:
                f.write(message)

    def save_debug_image(self, filename: str, image: np.ndarray):
        if self.save_debug_images and self.debug_img_dir:
            cv2.imwrite(os.path.join(self.debug_img_dir, filename), image)

    def save_debug_plot(self, filename: str):
        if self.save_debug_images and self.debug_img_dir:
            plt.savefig(os.path.join(self.debug_img_dir, filename))

    def debug_visualize_selected_mask(
        self, detections: sv.Detections, object_index: int, operation_name: str, last_rgb_msg
    ):
        """Debug visualization for the selected mask."""
        if (
            not (self.debug_visualizations or self.save_debug_images)
            or not last_rgb_msg
        ):
            return

        try:
            rgb_image_for_viz = self.cv_bridge.imgmsg_to_cv2(last_rgb_msg, "bgr8")
            single_mask_viz = rgb_image_for_viz.copy()
            # Create a colored overlay for the mask
            color_mask = np.zeros_like(single_mask_viz)
            current_mask = detections.mask[object_index]  # This is a boolean mask
            color_mask[current_mask] = [0, 255, 0]  # Green for the selected mask
            single_mask_viz = cv2.addWeighted(single_mask_viz, 0.7, color_mask, 0.3, 0)

            if self.debug_visualizations:
                cv2.imshow(
                    f"Selected Mask (Index {object_index}) for {operation_name}",
                    single_mask_viz,
                )
            if self.save_debug_images:
                self.save_debug_image(
                    f"debug_selected_mask_{operation_name.lower()}_{object_index}.jpg",
                    single_mask_viz,
                )
        except Exception as e:
            self.logger.warn(
                f"{operation_name}: cv_bridge.imgmsg_to_cv2 FAILED! Error: {e}"
            )
            import traceback

            self.logger.warn(f"{operation_name}: Traceback: {traceback.format_exc()}")

    def debug_visualize_masked_depth(
        self, masked_depth_image_mm: np.ndarray, operation_name: str, frame_count: int
    ):
        """Debug visualization for the masked depth image."""
        if not (self.debug_visualizations or self.save_debug_images):
            return

        # Normalize for display (imshow expects 0-255 for uint8 or 0-1 for float)
        display_depth = masked_depth_image_mm.copy()
        if np.any(display_depth > 0):  # Avoid division by zero if all are zero
            display_depth_norm = (
                display_depth - display_depth[display_depth > 0].min()
            ) / (
                display_depth[display_depth > 0].max()
                - display_depth[display_depth > 0].min()
            )
            display_depth_norm = (display_depth_norm * 255).astype(np.uint8)
        else:
            display_depth_norm = np.zeros_like(display_depth, dtype=np.uint8)

        if self.save_debug_images:
            self.save_debug_image(
                f"debug_masked_depth_{operation_name.lower()}_{frame_count}.jpg",
                display_depth_norm,
            )
        if self.debug_visualizations:
            cv2.imshow(
                f"Masked Depth (for {operation_name.lower()})", display_depth_norm
            )
            cv2.waitKey(0)

    def debug_visualize_minarearect(
        self,
        points: np.ndarray,
        center: tuple,
        dimensions: tuple,
        theta: float,
        third_coord: float,
        axes: tuple,
        operation_name: str,
        frame_count: int,
    ):
        """Debug visualization for the minAreaRect calculation."""
        if not (self.debug_visualizations or self.save_debug_images):
            return

        axis1_name, axis2_name = axes
        # Determine the third axis name
        all_axes = {"x", "y", "z"}
        third_axis_name = list(all_axes - {axis1_name.lower(), axis2_name.lower()})[0]

        plt.figure(
            f"{axis1_name.upper()}{axis2_name.upper()} points for minAreaRect (Base Frame) - {operation_name}"
        )
        plt.clf()  # Clear previous plot
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=5,
            label=f"Object {axis1_name.upper()}{axis2_name.upper()} Points",
        )

        # Reconstruct the rotated rectangle from minAreaRect output
        box = cv2.boxPoints(
            ((center[0], center[1]), (dimensions[0], dimensions[1]), theta)
        )
        # Close the rectangle for plotting
        box_plot = np.vstack([box, box[0]])

        plt.plot(box_plot[:, 0], box_plot[:, 1], "r-", label="minAreaRect BBox")
        plt.scatter(
            center[0],
            center[1],
            c="g",
            s=50,
            marker="x",
            label="Calculated Center (Base Frame)",
        )
        plt.xlabel(f"{axis1_name.upper()} (Base Frame)")
        plt.ylabel(f"{axis2_name.upper()} (Base Frame)")
        plt.title(
            f"Object {axis1_name.upper()}{axis2_name.upper()} in Base Frame ({third_axis_name.upper()} ~ {third_coord:.3f}m) - {operation_name}"
        )
        plt.axis("equal")  # Important for correct aspect ratio
        plt.legend()
        plt.grid(True)
        if self.save_debug_images:
            self.save_debug_plot(
                f"debug_minarearect_{axis1_name.lower()}{axis2_name.lower()}_{operation_name.lower()}_{frame_count}.png"
            )
        if self.debug_visualizations:
            plt.show(block=False)  # Use block=False for non-blocking
            plt.pause(0.01)  # Allow plot to render

    def debug_visualize_all_minarearects(
        self, points_base_frame: np.ndarray, operation_name: str, frame_count: int
    ):
        """Debug visualization for minAreaRect calculations on all axis pairs."""
        if not (self.debug_visualizations or self.save_debug_images):
            return

        if len(points_base_frame) < 3:
            self.logger.warn(
                f"Not enough points ({len(points_base_frame)}) for minAreaRect visualizations."
            )
            return

        # XY plane (looking down from above)
        xy_points = points_base_frame[:, :2].astype(np.float32)
        center_xy, dimensions_xy, theta_xy = cv2.minAreaRect(xy_points)
        grasp_z = np.mean(points_base_frame[:, 2])
        self.debug_visualize_minarearect(
            xy_points,
            center_xy,
            dimensions_xy,
            theta_xy,
            grasp_z,
            ("x", "y"),
            operation_name,
            frame_count,
        )

        # XZ plane (side view)
        xz_points = points_base_frame[:, [0, 2]].astype(np.float32)
        center_xz, dimensions_xz, theta_xz = cv2.minAreaRect(xz_points)
        self.debug_visualize_minarearect(
            xz_points,
            center_xz,
            dimensions_xz,
            theta_xz,
            center_xy[1],
            ("x", "z"),
            operation_name,
            frame_count,
        )

        # YZ plane (side view)
        yz_points = points_base_frame[:, [1, 2]].astype(np.float32)
        center_yz, dimensions_yz, theta_yz = cv2.minAreaRect(yz_points)
        self.debug_visualize_minarearect(
            yz_points,
            center_yz,
            dimensions_yz,
            theta_yz,
            center_xy[0],
            ("y", "z"),
            operation_name,
            frame_count,
        )

    def debug_visualize_detections(
        self, image: np.ndarray, detections: sv.Detections, object_classes: List[str], frame_count: int
    ):
        """Debug visualization for object detections with bounding boxes, labels, and masks."""
        # Create a BGR copy for OpenCV annotations
        bgr_image_annotated = image.copy()

        # Annotate DINO boxes
        box_annotator = sv.BoxAnnotator()
        annotated_dino_frame = box_annotator.annotate(
            scene=bgr_image_annotated.copy(), detections=detections
        )

        # Prepare labels carefully, ensuring class_id is valid index for object_classes
        custom_labels = []
        if detections.class_id is not None and len(detections.class_id) > 0:
            for i in range(len(detections.xyxy)):
                class_id_val = detections.class_id[i]
                confidence_val = detections.confidence[i]
                if 0 <= class_id_val < len(object_classes):
                    custom_labels.append(
                        f"{object_classes[class_id_val]} {confidence_val:0.2f}"
                    )
                else:  # Fallback if class_id is out of bounds for object_classes
                    custom_labels.append(f"ID:{class_id_val} C:{confidence_val:0.2f}")
        else:  # If no class_ids (e.g. all detections filtered out by NMS on class_id)
            for i in range(len(detections.xyxy)):  # Make generic labels if no class_ids
                custom_labels.append(f"Det {i} C:{detections.confidence[i]:0.2f}")

        if custom_labels:  # Only annotate labels if we have some
            label_annotator = sv.LabelAnnotator(
                text_scale=0.5,
                text_thickness=1,
                text_padding=3,
                text_position=sv.Position.TOP_CENTER,
            )
            # LabelAnnotator annotates ON TOP of the image passed to it.
            # So, we pass the image already annotated with boxes.
            annotated_dino_frame_with_labels = label_annotator.annotate(
                scene=annotated_dino_frame.copy(),
                # Use copy to avoid modifying prev
                detections=detections,
                labels=custom_labels,
            )
            if self.save_debug_images:
                self.save_debug_image(
                    f"debug_annotated_dino_boxes_labels_{frame_count}.jpg",
                    annotated_dino_frame_with_labels,
                )
                log_message = "\n--- Detections ---\n"
                for label in custom_labels:
                    log_message += f"{label}\n"
                self.log_debug_info(log_message)
            if self.debug_visualizations:
                cv2.imshow("DINO BBoxes & Labels", annotated_dino_frame_with_labels)
        else:  # If no labels, just show the boxes
            if self.save_debug_images:
                self.save_debug_image(
                    f"debug_annotated_dino_boxes_{frame_count}.jpg",
                    annotated_dino_frame,
                )
            if self.debug_visualizations:
                cv2.imshow("DINO BBoxes", annotated_dino_frame)

        # Annotate SAM masks
        if detections.mask is not None and len(detections.mask) > 0:
            mask_annotator = sv.MaskAnnotator(opacity=0.4)
            # Create a fresh copy for mask annotation if you want separate images
            annotated_sam_frame = mask_annotator.annotate(
                scene=bgr_image_annotated.copy(), detections=detections
            )  # Detections obj should have .mask
            if self.save_debug_images:
                self.save_debug_image(
                    f"debug_annotated_sam_masks_{frame_count}.jpg",
                    annotated_sam_frame,
                )
            if self.debug_visualizations:
                cv2.imshow("SAM Masks", annotated_sam_frame)
                cv2.waitKey(0)
        else:
            self.logger.info("No SAM masks to annotate.")

        if self.debug_visualizations:
            cv2.waitKey(1)  # Small delay to allow windows to update

    def debug_log_pose_info(
        self, pose, gripper_opening: float = None, operation_name: str = ""
    ):
        """Debug logging for pose and gripper information."""
        if not self.save_debug_images:
            return

        log_message = f"\n--- {operation_name} Pose ---\n"
        log_message += f"Position (x, y, z): {pose.position.x:.4f}, {pose.position.y:.4f}, {pose.position.z:.4f}\n"
        log_message += f"Orientation (x, y, z, w): {pose.orientation.x:.4f}, {pose.orientation.y:.4f}, {pose.orientation.z:.4f}, {pose.orientation.w:.4f}\n"

        if gripper_opening is not None:
            log_message += f"\n--- Gripper Position ---\n"
            log_message += f"Target Gripper Opening: {gripper_opening:.4f}\n"

        self.log_debug_info(log_message)
