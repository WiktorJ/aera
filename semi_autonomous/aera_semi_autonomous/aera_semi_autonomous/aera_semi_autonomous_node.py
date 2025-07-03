import json
import logging
import os
import threading
import time
from functools import cached_property
from typing import List

import faulthandler
import cv2
import numpy as np
import open3d as o3d
import rclpy
import supervision as sv
import tf2_ros
import torch
import torchvision
from cv_bridge import CvBridge
from geometry_msgs.msg import Pose, PoseStamped
from groundingdino.util.inference import Model
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from segment_anything import SamPredictor, sam_model_registry
from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Int64, String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
import matplotlib.pyplot as plt

from pymoveit2 import GripperInterface, MoveIt2

from .point_cloud_conversion import point_cloud_to_msg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GroundingDINO config and checkpoint
GSA_PATH = "./Grounded-Segment-Anything/Grounded-Segment-Anything"
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
)
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH, "groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "sam_vit_h_4b8939.pth")

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8
_PICK_OBJECT = "pick_object"
_DETECT_OBJECT = "detect_object"
_MOVE_ABOVE_OBJECT_AND_RELEASE = "move_above_object_and_release"
_RELEASE_GRIPPER = "release_gripper"
_FLICK_WRIST_WHILE_RELEASE = "flick_wrist_while_release"

_AVAILABLE_ACTIONS = (
    _PICK_OBJECT,
    _DETECT_OBJECT,
    _MOVE_ABOVE_OBJECT_AND_RELEASE,
    _RELEASE_GRIPPER,
    _FLICK_WRIST_WHILE_RELEASE,
)

# For only single object supported.
_OBJECT_DETECTION_INDEX = 0
_TF_PREFIX = "camera"
BASE_LINK_NAME = f"{_TF_PREFIX}base_link"


# Prompting SAM with detected boxes
def segment(
    sam_predictor: SamPredictor, image: np.ndarray, xyxy: np.ndarray
) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box, multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


class AeraSemiAutonomous(Node):
    def __init__(
        self,
        annotate: bool = False,
        publish_point_cloud: bool = False,
        # Adjust these offsets to your needs:
        offset_x: float = 0.015,
        offset_y: float = -0.015,
        offset_z: float = 0.12,  # accounts for the height of the gripper
    ):
        super().__init__("aera_semi_autonomous_node")

        self.logger = self.get_logger()
        self.debug_visualizations = False
        self.save_debug_images = True
        self.cv_bridge = CvBridge()
        self.gripper_joint_name = "gripper_joint"
        arm_callback_group = ReentrantCallbackGroup()
        gripper_callback_group = ReentrantCallbackGroup()
        promtpt_callback_group = MutuallyExclusiveCallbackGroup()
        # Create MoveIt 2 interface
        self.arm_joint_names = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        self.arm_joint_names = [
            f"{_TF_PREFIX}{joint_name}" for joint_name in self.arm_joint_names
        ]
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name=BASE_LINK_NAME,
            end_effector_name=f"{_TF_PREFIX}link_6",
            group_name="ar_manipulator",
            callback_group=arm_callback_group,
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.gripper_interface = GripperInterface(
            node=self,
            gripper_joint_names=[f"{_TF_PREFIX}gripper_jaw1_joint"],
            open_gripper_joint_positions=[-0.012],
            closed_gripper_joint_positions=[0.0],
            gripper_group_name="ar_gripper",
            callback_group=arm_callback_group,
            gripper_command_action_name="/gripper_controller/gripper_cmd",
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](
            checkpoint=SAM_CHECKPOINT_PATH
        )
        self.sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(self.sam)

        self.annotate = annotate
        self.publish_point_cloud = publish_point_cloud
        self.n_frames_processed = 0
        self._last_depth_msg = None
        self._last_rgb_msg = None
        self._last_detections: sv.Detections | None = None
        self._object_in_gripper: bool = False
        self.gripper_squeeze_factor = 0.5
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z
        self.camera_intrinsics = None
        self.image_width = None
        self.image_height = None

        self.image_sub = self.create_subscription(
            Image, "/camera/camera/color/image_raw", self.image_callback, 10
        )
        # self.depth_sub = self.create_subscription(
        #     Image, "/camera/aligned_depth_to_color/image_raw",
        #     self.depth_callback, 10)
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            "/camera/camera/color/camera_info",
            self.camera_info_callback,
            10,
        )
        self.depth_sub = self.create_subscription(
            Image, "/camera/camera/depth/image_rect_raw", self.depth_callback, 10
        )

        if self.publish_point_cloud:
            self.point_cloud_pub = self.create_publisher(PointCloud2, "/point_cloud", 2)
        self.prompt_sub = self.create_subscription(
            String,
            "/prompt",
            self.start,
            10,
            callback_group=promtpt_callback_group,
        )
        self.save_images_sub = self.create_subscription(
            String, "/save_images", self.save_images, 10
        )
        self.detect_objects_sub = self.create_subscription(
            String, "/detect_objects", self.detect_objects_cb, 10
        )
        self.release_at_sub = self.create_subscription(
            Int64, "/release_above", self.release_above_cb, 10
        )
        self.pick_object_sub = self.create_subscription(
            Int64, "/pick_object", self.pick_object_cb, 10
        )

        self.logger.info("Aera Semi Autonomous node initialized.")

    def camera_info_callback(self, msg: CameraInfo):
        if self.camera_intrinsics is None:
            self.logger.info("Received camera intrinsics.")
            self.image_width = msg.width
            self.image_height = msg.height
            # K is a 3x3 matrix (row-major order in a list of 9)
            # K = [fx, 0,  cx,
            #      0,  fy, cy,
            #      0,  0,  1]
            self.camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
                width=msg.width,
                height=msg.height,
                fx=msg.k[0],
                fy=msg.k[4],
                cx=msg.k[2],
                cy=msg.k[5],
            )
            # Unsubscribe after getting the info because it's static
            self.destroy_subscription(self.camera_info_sub)

    def _setup_debug_logging(self, tool_call: str, object_to_detect: str):
        if self.save_debug_images:
            timestamp = time.strftime("%Y%m%d-%H%M%S")
            self.debug_img_dir = os.path.join(".debug_img_logs", timestamp)
            os.makedirs(self.debug_img_dir, exist_ok=True)
            with open(os.path.join(self.debug_img_dir, "log.txt"), "w") as f:
                f.write(f"Timestamp: {timestamp}\n")
                f.write(f"Tool Call: {tool_call}\n")
                f.write(f"Object to Detect: {object_to_detect}\n")
                if self.camera_intrinsics:
                    f.write("\n--- Camera Intrinsics ---\n")
                    f.write(f"{self.camera_intrinsics}\n")
        else:
            self.debug_img_dir = None

    def _log_debug_info(self, message: str):
        if self.save_debug_images and self.debug_img_dir:
            with open(os.path.join(self.debug_img_dir, "log.txt"), "a") as f:
                f.write(message)

    def _save_debug_image(self, filename: str, image: np.ndarray):
        if self.save_debug_images and self.debug_img_dir:
            cv2.imwrite(os.path.join(self.debug_img_dir, filename), image)

    def _save_debug_plot(self, filename: str):
        if self.save_debug_images and self.debug_img_dir:
            plt.savefig(os.path.join(self.debug_img_dir, filename))

    def handle_tool_call(
        self,
        tool_call: str,
        object_to_detect: str,
        rgb_image: np.ndarray,
        depth_image: np.ndarray,
    ):
        self._setup_debug_logging(tool_call, object_to_detect)

        if self.camera_intrinsics is None:
            self.logger.error(
                "Camera intrinsics not yet received. Cannot create point cloud."
            )
            return
        if tool_call == _DETECT_OBJECT:
            self._last_detections = self.detect_objects(rgb_image, [object_to_detect])
            if len(self._last_detections.class_id) == 0:
                self.logger.info(
                    f"No {object_to_detect} detected. Got the following detection: {self._last_detections.class_id}"
                )
                return
        elif tool_call == _PICK_OBJECT:
            if self._last_detections is None or len(self._last_detections.mask) == 0:
                # logging.error("No detection available")
                # return
                self._last_detections = self.detect_objects(
                    rgb_image, [object_to_detect]
                )
                if len(self._last_detections.class_id) == 0:
                    self.logger.info(
                        f"No {object_to_detect} detected. Got the following detection: {self._last_detections.class_id}"
                    )
                    return
            if self._object_in_gripper:
                logging.error("Object in gripper")
                return

            self.pick_object(
                _OBJECT_DETECTION_INDEX, self._last_detections, depth_image
            )
            self.logger.info(
                f"done picking object. Joint states: {self.moveit2.joint_state.position}"
            )
            self._object_in_gripper = True
        elif tool_call == _MOVE_ABOVE_OBJECT_AND_RELEASE:
            if self._last_detections is None or len(self._last_detections.mask) == 0:
                logging.error("No detection available")
                return

            self.release_above(
                _OBJECT_DETECTION_INDEX, self._last_detections, depth_image
            )
            self._object_in_gripper = False
        elif tool_call == _RELEASE_GRIPPER:
            self.release_gripper()
            self._object_in_gripper = False
        elif tool_call == _FLICK_WRIST_WHILE_RELEASE:
            self.flick_wrist_while_release()
            self._object_in_gripper = False

    def start(self, msg: String):
        if not self._last_rgb_msg or not self._last_depth_msg:
            self.logger.warn(
                f"rgb_msg present: {self._last_rgb_msg is not None}, depth_msg present: {self._last_depth_msg is not None}"
            )
            return
        if msg.data not in _AVAILABLE_ACTIONS:
            self.logger.warn(
                f"Action: {msg} is not valid. Valid actions: {_AVAILABLE_ACTIONS}"
            )
            return

        rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        self._last_detections = None

        self.logger.info(f"Processing: {msg.data}")
        self.logger.info(f"Initial Joint states: {self.moveit2.joint_state.position}")
        # done = False
        # Hardcoded for now
        object_to_detect = "pen"
        # while not done:
        self.handle_tool_call(msg.data, object_to_detect, rgb_image, depth_image)

        self.go_home()
        self.logger.info("Task completed.")

    def detect_objects(self, image: np.ndarray, object_classes: List[str]):
        self.logger.info(f"Detecting objects of classes: {object_classes}")
        rgb_image_for_dino = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # DINO might prefer RGB

        detections: sv.Detections = self.grounding_dino_model.predict_with_classes(
            image=rgb_image_for_dino,
            classes=object_classes,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # NMS post process
        nms_idx = (
            torchvision.ops.nms(
                torch.from_numpy(detections.xyxy),
                torch.from_numpy(detections.confidence),
                NMS_THRESHOLD,
            )
            .numpy()
            .tolist()
        )

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        if len(detections.xyxy) > 0:  # Check if there are any detections
            nms_idx = (
                torchvision.ops.nms(
                    torch.from_numpy(detections.xyxy),
                    torch.from_numpy(detections.confidence),
                    NMS_THRESHOLD,
                )
                .numpy()
                .tolist()
            )
            detections.xyxy = detections.xyxy[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            # Ensure class_id is numpy array for indexing if it's a tensor
            if isinstance(detections.class_id, torch.Tensor):
                detections.class_id = detections.class_id.cpu().numpy()
            detections.class_id = detections.class_id[nms_idx]
        else:
            self.logger.warn("No initial detections before NMS.")
            detections.mask = np.array([])  # Ensure mask is empty if no detections
            return detections  # Return empty detections

        if len(detections.xyxy) == 0:  # Check after NMS
            self.logger.warn("No detections after NMS.")
            detections.mask = np.array([])
            return detections

        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),  # SAM expects RGB
            xyxy=detections.xyxy,
        )

        if self.annotate or self.debug_visualizations or self.save_debug_images:
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
                        custom_labels.append(
                            f"ID:{class_id_val} C:{confidence_val:0.2f}"
                        )
            else:  # If no class_ids (e.g. all detections filtered out by NMS on class_id)
                for i in range(
                    len(detections.xyxy)
                ):  # Make generic labels if no class_ids
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
                    self._save_debug_image(
                        f"debug_annotated_dino_boxes_labels_{self.n_frames_processed}.jpg",
                        annotated_dino_frame_with_labels,
                    )
                    log_message = "\n--- Detections ---\n"
                    for label in custom_labels:
                        log_message += f"{label}\n"
                    self._log_debug_info(log_message)
                if self.debug_visualizations:
                    cv2.imshow("DINO BBoxes & Labels", annotated_dino_frame_with_labels)
            else:  # If no labels, just show the boxes
                if self.save_debug_images:
                    self._save_debug_image(
                        f"debug_annotated_dino_boxes_{self.n_frames_processed}.jpg",
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
                    self._save_debug_image(
                        f"debug_annotated_sam_masks_{self.n_frames_processed}.jpg",
                        annotated_sam_frame,
                    )
                if self.debug_visualizations:
                    cv2.imshow("SAM Masks", annotated_sam_frame)
                    # cv2.waitKey(0)
            else:
                self.logger.info("No SAM masks to annotate.")

            if self.debug_visualizations:
                cv2.waitKey(1)  # Small delay to allow windows to updateÏ

        self.n_frames_processed += 1
        self.logger.info(f"Detected {detections}.")
        if self.save_debug_images:
            self._log_debug_info(f"detection confidence: {detections.confidence}\n")
        self.logger.info(f"detection confidence: {detections.confidence}")
        return detections

    def pick_object(
        self, object_index: int, detections: sv.Detections, depth_image: np.ndarray
    ):
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

        if (self.debug_visualizations or self.save_debug_images) and self._last_rgb_msg:
            current_rgb_msg_to_use = self._last_rgb_msg  # Use a local variable

            if current_rgb_msg_to_use is None:
                self.logger.warn(
                    "PICK_OBJECT: self._last_rgb_msg is None right before cv_bridge call!"
                )

            expected_data_len = (
                current_rgb_msg_to_use.step * current_rgb_msg_to_use.height
            )
            if len(current_rgb_msg_to_use.data) != expected_data_len:
                self.logger.warn(
                    f"PICK_OBJECT: RGB message data length mismatch! Expected {expected_data_len}, Got {len(current_rgb_msg_to_use.data)}"
                )

            try:
                rgb_image_for_viz = self.cv_bridge.imgmsg_to_cv2(
                    self._last_rgb_msg, "bgr8"
                )
                single_mask_viz = rgb_image_for_viz.copy()
                # Create a colored overlay for the mask
                color_mask = np.zeros_like(single_mask_viz)
                current_mask = detections.mask[object_index]  # This is a boolean mask
                color_mask[current_mask] = [0, 255, 0]  # Green for the selected mask
                single_mask_viz = cv2.addWeighted(
                    single_mask_viz, 0.7, color_mask, 0.3, 0
                )
                if self.debug_visualizations:
                    cv2.imshow(
                        f"Selected Mask (Index {object_index}) for Pick",
                        single_mask_viz,
                    )
                if self.save_debug_images:
                    self._save_debug_image(
                        f"debug_selected_mask_pick_{self.n_frames_processed}.jpg",
                        single_mask_viz,
                    )
            except Exception as e:
                self.logger.warn(
                    f"PICK_OBJECT: cv_bridge.imgmsg_to_cv2 FAILED! Error: {e}"
                )
                self.logger.warn(
                    f"PICK_OBJECT: Failing message details again: encoding={current_rgb_msg_to_use.encoding}, H={current_rgb_msg_to_use.height}, W={current_rgb_msg_to_use.width}, step={current_rgb_msg_to_use.step}, data_len={len(current_rgb_msg_to_use.data)}"
                )
                # Also log the exception traceback fully
                import traceback

                self.logger.warn(f"PICK_OBJECT: Traceback: {traceback.format_exc()}")
            # cv2.waitKey(0)

        # mask out the depth image except for the detected objects
        # masked_depth_image[mask] = depth_image[mask]
        # masked_depth_image /= 1000.0
        masked_depth_image_mm = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image_mm[mask] = depth_image[mask]  # Apply mask
        masked_depth_image_mm /= 1000.0

        if self.debug_visualizations or self.save_debug_images:
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
                self._save_debug_image(
                    f"debug_masked_depth_pick_{self.n_frames_processed}.jpg",
                    display_depth_norm,
                )
            if self.debug_visualizations:
                cv2.imshow("Masked Depth (for pick)", display_depth_norm)
            # cv2.waitKey(0)

        # pcd = o3d.geometry.PointCloud.create_from_depth_image(
        #     o3d.geometry.Image(masked_depth_image.astype(np.uint16)),
        #     self.camera_intrinsics,
        #     depth_scale=1000.0,
        #     depth_trunc=3.0,  # Max depth to consider, adjust as needed
        #     stride=1
        # )
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image_mm.astype(np.float32)),
            self.camera_intrinsics,
        )

        if self.debug_visualizations or self.save_debug_images:
            self.logger.info(f"Calculated PointCloud: {pcd}")
            if not hasattr(self, "pcd_cam_frame_pub"):
                self.pcd_cam_frame_pub = self.create_publisher(
                    PointCloud2, "/debug/pcd_camera_frame", 10
                )
            if len(pcd.points) > 0:
                # For Open3D >= 0.13.0, use pcd.get_rotation_matrix_from_xyz for frame convention
                # Create a coordinate frame
                # coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0,0,0])
                # o3d.visualization.draw_geometries([pcd, coord_frame]) # This blocks, useful for direct debug

                # Publish to ROS for RViz
                points_np = np.asarray(pcd.points)
                ros_pcd_cam = point_cloud_to_msg(
                    points_np, "camera_color_optical_frame"
                )  # Use YOUR point_cloud_to_msg
                self.pcd_cam_frame_pub.publish(ros_pcd_cam)
                self.logger.info(
                    f"Published debug PCD in camera frame with {len(points_np)} points."
                )
            else:
                self.logger.info("PCD in camera frame is empty for pick_object.")

        # convert the masked depth image to a point cloud
        pcd.transform(self.cam_to_base_affine)
        points_base_frame = np.asarray(pcd.points)

        if len(points_base_frame) == 0:
            self.logger.error(
                "No points in point cloud after transform to base frame for pick_object. Check TF or if mask resulted in empty depth."
            )
            return

        if self.debug_visualizations or self.save_debug_images:
            self.logger.info(f"Transformed PointCloud: {pcd}")
            if not hasattr(self, "pcd_base_frame_pub"):
                self.pcd_base_frame_pub = self.create_publisher(
                    PointCloud2, "/debug/pcd_base_frame", 10
                )

            ros_pcd_base = point_cloud_to_msg(points_base_frame, BASE_LINK_NAME)
            self.pcd_base_frame_pub.publish(ros_pcd_base)
            self.logger.info(
                f"Published debug PCD in base frame with {len(points_base_frame)} points."
            )
        z_coords = points_base_frame[:, 2]
        # Calculate grasp_z by filtering outliers from the top 25% of points.
        top_z_coords = z_coords[z_coords >= np.percentile(z_coords, 75)]
        if top_z_coords.size > 1:
            mean_z = np.mean(top_z_coords)
            std_z = np.std(top_z_coords)
            # Discard points more than 1 std from the mean.
            filtered_z_coords = top_z_coords[np.abs(top_z_coords - mean_z) <= std_z]
            if filtered_z_coords.size > 0:
                grasp_z = np.mean(filtered_z_coords)
            else:
                # Fallback if all points were filtered out.
                grasp_z = mean_z
        elif top_z_coords.size > 0:
            grasp_z = top_z_coords[0]
        else:
            # Fallback if there are no points in the top percentile (e.g., all points are the same).
            grasp_z = np.mean(z_coords)
        # Filter points near this top surface
        near_grasp_z_points = points_base_frame[
            points_base_frame[:, 2] > grasp_z - 0.01
        ]

        if len(near_grasp_z_points) < 3:  # minAreaRect needs at least 3 points
            self.logger.error(
                f"Not enough points ({len(near_grasp_z_points)}) near grasp_z for minAreaRect. Mask might be too small or object too thin/far."
            )
            # You might want to try using all points_base_frame if near_grasp_z_points is empty
            # or use a simpler centroid if minAreaRect fails
            if len(points_base_frame) > 0:
                self.logger.info(
                    "Falling back to centroid of all points in base frame."
                )
                center_x = np.mean(points_base_frame[:, 0])
                center_y = np.mean(points_base_frame[:, 1])
                center = (center_x, center_y)
                dimensions = (0.01, 0.01)  # dummy
                theta = 0.0
            else:
                return  # No points at all
        else:
            xy_points = near_grasp_z_points[:, :2].astype(
                np.float32
            )  # Get XY coords in base frame
            center, dimensions, theta = cv2.minAreaRect(
                xy_points
            )  # center is (x,y) tuple in base frame

            if self.debug_visualizations or self.save_debug_images:
                plt.figure("XY points for minAreaRect (Base Frame)")
                plt.clf()  # Clear previous plot
                plt.scatter(
                    xy_points[:, 0],
                    xy_points[:, 1],
                    s=5,
                    label="Object Top Surface XY Points",
                )

                # Reconstruct the rotated rectangle from minAreaRect output
                box = cv2.boxPoints(
                    ((center[0], center[1]), (dimensions[0], dimensions[1]), theta)
                )
                # box = np.int0(box)  # This conversion might not be needed if just plotting lines
                # For plotting, better to keep it float and close the loop
                box_plot = np.vstack([box, box[0]])  # Close the rectangle for plotting

                plt.plot(box_plot[:, 0], box_plot[:, 1], "r-", label="minAreaRect BBox")
                plt.scatter(
                    center[0],
                    center[1],
                    c="g",
                    s=50,
                    marker="x",
                    label="Calculated Center (Base Frame)",
                )
                plt.xlabel("X (Base Frame)")
                plt.ylabel("Y (Base Frame)")
                plt.title(f"Object Top XY in Base Frame (Z ~ {grasp_z:.3f}m)")
                plt.axis("equal")  # Important for correct aspect ratio
                plt.legend()
                plt.grid(True)
                if self.save_debug_images:
                    self._save_debug_plot(
                        f"debug_minarearect_xy_{self.n_frames_processed}.png"
                    )
                if self.debug_visualizations:
                    plt.show(block=False)  # Use block=False for non-blocking
                    plt.pause(0.01)  # Allow plot to render

        gripper_rotation = theta
        if dimensions[0] > dimensions[1]:
            gripper_rotation -= 90
        if gripper_rotation < -90:
            gripper_rotation += 180
        elif gripper_rotation > 90:
            gripper_rotation -= 180

        gripper_opening = min(dimensions)
        grasp_pose = Pose()
        grasp_pose.position.x = center[0] + self.offset_x
        grasp_pose.position.y = center[1] + self.offset_y
        grasp_pose.position.z = grasp_z + self.offset_z
        top_down_rot = Rotation.from_quat([0, 1, 0, 0])
        extra_rot = Rotation.from_euler("z", gripper_rotation, degrees=True)
        grasp_quat = (extra_rot * top_down_rot).as_quat()
        grasp_pose.orientation.x = grasp_quat[0]
        grasp_pose.orientation.y = grasp_quat[1]
        grasp_pose.orientation.z = grasp_quat[2]
        grasp_pose.orientation.w = grasp_quat[3]

        if self.debug_visualizations or self.save_debug_images:
            if not hasattr(self, "grasp_pose_pub_pick"):
                self.grasp_pose_pub_pick = self.create_publisher(
                    PoseStamped, "/debug/pick_grasp_pose", 10
                )

            pose_msg = PoseStamped()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = BASE_LINK_NAME  # Should be your robot's base
            pose_msg.pose = grasp_pose
            self.grasp_pose_pub_pick.publish(pose_msg)
            self.logger.info(
                f"Published debug pick_grasp_pose: {grasp_pose.position.x:.3f}, {grasp_pose.position.y:.3f}, {grasp_pose.position.z:.3f}"
            )
            if self.save_debug_images:
                log_message = (
                    "\n--- Grasp Pose ---\n"
                    f"Position (x, y, z): {grasp_pose.position.x:.4f}, {grasp_pose.position.y:.4f}, {grasp_pose.position.z:.4f}\n"
                    f"Orientation (x, y, z, w): {grasp_pose.orientation.x:.4f}, {grasp_pose.orientation.y:.4f}, {grasp_pose.orientation.z:.4f}, {grasp_pose.orientation.w:.4f}\n"
                )
                self._log_debug_info(log_message)

        if self.debug_visualizations:
            cv2.waitKey(1)  # Give OpenCV windows a chance to update

        self.grasp_at(grasp_pose, gripper_opening)

    def grasp_at(self, msg: Pose, gripper_opening: float):
        self.logger.info(f"Grasp at: {msg} with opening: {gripper_opening}")

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

        # move 5cm above the item first
        msg.position.z += 0.05
        self.move_to(msg)
        time.sleep(0.05)

        # grasp the item
        msg.position.z -= 0.05
        self.move_to(msg)
        time.sleep(0.05)

        gripper_pos = -gripper_opening / 2.0 * self.gripper_squeeze_factor
        gripper_pos = min(gripper_pos, 0.0)
        if self.save_debug_images:
            log_message = (
                "\n--- Gripper Position ---\n"
                f"Target Gripper Position: {gripper_pos:.4f}\n"
            )
            self._log_debug_info(log_message)
        self.gripper_interface.move_to_position(gripper_pos)
        self.gripper_interface.wait_until_executed()

        # lift the item
        msg.position.z += 0.12
        self.move_to(msg)
        time.sleep(0.05)

    def release_above(
        self, object_index: int, detections: sv.Detections, depth_image: np.ndarray
    ):
        """Move the robot arm above the object and release the gripper."""
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image[mask] = depth_image[mask]
        # masked_depth_image /= 1000.0

        # convert the masked depth image to a point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image.astype(np.uint16)),
            self.camera_intrinsics,
            depth_scale=1000.0,
            depth_trunc=3.0,  # Max depth to consider, adjust as needed
            stride=1,
        )
        pcd.transform(self.cam_to_base_affine)

        points = np.asarray(pcd.points).astype(np.float32)
        # release 5cm above the object
        drop_z = np.percentile(points[:, 2], 95) + 0.05
        median_z = np.median(points[:, 2])

        xy_points = points[points[:, 2] > median_z, :2]
        xy_points = xy_points.astype(np.float32)
        center, _, _ = cv2.minAreaRect(xy_points)

        drop_pose = Pose()
        drop_pose.position.x = center[0] + self.offset_x
        drop_pose.position.y = center[1] + self.offset_y
        drop_pose.position.z = drop_z + self.offset_z
        # Straight down pose
        drop_pose.orientation.x = 0.0
        drop_pose.orientation.y = 1.0
        drop_pose.orientation.z = 0.0
        drop_pose.orientation.w = 0.0

        self.release_at(drop_pose)

    def release_gripper(self):
        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

    def flick_wrist_while_release(self):
        if self.moveit2.joint_state is None:
            self.logger.error("Cannot flick wrist, arm joint state is not available.")
            return
        joint_positions = self.moveit2.joint_state.position
        joint_positions[4] -= np.deg2rad(25)
        trajectory = self.moveit2.plan(
            joint_positions=joint_positions,
            joint_names=self.arm_joint_names,
            tolerance_joint_position=0.005,
            start_joint_state=self.moveit2.joint_state,
        )
        if not trajectory:
            self.logger.error("Failed to plan for flick_wrist_while_release")
            return

        self.moveit2.execute(trajectory)
        self.moveit2.wait_until_executed()
        time.sleep(0.05)

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

    def go_home(self):
        if self.moveit2.joint_state is None:
            self.logger.error("Cannot go home, arm joint state is not available.")
            return
        joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        trajectory = self.moveit2.plan(
            joint_positions=joint_positions,
            joint_names=self.arm_joint_names,
            tolerance_joint_position=0.005,
            start_joint_state=self.moveit2.joint_state,
        )
        if trajectory:
            self.moveit2.execute(trajectory)
            self.moveit2.wait_until_executed()
        else:
            self.logger.error("Failed to plan trajectory for go_home.")

    @cached_property
    def cam_to_base_affine(self):
        cam_to_base_link_tf = self.tf_buffer.lookup_transform(
            target_frame=BASE_LINK_NAME,
            source_frame="camera_color_optical_frame",
            # source_frame="camera_color_frame",
            time=Time(),
            timeout=Duration(seconds=5),
        )
        cam_to_base_rot = Rotation.from_quat(
            [
                cam_to_base_link_tf.transform.rotation.x,
                cam_to_base_link_tf.transform.rotation.y,
                cam_to_base_link_tf.transform.rotation.z,
                cam_to_base_link_tf.transform.rotation.w,
            ]
        )
        cam_to_base_pos = np.array(
            [
                cam_to_base_link_tf.transform.translation.x,
                cam_to_base_link_tf.transform.translation.y,
                cam_to_base_link_tf.transform.translation.z,
            ]
        )
        affine = np.eye(4)
        affine[:3, :3] = cam_to_base_rot.as_matrix()
        affine[:3, 3] = cam_to_base_pos
        return affine

    def move_to(self, msg: Pose):
        if self.moveit2.joint_state is None:
            self.logger.error("Cannot move, arm joint state is not available.")
            return

        pose_goal = PoseStamped()
        pose_goal.header.frame_id = BASE_LINK_NAME
        pose_goal.pose = msg

        trajectory = self.moveit2.plan(
            pose=pose_goal, start_joint_state=self.moveit2.joint_state
        )
        if trajectory:
            if self.save_debug_images:
                log_message = "\n--- MoveIt Trajectory ---\n"
                log_message += f"Trajectory: {repr(trajectory)}\n"
                self._log_debug_info(log_message)
            self.moveit2.execute(trajectory)
            self.moveit2.wait_until_executed()
        else:
            self.logger.error("Failed to plan trajectory for move_to.")

    def release_at(self, msg: Pose):
        # NOTE: straight down is wxyz 0, 0, 1, 0
        # good pose is 0, -0.3, 0.35
        self.logger.info(f"Releasing at: {msg}")
        self.move_to(msg)

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

    def detect_objects_cb(self, msg: String):
        if self._last_rgb_msg is None:
            self.logger.warning("No RGB image available.")
            return

        class_names = msg.data.split(",")
        rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
        self._last_detections = self.detect_objects(rgb_image, class_names)
        detected_classes = [
            class_names[class_id] for class_id in self._last_detections.class_id
        ]
        self.logger.info(f"Detected objects: {detected_classes}")

    def pick_object_cb(self, msg: Int64):
        if self._last_detections is None or self._last_depth_msg is None:
            self.logger.warning("No detections or depth image available.")
            return

        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        self.pick_object(msg.data, self._last_detections, depth_image)

    def release_above_cb(self, msg: Int64):
        if self._last_detections is None or self._last_depth_msg is None:
            self.logger.warning("No detections or depth image available.")
            return

        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        self.release_above(msg.data, self._last_detections, depth_image)

    def depth_callback(self, msg):
        self._last_depth_msg = msg

    def image_callback(self, msg):
        self._last_rgb_msg = msg

    def save_images(self, msg):
        if not self._last_rgb_msg or not self._last_depth_msg:
            return

        rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)

        save_dir = msg.data
        cv2.imwrite(
            os.path.join(save_dir, f"rgb_image_{self.n_frames_processed}.png"),
            rgb_image,
        )
        np.save(
            os.path.join(save_dir, f"depth_image_{self.n_frames_processed}"),
            depth_image,
        )


def main():
    faulthandler.enable()
    rclpy.init()
    node = AeraSemiAutonomous()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        while executor.context.ok() and not executor._is_shutdown:
            executor.spin_once(1)
        # executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()
