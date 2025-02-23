import json
import logging
import os
import time
from functools import cached_property
from typing import List

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
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Int64, String
from sensor_msgs.msg import JointState

from pymoveit2 import GripperInterface, MoveIt2

from .point_cloud_conversion import point_cloud_to_msg

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# GroundingDINO config and checkpoint
GSA_PATH = "./Grounded-Segment-Anything/Grounded-Segment-Anything"
GROUNDING_DINO_CONFIG_PATH = os.path.join(
    GSA_PATH, "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py")
GROUNDING_DINO_CHECKPOINT_PATH = os.path.join(GSA_PATH,
                                              "groundingdino_swint_ogc.pth")

# Segment-Anything checkpoint
SAM_ENCODER_VERSION = "vit_h"
SAM_CHECKPOINT_PATH = os.path.join(GSA_PATH, "sam_vit_h_4b8939.pth")

# Predict classes and hyper-param for GroundingDINO
BOX_THRESHOLD = 0.4
TEXT_THRESHOLD = 0.25
NMS_THRESHOLD = 0.8


# Prompting SAM with detected boxes
def segment(sam_predictor: SamPredictor, image: np.ndarray,
            xyxy: np.ndarray) -> np.ndarray:
    sam_predictor.set_image(image)
    result_masks = []
    for box in xyxy:
        masks, scores, _ = sam_predictor.predict(box=box,
                                                 multimask_output=True)
        index = np.argmax(scores)
        result_masks.append(masks[index])
    return np.array(result_masks)


class AeraSemiAutonomous(Node):
    def __init__(
            self,
            annotate: bool = False,
            publish_point_cloud: bool = False,
            assistant_id: str = "",
            # Adjust these offsets to your needs:
            offset_x: float = 0.015,
            offset_y: float = -0.015,
            offset_z: float = 0.08,  # accounts for the height of the gripper
    ):
        super().__init__("aera_semi_autonomous_node")

        self.logger = self.get_logger()

        self.cv_bridge = CvBridge()
        self.gripper_joint_name = "gripper_joint"
        callback_group = ReentrantCallbackGroup()
        # Create MoveIt 2 interface
        self.arm_joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"
        ]
        self.moveit2 = MoveIt2(
            node=self,
            joint_names=self.arm_joint_names,
            base_link_name="base_link",
            end_effector_name="link_6",
            group_name="ar_manipulator",
            callback_group=callback_group,
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"
        self.gripper_interface = GripperInterface(
            node=self,
            gripper_joint_names=["gripper_jaw1_joint"],
            open_gripper_joint_positions=[-0.012],
            closed_gripper_joint_positions=[0.0],
            gripper_group_name="ar_gripper",
            callback_group=callback_group,
            gripper_command_action_name="/gripper_controller/gripper_cmd",
        )
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.grounding_dino_model = Model(
            model_config_path=GROUNDING_DINO_CONFIG_PATH,
            model_checkpoint_path=GROUNDING_DINO_CHECKPOINT_PATH,
        )
        self.sam = sam_model_registry[SAM_ENCODER_VERSION](
            checkpoint=SAM_CHECKPOINT_PATH)
        self.sam.to(device=DEVICE)
        self.sam_predictor = SamPredictor(self.sam)

        self.annotate = annotate
        self.publish_point_cloud = publish_point_cloud
        self.n_frames_processed = 0
        self._last_depth_msg = None
        self._last_rgb_msg = None
        self.arm_joint_state: JointState | None = None
        self._last_detections: sv.Detections | None = None
        self._object_in_gripper: bool = False
        self.gripper_squeeze_factor = 0.5
        self.offset_x = offset_x
        self.offset_y = offset_y
        self.offset_z = offset_z

        self.image_sub = self.create_subscription(Image,
                                                  "/camera/color/image_raw",
                                                  self.image_callback, 10)
        self.depth_sub = self.create_subscription(
            Image, "/camera/aligned_depth_to_color/image_raw",
            self.depth_callback, 10)
        self.joint_states_sub = self.create_subscription(
            JointState, "/joint_states", self.joint_states_callback, 10)

        if self.publish_point_cloud:
            self.point_cloud_pub = self.create_publisher(
                PointCloud2, "/point_cloud", 2)
        self.prompt_sub = self.create_subscription(
            String,
            "/prompt",
            self.start,
            10,
            callback_group=MutuallyExclusiveCallbackGroup())
        self.save_images_sub = self.create_subscription(
            String, "/save_images", self.save_images, 10)
        self.detect_objects_sub = self.create_subscription(
            String, "/detect_objects", self.detect_objects_cb, 10)
        self.release_at_sub = self.create_subscription(Int64, "/release_above",
                                                       self.release_above_cb,
                                                       10)
        self.pick_object_sub = self.create_subscription(
            Int64, "/pick_object", self.pick_object_cb, 10)

        self.logger.info("Aera Semi Autonomous node initialized.")


    def handle_tool_call(self, tool_call: str, args: dict,
                         rgb_image: np.ndarray, depth_image: np.ndarray):
        if tool_call == "detect_objects":
            classes_str = args["object_classes"]
            classes = classes_str.split(",")
            self._last_detections = self.detect_objects(rgb_image, classes)
            detected_classes = [
                classes[class_id]
                for class_id in self._last_detections.class_id
            ]
            self.logger.info(f"Detected {detected_classes}.")
            self.logger.info(
                f"detection confidence: {self._last_detections.confidence}")
        elif tool_call == "pick_object":
            if self._last_detections is None:
                logging.error("No detection available")
                return

            if self._object_in_gripper:
                logging.error("Object in gripper")
                return

            if args["object_index"] >= len(self._last_detections.mask):
                logging.error("Invalid object index")
                return

            self.pick_object(args["object_index"], self._last_detections,
                             depth_image)
            self.logger.info(
                f"done picking object. Joint states: {self.arm_joint_state.position}"
            )
            self._object_in_gripper = True
        elif tool_call == "move_above_object_and_release":
            if self._last_detections is None:
                logging.error("No detection available")
                return

            if args["object_index"] >= len(self._last_detections.mask):
                logging.error("Invalid object index")
                return

            self.release_above(args["object_index"], self._last_detections,
                               depth_image)
            self._object_in_gripper = False
        elif tool_call == "release_gripper":
            self.release_gripper()
            self._object_in_gripper = False
        elif tool_call == "flick_wrist_while_release":
            self.flick_wrist_while_release()
            self._object_in_gripper = False

    def start(self, msg: String):
        if not self._last_rgb_msg or not self._last_depth_msg:
            return

        rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
        self._last_detections = None

        self.logger.info(f"Processing: {msg.data}")
        self.logger.info(
            f"Initial Joint states: {self.arm_joint_state.position}")
        tool_call = "detect_objects"
        done = False
        while not done:
            self.handle_tool_call(tool_call, rgb_image, depth_image)

        self.go_home()
        self.logger.info("Task completed.")

    def detect_objects(self, image: np.ndarray, object_classes: List[str]):
        self.logger.info(f"Detecting objects of classes: {object_classes}")
        detections: sv.Detections = self.grounding_dino_model.predict_with_classes(
            image=image,
            classes=object_classes,
            box_threshold=BOX_THRESHOLD,
            text_threshold=TEXT_THRESHOLD,
        )

        # NMS post process
        nms_idx = (torchvision.ops.nms(
            torch.from_numpy(detections.xyxy),
            torch.from_numpy(detections.confidence),
            NMS_THRESHOLD,
        ).numpy().tolist())

        detections.xyxy = detections.xyxy[nms_idx]
        detections.confidence = detections.confidence[nms_idx]
        detections.class_id = detections.class_id[nms_idx]

        detections.mask = segment(
            sam_predictor=self.sam_predictor,
            image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            xyxy=detections.xyxy,
        )

        if self.annotate:
            box_annotator = sv.BoxAnnotator()
            mask_annotator = sv.MaskAnnotator()
            labels = [
                f"{object_classes[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _ in detections
            ]
            annotated_frame = box_annotator.annotate(scene=image.copy(),
                                                     detections=detections,
                                                     labels=labels)
            cv2.imwrite(
                f"annotated_image_detections_{self.n_frames_processed}.jpg",
                annotated_frame)

            annotated_frame = mask_annotator.annotate(scene=image.copy(),
                                                      detections=detections)
            cv2.imwrite(f"annotated_image_masks_{self.n_frames_processed}.jpg",
                        annotated_frame)

        if self.publish_point_cloud:
            depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)
            # mask out the depth image except for the detected objects
            masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
            for mask in detections.mask:
                masked_depth_image[mask] = depth_image[mask]
            masked_depth_image /= 1000.0

            # convert the masked depth image to a point cloud
            pcd = o3d.geometry.PointCloud.create_from_depth_image(
                o3d.geometry.Image(masked_depth_image),
                o3d.camera.PinholeCameraIntrinsic(
                    o3d.camera.PinholeCameraIntrinsicParameters.
                    PrimeSenseDefault),
            )

            # convert it to a ROS PointCloud2 message
            points = np.asarray(pcd.points)
            pc_msg = point_cloud_to_msg(points, "/camera_color_frame")
            self.point_cloud_pub.publish(pc_msg)

        self.n_frames_processed += 1
        return detections

    def pick_object(self, object_index: int, detections: sv.Detections,
                    depth_image: np.ndarray):
        """Perform a top-down grasp on the object."""
        # mask out the depth image except for the detected objects
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image[mask] = depth_image[mask]
        masked_depth_image /= 1000.0

        # convert the masked depth image to a point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
        )
        pcd.transform(self.cam_to_base_affine)
        points = np.asarray(pcd.points)
        grasp_z = points[:, 2].max()

        near_grasp_z_points = points[points[:, 2] > grasp_z - 0.008]
        xy_points = near_grasp_z_points[:, :2]
        xy_points = xy_points.astype(np.float32)
        center, dimensions, theta = cv2.minAreaRect(xy_points)

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

        gripper_pos = -gripper_opening / 2. * self.gripper_squeeze_factor
        gripper_pos = min(gripper_pos, 0.0)
        self.gripper_interface.move_to_position(gripper_pos)
        self.gripper_interface.wait_until_executed()

        # lift the item
        msg.position.z += 0.12
        self.move_to(msg)
        time.sleep(0.05)

    def release_above(self, object_index: int, detections: sv.Detections,
                      depth_image: np.ndarray):
        """Move the robot arm above the object and release the gripper."""
        masked_depth_image = np.zeros_like(depth_image, dtype=np.float32)
        mask = detections.mask[object_index]
        masked_depth_image[mask] = depth_image[mask]
        masked_depth_image /= 1000.0

        # convert the masked depth image to a point cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(
            o3d.geometry.Image(masked_depth_image),
            o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault),
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
        joint_positions = self.arm_joint_state.position
        joint_positions[4] -= np.deg2rad(25)
        self.moveit2.move_to_configuration(joint_positions,
                                           self.arm_joint_names,
                                           tolerance=0.005)
        time.sleep(3)

        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()
        self.moveit2.wait_until_executed()

    def go_home(self):
        joint_positions = [0., 0., 0., 0., 0., 0.]
        self.moveit2.move_to_configuration(joint_positions,
                                           self.arm_joint_names,
                                           tolerance=0.005)
        self.moveit2.wait_until_executed()

    @cached_property
    def cam_to_base_affine(self):
        cam_to_base_link_tf = self.tf_buffer.lookup_transform(
            target_frame="base_link",
            source_frame="camera_color_frame",
            time=Time(),
            timeout=Duration(seconds=5))
        cam_to_base_rot = Rotation.from_quat([
            cam_to_base_link_tf.transform.rotation.x,
            cam_to_base_link_tf.transform.rotation.y,
            cam_to_base_link_tf.transform.rotation.z,
            cam_to_base_link_tf.transform.rotation.w,
        ])
        cam_to_base_pos = np.array([
            cam_to_base_link_tf.transform.translation.x,
            cam_to_base_link_tf.transform.translation.y,
            cam_to_base_link_tf.transform.translation.z,
        ])
        affine = np.eye(4)
        affine[:3, :3] = cam_to_base_rot.as_matrix()
        affine[:3, 3] = cam_to_base_pos
        return affine

    def move_to(self, msg: Pose):
        pose_goal = PoseStamped()
        pose_goal.header.frame_id = "base_link"
        pose_goal.pose = msg

        self.moveit2.move_to_pose(pose=pose_goal)
        self.moveit2.wait_until_executed()

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
            class_names[class_id]
            for class_id in self._last_detections.class_id
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

    def joint_states_callback(self, msg: JointState):
        joint_state = JointState()
        joint_state.header = msg.header
        for name in self.arm_joint_names:
            for i, joint_state_joint_name in enumerate(msg.name):
                if name == joint_state_joint_name:
                    joint_state.name.append(name)
                    joint_state.position.append(msg.position[i])
                    joint_state.velocity.append(msg.velocity[i])
                    joint_state.effort.append(msg.effort[i])

        self.arm_joint_state = joint_state

    def save_images(self, msg):
        if not self._last_rgb_msg or not self._last_depth_msg:
            return

        rgb_image = self.cv_bridge.imgmsg_to_cv2(self._last_rgb_msg)
        depth_image = self.cv_bridge.imgmsg_to_cv2(self._last_depth_msg)

        save_dir = msg.data
        cv2.imwrite(
            os.path.join(save_dir, f"rgb_image_{self.n_frames_processed}.png"),
            rgb_image)
        np.save(
            os.path.join(save_dir, f"depth_image_{self.n_frames_processed}"),
            depth_image)


def main():
    rclpy.init()
    node = AeraSemiAutonomous()
    executor = MultiThreadedExecutor(4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()


if __name__ == "__main__":
    main()