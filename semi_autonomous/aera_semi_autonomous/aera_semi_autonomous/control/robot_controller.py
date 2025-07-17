import time
from geometry_msgs.msg import Pose, PoseStamped
from pymoveit2 import GripperInterface, MoveIt2
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation
from sensor_msgs.msg import JointState

from ..config.constants import BASE_LINK_NAME


class RobotController:
    def __init__(self, node, arm_joint_names, tf_prefix, debug_mode=False, feedback_callback=None):
        self.node = node
        self.logger = node.get_logger()
        self.debug_mode = debug_mode
        self.feedback_callback = feedback_callback
        self.tf_prefix = tf_prefix

        arm_callback_group = ReentrantCallbackGroup()
        gripper_callback_group = ReentrantCallbackGroup()

        # Initialize MoveIt2 interface
        self.moveit2 = MoveIt2(
            node=node,
            joint_names=arm_joint_names,
            base_link_name=BASE_LINK_NAME,
            end_effector_name=f"{tf_prefix}link_6",
            group_name="ar_manipulator",
            callback_group=arm_callback_group,
        )
        self.moveit2.planner_id = "RRTConnectkConfigDefault"

        # Initialize gripper interface
        self.gripper_interface = GripperInterface(
            node=node,
            gripper_joint_names=[f"{tf_prefix}gripper_jaw1_joint"],
            open_gripper_joint_positions=[-0.012],
            closed_gripper_joint_positions=[0.0],
            gripper_group_name="ar_gripper",
            gripper_command_action_name="/gripper_controller/gripper_cmd",
            callback_group=gripper_callback_group,
        )

    def move_to(self, pose: Pose):
        """Move the robot arm to the specified pose."""
        if self.moveit2.joint_state is None:
            self.logger.error("Cannot move, arm joint state is not available.")
            return

        self.logger.info(
            f"Joint states before move: {self.moveit2.joint_state.position}"
        )

        pose_goal = PoseStamped()
        pose_goal.header.frame_id = BASE_LINK_NAME
        pose_goal.pose = pose

        trajectory = self.moveit2.plan(
            pose=pose_goal, start_joint_state=self.moveit2.joint_state
        )
        if trajectory:
            self.moveit2.execute(trajectory, feedback_callable=self.feedback_callback)
            self.moveit2.wait_until_executed()
        else:
            self.logger.error("Failed to plan trajectory for move_to.")

    def release_gripper(self):
        """Open the gripper to release objects."""
        if self.debug_mode:
            return
        self.gripper_interface.open()
        self.gripper_interface.wait_until_executed()

    def grasp_at(self, pose: Pose, gripper_pos: float):
        """Perform a grasp at the specified pose with the given gripper position."""
        self.logger.info(f"Grasp at: {pose} with opening: {gripper_pos}")
        if self.debug_mode:
            return

        self.release_gripper()

        # move 5cm above the item first
        pose.position.z += 0.05
        self.move_to(pose)
        time.sleep(0.05)

        # grasp the item
        pose.position.z -= 0.05
        self.move_to(pose)
        time.sleep(0.05)

        self.gripper_interface.move_to_position(gripper_pos)
        self.gripper_interface.wait_until_executed()

        # lift the item
        pose.position.z += 0.12
        self.move_to(pose)
        time.sleep(0.05)
        self.logger.info(
            f"done grasping object. Joint states: {self.moveit2.joint_state.position}"
        )

    def release_at(self, pose: Pose):
        """Move to the specified pose and release the gripper."""
        # NOTE: straight down is wxyz 0, 0, 1, 0
        # good pose is 0, -0.3, 0.35
        self.logger.info(f"Releasing at: {pose}")
        if self.debug_mode:
            return
        self.move_to(pose)
        self.release_gripper()

    def compute_end_effector_pose(self, joint_positions=None):
        """Compute end-effector pose from joint positions using MoveIt2 forward kinematics."""
        if joint_positions is None:
            # Use current joint state if no specific positions provided
            if self.moveit2.joint_state is None:
                self.logger.error("Joint state not available for FK computation")
                return None
            joint_state = self.moveit2.joint_state
        else:
            # Create joint state from provided positions
            joint_state = JointState()
            joint_state.name = self.node.arm_joint_names
            joint_state.position = joint_positions

        # Compute forward kinematics using MoveIt2
        pose_stamped = self.moveit2.compute_fk(
            joint_state=joint_state,
            fk_link_names=[f"{self.tf_prefix}link_6"]
        )
        
        if pose_stamped:
            return pose_stamped.pose if hasattr(pose_stamped, 'pose') else pose_stamped[0].pose
        else:
            self.logger.error("Failed to compute forward kinematics")
            return None

    def get_current_end_effector_pose(self):
        """Get current end-effector pose using TF."""
        try:
            transform = self.node.tf_buffer.lookup_transform(
                target_frame=BASE_LINK_NAME,
                source_frame=f"{self.tf_prefix}link_6",
                time=self.node.get_clock().now(),
                timeout=Duration(seconds=1.0)
            )
            
            pose = Pose()
            pose.position.x = transform.transform.translation.x
            pose.position.y = transform.transform.translation.y
            pose.position.z = transform.transform.translation.z
            pose.orientation = transform.transform.rotation
            
            return pose
        except Exception as e:
            self.logger.error(f"Failed to get end-effector pose via TF: {e}")
            return None

    def go_home(self):
        """Move the robot arm to the home position."""
        if self.debug_mode:
            return
        if self.moveit2.joint_state is None:
            self.logger.error("Cannot go home, arm joint state is not available.")
            return

        joint_positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        trajectory = self.moveit2.plan(
            joint_positions=joint_positions,
            joint_names=self.node.arm_joint_names,
            tolerance_joint_position=0.005,
            start_joint_state=self.moveit2.joint_state,
        )
        if trajectory:
            self.moveit2.execute(trajectory, feedback_callable=self.feedback_callback)
            self.moveit2.wait_until_executed()
        else:
            self.logger.error("Failed to plan trajectory for go_home.")
