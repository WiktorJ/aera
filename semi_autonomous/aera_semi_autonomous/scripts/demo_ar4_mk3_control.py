#!/usr/bin/env python3
"""
Simple demonstration script for controlling the AR4 MK3 arm simulation
via the Ar4Mk3RobotInterface.
"""

import time
import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion

from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = (
    -0.36336720179946663,
    -0.8203835174702869,
    0.22865474664402222,
    0.37769321910336584,
)


def create_pose(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    """Helper function to create a Pose message."""
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


def main():
    print("Starting AR4 MK3 arm control demonstration...")

    # Create the MuJoCo environment
    # Using pick and place environment with end-effector control
    import os

    # Get the absolute path to the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))

    possible_model_paths = [
        os.path.join(
            project_root,
            "aera",
            "autonomous",
            "simulation",
            "mujoco",
            "ar4_mk3",
            "scene.xml",
        ),
        "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
        "../aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
        "../../aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
    ]

    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = os.path.abspath(path)
            break

    if model_path is None:
        print(
            "Error: Could not find AR4 MK3 model file. Please ensure the MuJoCo model exists."
        )
        print("Tried the following paths:")
        for path in possible_model_paths:
            print(f"  - {path}")
        return

    print(f"Using model file: {model_path}")

    env = Ar4Mk3PickAndPlaceEnv(
        model_path=model_path,
        render_mode="human",
        reward_type="sparse",
        translation=T,
        quaterion=Q,
        distance_multiplier=1.2,
        z_offset=0.3,
    )

    # Reset the environment
    observation, info = env.reset()
    print("Environment reset complete")

    # Create the robot interface
    robot = Ar4Mk3RobotInterface(env)
    print("Robot interface created")

    try:
        env.render()
        # Step 1: Go to home position
        print("\n1. Moving to home position...")
        if robot.go_home():
            print("✓ Successfully moved to home position")
        else:
            print("✗ Failed to move to home position")
        # time.sleep(0.5)

        # Step 2: Get current end-effector pose
        print("\n2. Getting current end-effector pose...")
        current_pose = robot.get_end_effector_pose()
        if current_pose:
            print(
                f"✓ Current position: ({current_pose.position.x:.3f}, "
                f"{current_pose.position.y:.3f}, {current_pose.position.z:.3f})"
            )
        else:
            print("✗ Failed to get current pose")
            return

        # Step 3: Move to a target position
        print("\n3. Moving to target position...")
        # target_pose = create_pose(
        #     x=current_pose.position.x - 0.2,
        #     y=current_pose.position.y,
        #     z=current_pose.position.z - 0.1,
        #     qx=current_pose.orientation.x,
        #     qy=current_pose.orientation.y,
        #     qz=current_pose.orientation.z,
        #     qw=current_pose.orientation.w,
        # )
        target_pose = create_pose(
            x=current_pose.position.x - 0.2,
            y=current_pose.position.y,
            z=current_pose.position.z - 0.1,
        )

        if robot.move_to(target_pose):
            print("✓ Successfully moved to target position")
        else:
            print("✗ Failed to move to target position")
        # time.sleep(1)
        # print("Moving home")
        # robot.go_home()
        time.sleep(1)
        # target_pose = create_pose(
        #     x=current_pose.position.x - 0.3,
        #     y=current_pose.position.y + 0.2,
        #     z=current_pose.position.z - 0.1,
        #     qx=current_pose.orientation.x,
        #     qy=current_pose.orientation.y,
        #     qz=current_pose.orientation.z,
        #     qw=current_pose.orientation.w,
        # )
        print("Graping")
        robot.grasp_at(target_pose, -0.006)
        time.sleep(1)
        print("releasing")
        robot.release_at(target_pose)
        time.sleep(1)
        print("Grasping")
        robot.grasp_at(target_pose, 0)
        time.sleep(1)
        # target_pose = create_pose(
        #     x=current_pose.position.x + 0.1,
        #     y=current_pose.position.y - 0.1,
        #     z=current_pose.position.z,
        #     qx=current_pose.orientation.x,
        #     qy=current_pose.orientation.y,
        #     qz=current_pose.orientation.z,
        #     qw=current_pose.orientation.w,
        # )
        robot.go_home()
        time.sleep(1)
        #
        # # Step 4: Test gripper control
        # print("\n4. Testing gripper control...")
        # print("   Closing gripper...")
        # grasp_pose = create_pose(
        #     x=target_pose.position.x,
        #     y=target_pose.position.y,
        #     z=target_pose.position.z - 0.05,
        # )
        #
        # if robot.grasp_at(grasp_pose, gripper_pos=0.8):
        #     print("✓ Successfully performed grasp motion")
        # else:
        #     print("✗ Failed to perform grasp motion")
        #
        # time.sleep(3)
        #
        # # Step 5: Release gripper
        # print("\n5. Opening gripper...")
        # if robot.release_gripper():
        #     print("✓ Successfully opened gripper")
        # else:
        #     print("✗ Failed to open gripper")
        # time.sleep(3)
        #
        # # Step 6: Test camera functionality
        # print("\n6. Testing camera functionality...")
        # # rgb_image = robot.get_latest_rgb_image()
        # # depth_image = robot.get_latest_depth_image()
        #
        # # if rgb_image is not None:
        # #     print(f"✓ RGB image captured: shape {rgb_image.shape}")
        # # else:
        # #     print("✗ Failed to capture RGB image")
        # #
        # # if depth_image is not None:
        # #     print(f"✓ Depth image captured: shape {depth_image.shape}")
        # # else:
        # #     print("✗ Failed to capture depth image")
        #
        # # Step 7: Get camera intrinsics and transform
        # print("\n7. Testing camera parameters...")
        # intrinsics = robot.get_camera_intrinsics()
        # transform = robot.get_cam_to_base_transform()
        #
        # if intrinsics:
        #     print(f"✓ Camera intrinsics: {intrinsics.width}x{intrinsics.height}")
        # else:
        #     print("✗ Failed to get camera intrinsics")
        #
        # if transform is not None:
        #     print(f"✓ Camera transform: shape {transform.shape}")
        # else:
        #     print("✗ Failed to get camera transform")
        #
        # # Step 8: Return to home
        # print("\n8. Returning to home position...")
        # if robot.go_home():
        #     print("✓ Successfully returned to home position")
        # else:
        #     print("✗ Failed to return to home position")
        # time.sleep(3)
        #
        # print("\n🎉 Demonstration completed successfully!")

    except KeyboardInterrupt:
        print("\n⚠️  Demonstration interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    finally:
        print("Closing environment...")
        env.close()


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    main()
