#!/usr/bin/env python3
"""
Headless demonstration script for controlling the AR4 MK3 arm simulation.
"""

import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion

from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_robot_interface import (
    Ar4Mk3RobotInterface,
)


def create_pose(x, y, z, qx=0.0, qy=0.0, qz=0.0, qw=1.0):
    """Helper function to create a Pose message."""
    pose = Pose()
    pose.position = Point(x=x, y=y, z=z)
    pose.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)
    return pose


def run_movement_sequence(robot, env):
    """Run a sequence of movements to test the robot interface."""

    # Get initial pose
    initial_pose = robot.get_end_effector_pose()
    if not initial_pose:
        print("Failed to get initial pose")
        return False

    print(
        f"Initial position: ({initial_pose.position.x:.3f}, "
        f"{initial_pose.position.y:.3f}, {initial_pose.position.z:.3f})"
    )

    # Define a sequence of target positions relative to initial pose
    movements = [
        (0.05, 0.0, 0.0),  # Move right
        (0.0, 0.05, 0.0),  # Move forward
        (0.0, 0.0, 0.05),  # Move up
        (-0.05, 0.0, 0.0),  # Move left
        (0.0, -0.05, 0.0),  # Move back
        (0.0, 0.0, -0.05),  # Move down
    ]

    for i, (dx, dy, dz) in enumerate(movements):
        print(f"Movement {i + 1}: ({dx:+.2f}, {dy:+.2f}, {dz:+.2f})")

        target_pose = create_pose(
            x=initial_pose.position.x + dx,
            y=initial_pose.position.y + dy,
            z=initial_pose.position.z + dz,
        )

        if robot.move_to(target_pose):
            print(f"✓ Movement {i + 1} completed")
        else:
            print(f"✗ Movement {i + 1} failed")
            return False

        # Step the simulation a few times
        for _ in range(10):
            env.step(np.array([0.0, 0.0, 0.0, 0.0]))

    return True


def main():
    print("Starting headless AR4 MK3 arm control demonstration...")

    # Create environment without rendering
    # Try to find the model file in common locations
    import os
    
    # Get the absolute path to the model file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    
    possible_model_paths = [
        os.path.join(project_root, "aera", "autonomous", "simulation", "mujoco", "ar4_mk3", "scene.xml"),
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
        print("Error: Could not find AR4 MK3 model file. Please ensure the MuJoCo model exists.")
        print("Tried the following paths:")
        for path in possible_model_paths:
            print(f"  - {path}")
        return
    
    print(f"Using model file: {model_path}")
    
    try:
        env = Ar4Mk3PickAndPlaceEnv(
            model_path=model_path,
            use_eef_control=True,
            render_mode=None,  # No rendering for speed
            reward_type="sparse",
        )
    except Exception as e:
        print(f"Error creating environment: {e}")
        return

    # Reset environment
    observation, info = env.reset()
    robot = Ar4Mk3RobotInterface(env)

    try:
        # Test basic functionality
        print("\n=== Testing Basic Functionality ===")

        # Home position
        print("Going home...")
        if robot.go_home():
            print("✓ Home position reached")
        else:
            print("✗ Failed to reach home position")
            return

        # Movement sequence
        print("\n=== Testing Movement Sequence ===")
        if run_movement_sequence(robot, env):
            print("✓ Movement sequence completed")
        else:
            print("✗ Movement sequence failed")

        # Gripper tests
        print("\n=== Testing Gripper Control ===")
        current_pose = robot.get_end_effector_pose()
        if current_pose:
            # Test grasp
            if robot.grasp_at(current_pose, gripper_pos=0.5):
                print("✓ Grasp motion completed")
            else:
                print("✗ Grasp motion failed")

            # Test release
            if robot.release_gripper():
                print("✓ Gripper released")
            else:
                print("✗ Failed to release gripper")

        # Camera tests
        print("\n=== Testing Camera Functions ===")
        rgb_image = robot.get_latest_rgb_image()
        depth_image = robot.get_latest_depth_image()
        intrinsics = robot.get_camera_intrinsics()
        transform = robot.get_cam_to_base_transform()

        print(f"RGB image: {'✓' if rgb_image is not None else '✗'}")
        print(f"Depth image: {'✓' if depth_image is not None else '✗'}")
        print(f"Camera intrinsics: {'✓' if intrinsics is not None else '✗'}")
        print(f"Camera transform: {'✓' if transform is not None else '✗'}")

        print("\n🎉 All tests completed!")

    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
