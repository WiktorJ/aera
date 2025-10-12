#!/usr/bin/env python3
"""
Demo script for AR4 MK3 pick and place using Ar4Mk3RobotInterface.

This script demonstrates how to use the Ar4Mk3RobotInterface to:
1. Initialize the robot environment
2. Locate object0 in the simulation
3. Pick up the object
4. Place it at the target0 location
"""

import argparse
import logging
import time
from typing import Optional
import os

import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation

from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = (
    -0.36336720179946663,
    -0.8203835174702869,
    0.22865474664402222,
    0.37769321910336584,
)


def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def get_object_pose(env) -> Optional[Pose]:
    """Get the current pose of object0 from the environment."""
    try:
        # Get object position from MuJoCo (this is the center of the object)
        object_pos = env._utils.get_site_xpos(env.model, env.data, "object0")

        # Get object orientation to align gripper
        object_body_id = env.model.body("object0").id

        # Find the geom associated with the object body to check its dimensions
        geom_id = -1
        for i in range(env.model.ngeom):
            if env.model.geom_bodyid[i] == object_body_id:
                geom_id = i
                break

        additional_yaw = 0.0
        if geom_id != -1:
            geom_size = env.model.geom_size[geom_id]
            # For a box, size is [dx, dy, dz] (half-lengths).
            # If the object is longer along its y-axis (dy > dx), we want to
            # align the gripper with the object's y-axis. This requires an
            # additional 90-degree rotation.
            if geom_size[1] < geom_size[0]:
                additional_yaw = 90.0

        # MuJoCo quat is w,x,y,z. Scipy is x,y,z,w
        object_quat_wxyz = env.data.xquat[object_body_id]
        object_quat_xyzw = np.array(
            [
                object_quat_wxyz[1],
                object_quat_wxyz[2],
                object_quat_wxyz[3],
                object_quat_wxyz[0],
            ]
        )
        object_rotation = Rotation.from_quat(object_quat_xyzw)
        object_yaw_deg = object_rotation.as_euler("xyz", degrees=True)[2]

        # Create pose for the top surface of the object
        pose = Pose()
        pose.position = Point(
            x=float(object_pos[0]),
            y=float(object_pos[1]),
            z=float(object_pos[2]),
        )

        # Combine top-down orientation with object's yaw
        top_down_rot = Rotation.from_quat([0, 1, 0, 0])  # x, y, z, w
        z_rot = Rotation.from_euler("z", object_yaw_deg + additional_yaw, degrees=True)
        grasp_rot = z_rot * top_down_rot
        grasp_quat_xyzw = grasp_rot.as_quat()

        pose.orientation = Quaternion(
            x=grasp_quat_xyzw[0],
            y=grasp_quat_xyzw[1],
            z=grasp_quat_xyzw[2],
            w=grasp_quat_xyzw[3],
        )

        return pose
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get object pose: {e}")
        return None


def main():
    """Main function to demonstrate pick and place operation."""
    parser = argparse.ArgumentParser(description="AR4 MK3 Pick and Place Demo")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--steps", type=int, default=1000, help="Max simulation steps")
    args = parser.parse_args()

    logger = setup_logging(args.debug)
    logger.info("Starting AR4 MK3 Pick and Place Demo")
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

    try:
        # Initialize the environment
        logger.info("Initializing AR4 MK3 environment...")
        env = Ar4Mk3PickAndPlaceEnv(
            model_path=model_path,
            render_mode="human",
            reward_type="sparse",
            use_eef_control=False,  # Use joint control for better precision
            translation=T,
            quaterion=Q,
            distance_multiplier=1.2,
            z_offset=0.3,
        )

        # Reset environment to get initial state
        obs, info = env.reset()
        logger.info("Environment initialized and reset")

        # Initialize robot interface
        logger.info("Initializing robot interface...")
        robot = Ar4Mk3RobotInterface(env)
        logger.info("Robot interface initialized")

        # Step 1: Go to home position
        logger.info("Moving robot to home position...")
        if not robot.go_home():
            logger.error("Failed to move robot to home position")
            return False
        logger.info("Robot moved to home position")

        # Step 2: Get object position
        logger.info("Locating object...")
        object_pose = get_object_pose(env)
        if object_pose is None:
            logger.error("Failed to locate object")
            return False

        logger.info(
            f"Object found at position: ({object_pose.position.x:.3f}, "
            f"{object_pose.position.y:.3f}, {object_pose.position.z:.3f})"
        )

        # Step 3: Pick up the object
        logger.info("Attempting to pick up object...")
        gripper_pos = -0.007  # Halfway between open (-0.014) and closed (0)

        if not robot.grasp_at(object_pose, gripper_pos):
            logger.error("Failed to pick up object")
            return False
        logger.info("Successfully picked up object")

        # Step 4: Get target location from env (target0)
        target_pos = env.goal
        target_pose = Pose()
        target_pose.position = Point(
            x=float(target_pos[0]), y=float(target_pos[1]), z=float(target_pos[2])
        )
        # Top-down orientation for placing
        target_pose.orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        logger.info(
            f"Target position from 'target0': ({target_pose.position.x:.3f}, "
            f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f})"
        )

        # Step 5: Move to target and release
        logger.info("Moving to target location and releasing object...")
        if not robot.release_at(target_pose):
            logger.error("Failed to place object at target location")
            return False
        logger.info("Successfully placed object at target location")

        # Let the simulation run for a bit to see the result
        logger.info("Demo completed! Letting simulation run to observe result...")
        time.sleep(2)

        logger.info("Pick and place demo completed successfully!")
        return True

    except Exception as e:
        logger.error(f"Demo failed with exception: {e}", exc_info=True)
        return False

    finally:
        try:
            env.close()
        except:
            pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
