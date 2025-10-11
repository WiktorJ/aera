#!/usr/bin/env python3
"""
Demo script for AR4 MK3 pick and place using Ar4Mk3RobotInterface.

This script demonstrates how to use the Ar4Mk3RobotInterface to:
1. Initialize the robot environment
2. Locate object0 in the simulation
3. Pick up the object
4. Place it at a random location
"""

import argparse
import logging
import time
from typing import Optional

import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion

from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface


def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)


def get_object_pose(env) -> Optional[Pose]:
    """Get the current pose of object0 from the environment."""
    try:
        # Get object position from MuJoCo
        object_pos = env._utils.get_site_xpos(env.model, env.data, "object0")
        
        # For simplicity, assume object has identity orientation
        # In a real scenario, you might want to get the actual orientation
        pose = Pose()
        pose.position = Point(x=float(object_pos[0]), y=float(object_pos[1]), z=float(object_pos[2]))
        pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        
        return pose
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get object pose: {e}")
        return None


def generate_random_target_pose(env, min_distance: float = 0.1, max_distance: float = 0.3) -> Pose:
    """Generate a random target pose for placing the object."""
    # Get the initial gripper position as reference
    initial_pos = env.initial_gripper_xpos
    
    # Generate random offset within specified distance range
    angle = np.random.uniform(0, 2 * np.pi)
    distance = np.random.uniform(min_distance, max_distance)
    
    target_x = initial_pos[0] + distance * np.cos(angle)
    target_y = initial_pos[1] + distance * np.sin(angle)
    target_z = 0.025  # Object height (half of object size)
    
    pose = Pose()
    pose.position = Point(x=float(target_x), y=float(target_y), z=float(target_z))
    pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    
    return pose


def main():
    """Main function to demonstrate pick and place operation."""
    parser = argparse.ArgumentParser(description="AR4 MK3 Pick and Place Demo")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument("--steps", type=int, default=1000, help="Max simulation steps")
    args = parser.parse_args()
    
    logger = setup_logging(args.debug)
    logger.info("Starting AR4 MK3 Pick and Place Demo")
    
    try:
        # Initialize the environment
        logger.info("Initializing AR4 MK3 environment...")
        env = Ar4Mk3PickAndPlaceEnv(
            render_mode="human" if args.render else None,
            has_object=True,
            target_in_the_air=False,
            target_range=0.15,
            reward_type="sparse",
            use_eef_control=False,  # Use joint control for better precision
        )
        
        # Reset environment to get initial state
        obs, info = env.reset()
        logger.info("Environment initialized and reset")
        
        # Initialize robot interface
        logger.info("Initializing robot interface...")
        robot = Ar4Mk3RobotInterface(env)
        logger.info("Robot interface initialized")
        
        # Let the simulation settle
        for _ in range(10):
            env.step(np.zeros(env.action_space.shape))
            if args.render:
                env.render()
                time.sleep(0.1)
        
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
        
        logger.info(f"Object found at position: ({object_pose.position.x:.3f}, "
                   f"{object_pose.position.y:.3f}, {object_pose.position.z:.3f})")
        
        # Step 3: Pick up the object
        logger.info("Attempting to pick up object...")
        gripper_pos = -0.007  # Halfway between open (-0.014) and closed (0)
        
        if not robot.grasp_at(object_pose, gripper_pos):
            logger.error("Failed to pick up object")
            return False
        logger.info("Successfully picked up object")
        
        # Let the simulation settle after grasping
        for _ in range(20):
            env.step(np.zeros(env.action_space.shape))
            if args.render:
                env.render()
                time.sleep(0.05)
        
        # Step 4: Generate random target location
        target_pose = generate_random_target_pose(env)
        logger.info(f"Generated target position: ({target_pose.position.x:.3f}, "
                   f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f})")
        
        # Step 5: Move to target and release
        logger.info("Moving to target location and releasing object...")
        if not robot.release_at(target_pose):
            logger.error("Failed to place object at target location")
            return False
        logger.info("Successfully placed object at target location")
        
        # Let the simulation run for a bit to see the result
        logger.info("Demo completed! Letting simulation run to observe result...")
        for i in range(100):
            env.step(np.zeros(env.action_space.shape))
            if args.render:
                env.render()
                time.sleep(0.05)
        
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
