#!/usr/bin/env python3
"""
Demo script for AR4 MK3 pick and place using Ar4Mk3RobotInterface.

This script demonstrates how to use the Ar4Mk3RobotInterface to:
1. Initialize the robot environment
2. Locate object0 in the simulation
3. Pick up the object
4. Place it at the target0 location
"""

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import tyro
from geometry_msgs.msg import Pose, Point, Quaternion

from aera.autonomous.envs.ar4_mk3_config import (
    Ar4Mk3EnvConfig,
    DomainRandConfig,
    DynamicsConfig,
    LightConfig,
    MaterialConfig,
)
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_interface_config import (
    Ar4Mk3InterfaceConfig,
)
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface
from aera_semi_autonomous.data.domain_rand_config_generator import (
    generate_random_domain_rand_config,
)
from aera_semi_autonomous.data.pick_and_place_helpers import get_object_pose
from aera_semi_autonomous.data.trajectory_perturbation import (
    PerturbationConfig,
    generate_waypoints,
    perturb_ik_config,
)

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = (
    -0.36336720179946663,
    -0.8203835174702869,
    0.22865474664402222,
    0.37769321910336584,
)


@dataclass
class DemoConfig:
    debug: bool = False
    render: bool = False
    steps: int = 1000
    domain_rand: bool = False
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    interface: Ar4Mk3InterfaceConfig = field(default_factory=Ar4Mk3InterfaceConfig)


def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def main():
    """Main function to demonstrate pick and place operation."""
    cfg = tyro.cli(DemoConfig)

    logger = setup_logging(cfg.debug)
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
        logger.error(
            "Could not find AR4 MK3 model file. Please ensure the MuJoCo model exists."
        )
        logger.error("Tried the following paths:")
        for path in possible_model_paths:
            logger.error(f"  - {path}")
        return

    try:
        # Initialize the environment
        logger.info("Initializing AR4 MK3 environment...")

        domain_rand_config = None
        if cfg.domain_rand:
            logger.info("Enabling domain randomization")
            domain_rand_config, object_color, target_color = (
                generate_random_domain_rand_config()
            )
            logger.info(f"Object color: {object_color}")
            logger.info(f"Target color: {target_color}")

        env_config = Ar4Mk3EnvConfig(
            model_path=model_path,
            reward_type="sparse",
            use_eef_control=False,  # Use joint control for better precision
            translation=T,
            quaterion=Q,
            distance_multiplier=1.2,
            z_offset=0.3,
            domain_rand=domain_rand_config,
        )
        env = Ar4Mk3PickAndPlaceEnv(
            render_mode="human" if cfg.render else "rgb_array",
            config=env_config,
        )

        # Reset environment to get initial state
        obs, info = env.reset()
        logger.info("Environment initialized and reset")

        # Initialize robot interface
        logger.info("Initializing robot interface...")
        interface_config = cfg.interface
        if cfg.perturbation.mode == "ik_noise":
            interface_config = dataclasses.replace(
                cfg.interface,
                ik=perturb_ik_config(cfg.interface.ik, cfg.perturbation.ik_noise),
            )
        robot = Ar4Mk3RobotInterface(env, config=interface_config)
        logger.info("Robot interface initialized")

        # Step 1: Go to home position
        logger.info("Moving robot to home position...")
        if not robot.go_home():
            logger.error("Failed to move robot to home position")
            return False
        logger.info("Robot moved to home position")

        # Step 2: Get object position
        logger.info("Locating object...")
        object_pose = get_object_pose(env, logger)
        if object_pose is None:
            logger.error("Failed to locate object")
            return False

        logger.info(
            f"Object found at position: ({object_pose.position.x:.3f}, "
            f"{object_pose.position.y:.3f}, {object_pose.position.z:.3f})"
        )

        # Step 3: Pick up the object
        logger.info("Attempting to pick up object...")
        gripper_pos = 0.0  # Fully closed

        if cfg.perturbation.perturb_pick:
            for wp in generate_waypoints(object_pose, cfg.perturbation):
                robot.move_to(wp)

        if not robot.grasp_at(object_pose, gripper_pos):
            logger.error("Failed to pick up object")
            return False
        logger.info("Successfully picked up object")

        # Step 4: Get target location from env (target0)
        target_pos = env.goal
        target_pose = Pose()
        target_pose.position = Point(
            x=float(target_pos[0]),
            y=float(target_pos[1]),
            z=float(target_pos[2] + object_pose.position.z),
        )
        # Top-down orientation for placing
        target_pose.orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)
        logger.info(
            f"Target position from 'target0': ({target_pose.position.x:.3f}, "
            f"{target_pose.position.y:.3f}, {target_pose.position.z:.3f})"
        )

        # Step 5: Move to target and release
        logger.info("Moving to target location and releasing object...")

        if cfg.perturbation.perturb_place:
            for wp in generate_waypoints(target_pose, cfg.perturbation):
                robot.move_to(wp)

        if not robot.release_at(target_pose):
            logger.error("Failed to place object at target location")
            return False
        logger.info("Successfully placed object at target location")

        robot.go_home()
        # Let the simulation run for a bit to see the result

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
