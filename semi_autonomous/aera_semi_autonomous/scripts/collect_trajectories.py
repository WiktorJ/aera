#!/usr/bin/env python3
"""
Script for collecting pick-and-place trajectories with domain randomization.
"""

import dataclasses
import logging
import os
from dataclasses import dataclass, field

import numpy as np
import tyro
from geometry_msgs.msg import Pose, Point, Quaternion

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_interface_config import (
    Ar4Mk3InterfaceConfig,
)
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface
from aera_semi_autonomous.data.domain_rand_config_generator import (
    generate_random_domain_rand_config,
)
from aera_semi_autonomous.data.pick_and_place_helpers import get_object_pose
from aera_semi_autonomous.data.trajectory_data_collector import TrajectoryDataCollector
from aera_semi_autonomous.data.trajectory_perturbation import (
    PerturbationConfig,
    generate_waypoints,
    perturb_ik_config,
)

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = np.array(
    [
        -0.36336720179946663,
        -0.8203835174702869,
        0.22865474664402222,
        0.37769321910336584,
    ]
)


@dataclass
class CollectConfig:
    debug: bool = False
    render: bool = False
    num_trajectories: int = 1
    save_dir: str = "rl_training_data"
    seed: int = -1
    perturbation: PerturbationConfig = field(default_factory=PerturbationConfig)
    interface: Ar4Mk3InterfaceConfig = field(default_factory=Ar4Mk3InterfaceConfig)


def setup_logging(debug: bool = False) -> logging.Logger:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def run_pick_and_place_and_collect(
    robot: Ar4Mk3RobotInterface,
    data_collector: TrajectoryDataCollector,
    object_color: str,
    target_color: str,
    perturbation_config: PerturbationConfig = PerturbationConfig(),
) -> bool:
    """Run a single pick and place task and collect data."""
    logger = robot.get_logger()
    env = robot.env

    # Start episode
    input_message = (
        f"pick the {object_color} block and place it on the {target_color} target"
    )
    data_collector.start_episode(input_message)

    # Go home
    data_collector.record_current_prompt("go home")
    if not robot.go_home():
        logger.error("Failed to move robot to home position")
        return False

    # Get object pose
    object_pose = get_object_pose(env)
    if object_pose is None:
        logger.error("Failed to locate object")
        return False

    # Pick up the object
    data_collector.record_current_prompt(f"pick {object_color} block")
    if perturbation_config.perturb_pick:
        for wp in generate_waypoints(object_pose, perturbation_config):
            robot.move_to(wp)
    if not robot.grasp_at(object_pose, gripper_pos=0.0):
        logger.error("Failed to pick up object")
        return False

    # Get target location
    target_pos = env.goal
    target_pose = Pose()
    target_pose.position = Point(
        x=float(target_pos[0]),
        y=float(target_pos[1]),
        z=float(target_pos[2] + object_pose.position.z),
    )
    target_pose.orientation = Quaternion(x=0.0, y=1.0, z=0.0, w=0.0)

    # Move to target and release
    data_collector.record_current_prompt(f"place on {target_color} target")
    if perturbation_config.perturb_place:
        for wp in generate_waypoints(target_pose, perturbation_config):
            robot.move_to(wp)
    if not robot.release_at(target_pose):
        logger.error("Failed to place object at target location")
        return False

    # Go home
    data_collector.record_current_prompt("go home")
    robot.go_home()

    # Check success condition (e.g., object is close to target)
    object_final_pos = env._utils.get_site_xpos(env.model, env.data, "object0")
    distance_to_target = np.linalg.norm(object_final_pos - target_pos)

    success = bool(distance_to_target < env.distance_threshold)
    data_collector.stop_episode(success)
    data_collector.log_trajectory_summary()

    if not success:
        logger.warning(
            f"Trajectory collection finished, but object not at target. Distance: {distance_to_target:.4f}"
        )
        return False
    logger.info(
        f"Trajectory collection successful! Object is at target. Distance: {distance_to_target:.4f}"
    )

    return True


def main():
    """Main function to collect trajectories."""
    cfg = tyro.cli(CollectConfig)

    if cfg.seed != -1:
        np.random.seed(cfg.seed)

    logger = setup_logging(cfg.debug)
    logger.info(f"Starting trajectory collection for {cfg.num_trajectories} episodes.")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    model_path = os.path.join(
        project_root, "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml"
    )

    if not os.path.exists(model_path):
        logger.error(f"Could not find model path: {model_path}")
        return

    successful_collections = 0
    for i in range(cfg.num_trajectories):
        logger.info(f"--- Starting trajectory {i + 1}/{cfg.num_trajectories} ---")
        env = None
        try:
            (
                domain_rand_config,
                object_color,
                target_color,
            ) = generate_random_domain_rand_config()

            env_config = Ar4Mk3EnvConfig(
                model_path=model_path,
                reward_type="sparse",
                use_eef_control=False,
                translation=T,
                quaterion=Q,
                distance_multiplier=1.2,
                z_offset=0.3,
                domain_rand=domain_rand_config,
            )
            env = Ar4Mk3PickAndPlaceEnv(
                render_mode="human" if cfg.render else None,
                config=env_config,
            )
            _, _ = env.reset()

            interface_config = cfg.interface
            if cfg.perturbation.mode == "ik_noise":
                interface_config = dataclasses.replace(
                    cfg.interface,
                    ik=perturb_ik_config(cfg.interface.ik, cfg.perturbation.ik_noise),
                )
            robot = Ar4Mk3RobotInterface(env, config=interface_config)

            data_collector = TrajectoryDataCollector(
                logger=logger,
                arm_joint_names=[f"joint_{j}" for j in range(1, 7)],
                gripper_joint_names=["gripper_jaw1_joint", "gripper_jaw2_joint"],
                save_directory=cfg.save_dir,
            )
            robot.set_data_collector(data_collector)

            success = run_pick_and_place_and_collect(
                robot, data_collector, object_color, target_color,
                perturbation_config=cfg.perturbation,
            )
            successful_collections += int(success)

        except Exception as e:
            logger.error(
                f"Trajectory {i + 1} failed with exception: {e}", exc_info=True
            )
        finally:
            if env:
                env.close()

    logger.info("--- Trajectory collection finished ---")
    logger.info(
        f"Successfully collected {successful_collections}/{cfg.num_trajectories} trajectories."
    )


if __name__ == "__main__":
    main()
