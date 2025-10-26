#!/usr/bin/env python3
"""
Script for collecting pick-and-place trajectories with domain randomization.
"""

import argparse
import logging
import os
import uuid
from typing import Optional

import numpy as np
from geometry_msgs.msg import Pose, Point, Quaternion
from scipy.spatial.transform import Rotation

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.control.ar4_mk3_interface_config import (
    Ar4Mk3InterfaceConfig,
)
from aera_semi_autonomous.control.ar4_mk3_robot_interface import Ar4Mk3RobotInterface
from aera_semi_autonomous.data.domain_rand_config_generator import (
    generate_random_domain_rand_config,
)
from aera_semi_autonomous.data.trajectory_data_collector import TrajectoryDataCollector

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
        object_qpos = env._utils.get_joint_qpos(env.model, env.data, "object0:joint")
        object_pos = object_qpos[:3]
        object_body_id = env.model.body("object0").id
        geom_id = -1
        for i in range(env.model.ngeom):
            if env.model.geom_bodyid[i] == object_body_id:
                geom_id = i
                break

        additional_yaw = 0.0
        if geom_id != -1:
            geom_size = env.model.geom_size[geom_id]
            if geom_size[1] < geom_size[0]:
                additional_yaw = 90.0

        object_quat_wxyz = object_qpos[3:]
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

        pose = Pose()
        pose.position = Point(
            x=float(object_pos[0]),
            y=float(object_pos[1]),
            z=2 * float(object_pos[2]),
        )

        top_down_rot = Rotation.from_quat([0, 1, 0, 0])
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


def run_pick_and_place_and_collect(
    robot: Ar4Mk3RobotInterface,
    data_collector: TrajectoryDataCollector,
    object_color: str,
    target_color: str,
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
    if not robot.release_at(target_pose):
        logger.error("Failed to place object at target location")
        return False

    # Go home
    data_collector.record_current_prompt("go home")
    robot.go_home()

    # Stop episode and save data
    data_collector.stop_episode()
    data_collector.log_trajectory_summary()

    # Check success condition (e.g., object is close to target)
    object_final_pos = env._utils.get_site_xpos(env.model, env.data, "object0")
    distance_to_target = np.linalg.norm(object_final_pos - target_pos)

    success = distance_to_target < env.distance_threshold
    if success:
        logger.info(
            f"Trajectory collection successful! Object is at target. Distance: {distance_to_target:.4f}"
        )
    else:
        logger.warning(
            f"Trajectory collection finished, but object not at target. Distance: {distance_to_target:.4f}"
        )

    return success


def main():
    """Main function to collect trajectories."""
    parser = argparse.ArgumentParser(description="AR4 MK3 Trajectory Collection")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--render", action="store_true", help="Enable rendering")
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=10,
        help="Number of trajectories to collect",
    )
    parser.add_argument(
        "--save-dir", type=str, default="rl_training_data", help="Directory to save data"
    )
    args = parser.parse_args()

    logger = setup_logging(args.debug)
    logger.info(
        f"Starting trajectory collection for {args.num_trajectories} episodes."
    )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    model_path = os.path.join(
        project_root, "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml"
    )

    if not os.path.exists(model_path):
        logger.error(f"Could not find model path: {model_path}")
        return

    successful_collections = 0
    for i in range(args.num_trajectories):
        logger.info(f"--- Starting trajectory {i+1}/{args.num_trajectories} ---")
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
                render_mode="human" if args.render else None,
                config=env_config,
            )
            obs, info = env.reset()

            interface_config = Ar4Mk3InterfaceConfig()
            robot = Ar4Mk3RobotInterface(env, config=interface_config)

            data_collector = TrajectoryDataCollector(
                logger=logger,
                arm_joint_names=[f"joint_{j}" for j in range(1, 7)],
                gripper_joint_names=["gripper_jaw1_joint", "gripper_jaw2_joint"],
                save_directory=args.save_dir,
            )
            robot.set_data_collector(data_collector)

            success = run_pick_and_place_and_collect(
                robot, data_collector, object_color, target_color
            )

            if success:
                successful_collections += 1
            else:
                # Optional: delete unsuccessful trajectory data
                episode_dir = os.path.join(args.save_dir, data_collector.episode_id)
                if os.path.exists(episode_dir):
                    logger.warning(
                        f"Deleting data for unsuccessful trajectory: {data_collector.episode_id}"
                    )
                    # import shutil; shutil.rmtree(episode_dir) # Be careful with this

        except Exception as e:
            logger.error(
                f"Trajectory {i+1} failed with exception: {e}", exc_info=True
            )
        finally:
            if env:
                env.close()

    logger.info("--- Trajectory collection finished ---")
    logger.info(
        f"Successfully collected {successful_collections}/{args.num_trajectories} trajectories."
    )


if __name__ == "__main__":
    main()
