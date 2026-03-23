"""Trajectory perturbation utilities for generating movement diversity.

Provides a perturbation mode that can be applied before pick/place actions
to create varied arm trajectories while preserving the exact grasp/release targets.

The robot always grasps at the exact object pose and places at the exact target
pose. What varies is the path it takes to get there: the robot visits one or more
random waypoints on a disk above the target before descending.

Usage:
    config = PerturbationConfig(mode="offset_approach", num_approach_waypoints=2)
    waypoints = generate_waypoints(target_pose, config)
    for wp in waypoints:
        robot.move_to(wp)
    robot.grasp_at(target_pose, gripper_pos=0.0)
"""

import copy
from dataclasses import dataclass

import numpy as np
from geometry_msgs.msg import Pose


@dataclass
class PerturbationConfig:
    """Configuration for trajectory perturbation.

    Attributes:
        mode: Perturbation mode. One of "none" or "offset_approach".
        perturb_pick: Whether to perturb the pick (grasp) approach.
        perturb_place: Whether to perturb the place (release) approach.
        num_approach_waypoints: Number of offset waypoints to generate (default 1).
        approach_min_offset: Minimum XY distance from target for offset waypoint (meters).
        approach_max_offset: Maximum XY distance from target for offset waypoint (meters).
        approach_height: Base height above target for the offset waypoint (meters).
        approach_height_noise: Random additional height variation (meters).
    """

    mode: str = "none"

    perturb_pick: bool = True
    perturb_place: bool = True

    num_approach_waypoints: int = 1
    approach_min_offset: float = 0.01
    approach_max_offset: float = 0.04
    approach_height: float = 0.06
    approach_height_noise: float = 0.02


def generate_offset_approach(target_pose: Pose, config: PerturbationConfig) -> list:
    """Generate waypoints on a disk above the target.

    Each waypoint is placed at a random angle and radius from the target,
    at a height above it. This causes the robot to approach the target
    from a different direction each time. When multiple waypoints are
    generated, the robot visits several positions above the target,
    creating more varied trajectories.

    Args:
        target_pose: The final target pose the robot will move to.
        config: Perturbation configuration.

    Returns:
        A list of Pose waypoints above the target.
    """
    waypoints = []
    for _ in range(config.num_approach_waypoints):
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(config.approach_min_offset, config.approach_max_offset)

        waypoint = copy.deepcopy(target_pose)
        waypoint.position.x += radius * np.cos(angle)
        waypoint.position.y += radius * np.sin(angle)
        waypoint.position.z += config.approach_height + np.random.uniform(
            0, config.approach_height_noise
        )
        waypoints.append(waypoint)
    return waypoints


def generate_waypoints(target_pose: Pose, config: PerturbationConfig) -> list:
    """Generate perturbation waypoints based on the configured mode.

    This is the main entry point. Call this before a pick or place action
    and execute each returned waypoint via robot.move_to().

    Args:
        target_pose: The final target pose for the upcoming action.
        config: Perturbation configuration.

    Returns:
        A list of Pose waypoints to visit before the target action.
        Returns an empty list when mode is "none".
    """
    if config.mode == "none":
        return []

    if config.mode == "offset_approach":
        return generate_offset_approach(target_pose, config)

    return []
