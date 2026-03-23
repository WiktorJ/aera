"""Trajectory perturbation utilities for generating movement diversity.

Provides two perturbation modes that can be applied before pick/place actions
to create varied arm trajectories while preserving the exact grasp/release targets.

Modes:
    offset_approach: Single waypoint on a disk above the target, creating a
                     different approach angle each run.
    noisy_path:      Multiple tiny waypoints along the straight-line path from
                     current pose to target, each slightly jittered.
    both:            Noisy path leading to an offset approach point.

Usage:
    config = PerturbationConfig(mode="offset_approach")
    waypoints = generate_waypoints(current_pose, target_pose, config)
    for wp in waypoints:
        robot.move_to(wp)
    robot.grasp_at(target_pose, gripper_pos=0.0)
"""

import copy
from dataclasses import dataclass, field

import numpy as np
from geometry_msgs.msg import Pose


@dataclass
class PerturbationConfig:
    """Configuration for trajectory perturbation.

    Attributes:
        mode: Perturbation mode. One of "none", "offset_approach", "noisy_path", "both".
        perturb_pick: Whether to perturb the pick (grasp) approach.
        perturb_place: Whether to perturb the place (release) approach.
        approach_min_offset: Minimum XY distance from target for offset waypoint (meters).
        approach_max_offset: Maximum XY distance from target for offset waypoint (meters).
        approach_height: Base height above target for the offset waypoint (meters).
        approach_height_noise: Random additional height variation (meters).
        num_path_points: Number of intermediate waypoints for noisy_path mode.
        path_pos_noise: Maximum XY deviation per waypoint in noisy_path mode (meters).
        path_height_noise: Maximum Z deviation per waypoint in noisy_path mode (meters).
    """

    mode: str = "none"

    perturb_pick: bool = True
    perturb_place: bool = True

    # Offset approach parameters
    approach_min_offset: float = 0.01
    approach_max_offset: float = 0.04
    approach_height: float = 0.06
    approach_height_noise: float = 0.02

    # Noisy path parameters
    num_path_points: int = 5
    path_pos_noise: float = 0.008
    path_height_noise: float = 0.005


def generate_offset_approach(target_pose: Pose, config: PerturbationConfig) -> list:
    """Generate a single waypoint on a disk above the target.

    The waypoint is placed at a random angle and radius from the target,
    at a height above it. This causes the robot to approach the target
    from a different direction each time.

    Args:
        target_pose: The final target pose the robot will move to.
        config: Perturbation configuration.

    Returns:
        A list containing one Pose (the offset approach waypoint).
    """
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(config.approach_min_offset, config.approach_max_offset)

    waypoint = copy.deepcopy(target_pose)
    waypoint.position.x += radius * np.cos(angle)
    waypoint.position.y += radius * np.sin(angle)
    waypoint.position.z += config.approach_height + np.random.uniform(
        0, config.approach_height_noise
    )
    return [waypoint]


def generate_noisy_path(
    current_pose: Pose, target_pose: Pose, config: PerturbationConfig
) -> list:
    """Generate intermediate waypoints along the path from current to target.

    Points are linearly interpolated between current_pose and target_pose,
    then each is jittered by a small random offset. This creates a slightly
    wobbly path rather than a straight line, while still converging on the target.

    The orientation is kept identical to current_pose for all waypoints so that
    the robot does not attempt a large reorientation at the first waypoint.
    The final orientation is handled by the subsequent grasp_at / release_at call.

    Args:
        current_pose: The robot's current end-effector pose.
        target_pose: The final target pose.
        config: Perturbation configuration.

    Returns:
        A list of Pose waypoints (excluding start and end).
    """
    waypoints = []
    for i in range(1, config.num_path_points + 1):
        alpha = i / (config.num_path_points + 1)

        wp = copy.deepcopy(current_pose)
        wp.position.x = (1 - alpha) * current_pose.position.x + alpha * target_pose.position.x
        wp.position.y = (1 - alpha) * current_pose.position.y + alpha * target_pose.position.y
        wp.position.z = (1 - alpha) * current_pose.position.z + alpha * target_pose.position.z

        wp.position.x += np.random.uniform(-config.path_pos_noise, config.path_pos_noise)
        wp.position.y += np.random.uniform(-config.path_pos_noise, config.path_pos_noise)
        wp.position.z += np.random.uniform(-config.path_height_noise, config.path_height_noise)

        waypoints.append(wp)
    return waypoints


def generate_waypoints(
    current_pose, target_pose: Pose, config: PerturbationConfig
) -> list:
    """Generate perturbation waypoints based on the configured mode.

    This is the main entry point. Call this before a pick or place action
    and execute each returned waypoint via robot.move_to().

    Args:
        current_pose: The robot's current end-effector pose, or None.
                      Required for "noisy_path" and "both" modes.
        target_pose: The final target pose for the upcoming action.
        config: Perturbation configuration.

    Returns:
        A list of Pose waypoints to visit before the target action.
        Returns an empty list when mode is "none" or inputs are insufficient.
    """
    if config.mode == "none":
        return []

    if config.mode == "offset_approach":
        return generate_offset_approach(target_pose, config)

    if config.mode == "noisy_path":
        if current_pose is None:
            return []
        return generate_noisy_path(current_pose, target_pose, config)

    if config.mode == "both":
        offset_wp = generate_offset_approach(target_pose, config)
        if current_pose is None:
            return offset_wp
        noisy = generate_noisy_path(current_pose, offset_wp[0], config)
        return noisy + offset_wp

    return []
