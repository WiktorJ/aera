"""Trajectory perturbation utilities for generating movement diversity.

Provides perturbation modes that can be applied before pick/place actions
to create varied arm trajectories while preserving the exact grasp/release targets.

The robot always grasps at the exact object pose and places at the exact target
pose. What varies is the path it takes to get there.

Modes
-----
offset_approach
    The robot visits one or more random waypoints on a disk above the target
    before descending.

ik_noise
    IK solver config values are randomized before constructing the robot
    interface. Use ``perturb_ik_config`` to obtain a noisy ``IKConfig`` and
    pass it inside ``Ar4Mk3InterfaceConfig`` when creating the robot interface.

Usage:
    config = PerturbationConfig(mode="offset_approach", num_approach_waypoints=2)
    waypoints = generate_waypoints(target_pose, config)
    for wp in waypoints:
        robot.move_to(wp)
    robot.grasp_at(target_pose, gripper_pos=0.0)

    # IK noise (applied once per episode, before robot construction):
    config = PerturbationConfig(mode="ik_noise", ik_noise=IKNoisePerturbation(...))
    noisy_ik = perturb_ik_config(IKConfig(), config.ik_noise)
    interface_config = Ar4Mk3InterfaceConfig(ik=noisy_ik)
    robot = Ar4Mk3RobotInterface(env, config=interface_config)
"""

import copy
from dataclasses import dataclass, field
from typing import List, Literal

import numpy as np
from geometry_msgs.msg import Pose

from aera_semi_autonomous.control.ar4_mk3_interface_config import IKConfig


@dataclass
class IKNoisePerturbation:
    """Noise ranges for randomizing IK solver config values.

    Each field specifies the half-width of a uniform distribution centred on
    the base ``IKConfig`` value.  A value of ``0.0`` (the default) means no
    noise is applied to that parameter.

    Attributes:
        pos_gain_noise: ± noise on ``IKConfig.pos_gain``.
        orientation_gain_noise: ± noise on ``IKConfig.orientation_gain``.
        integration_dt_noise: ± noise on ``IKConfig.integration_dt``.
        max_update_norm_noise: ± noise on ``IKConfig.max_update_norm``.
        regularization_strength_noise: ± noise on ``IKConfig.regularization_strength``.
        joints_update_scaling_noise: ± per-joint noise on ``IKConfig.joints_update_scaling``.
    """

    pos_gain_noise: float = 0.0
    orientation_gain_noise: float = 0.0
    integration_dt_noise: float = 0.0
    max_update_norm_noise: float = 0.0
    regularization_strength_noise: float = 0.0
    joints_update_scaling_noise: float = 0.0


@dataclass
class PerturbationConfig:
    """Configuration for trajectory perturbation.

    Attributes:
        mode: Perturbation mode. One of "none", "offset_approach", or "ik_noise".
        perturb_pick: Whether to perturb the pick (grasp) approach.
        perturb_place: Whether to perturb the place (release) approach.
        num_approach_waypoints: Number of offset waypoints to generate (default 1).
        approach_min_offset: Minimum XY distance from target for offset waypoint (meters).
        approach_max_offset: Maximum XY distance from target for offset waypoint (meters).
        approach_height: Base height above target for the offset waypoint (meters).
        approach_height_noise: Random additional height variation (meters).
        ik_noise: Noise configuration for the "ik_noise" mode.
    """

    mode: Literal["none", "offset_approach", "ik_noise"] = "none"

    perturb_pick: bool = True
    perturb_place: bool = True

    num_approach_waypoints: int = 1
    approach_min_offset: float = 0.01
    approach_max_offset: float = 0.04
    approach_height: float = 0.06
    approach_height_noise: float = 0.02

    ik_noise: IKNoisePerturbation = field(default_factory=IKNoisePerturbation)


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
        Returns an empty list when mode is "none" or "ik_noise".
    """
    if config.mode == "none":
        return []

    if config.mode == "offset_approach":
        return generate_offset_approach(target_pose, config)

    return []


def perturb_ik_config(base: IKConfig, noise: IKNoisePerturbation) -> IKConfig:
    """Return a new IKConfig with each float field perturbed by uniform noise.

    Each parameter is sampled from ``Uniform(base - noise, base + noise)``.
    Gains and norms are clamped to stay positive.  Integer and boolean fields
    are copied unchanged.

    Args:
        base: The baseline IK solver configuration.
        noise: Half-widths of the uniform noise distributions.

    Returns:
        A new IKConfig instance with randomized values.
    """

    def _noisy(value: float, half_width: float) -> float:
        if half_width == 0.0:
            return value
        return float(value + np.random.uniform(-half_width, half_width))

    noisy_scaling: List[float] = [
        max(1e-6, _noisy(s, noise.joints_update_scaling_noise))
        for s in base.joints_update_scaling
    ]

    return IKConfig(
        tolerance=base.tolerance,
        regularization_threshold=base.regularization_threshold,
        regularization_strength=max(
            0.0, _noisy(base.regularization_strength, noise.regularization_strength_noise)
        ),
        max_update_norm=max(1e-6, _noisy(base.max_update_norm, noise.max_update_norm_noise)),
        integration_dt=max(1e-6, _noisy(base.integration_dt, noise.integration_dt_noise)),
        pos_gain=max(1e-6, _noisy(base.pos_gain, noise.pos_gain_noise)),
        orientation_gain=max(1e-6, _noisy(base.orientation_gain, noise.orientation_gain_noise)),
        max_steps=base.max_steps,
        min_height=base.min_height,
        include_rotation_in_target_error_measure=base.include_rotation_in_target_error_measure,
        joints_update_scaling=noisy_scaling,
    )
