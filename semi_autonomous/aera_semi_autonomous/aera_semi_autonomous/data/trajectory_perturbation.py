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

    # IK noise — 10% multiplicative noise on all params:
    config = PerturbationConfig(
        mode="ik_noise",
        ik_noise=IKNoisePerturbation(default_fraction=0.1),
    )
    noisy_ik = perturb_ik_config(IKConfig(), config.ik_noise)
    interface_config = Ar4Mk3InterfaceConfig(ik=noisy_ik)
    robot = Ar4Mk3RobotInterface(env, config=interface_config)

    # Per-joint override — 5% on joint 3 (wrist roll), 15% on the rest:
    config = PerturbationConfig(
        mode="ik_noise",
        ik_noise=IKNoisePerturbation(
            default_fraction=0.15,
            per_joint_scaling_fractions={3: 0.05},
        ),
    )
"""

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Literal

import numpy as np
from geometry_msgs.msg import Pose

from aera_semi_autonomous.control.ar4_mk3_interface_config import IKConfig


@dataclass
class IKNoisePerturbation:
    """Multiplicative noise for randomizing IK solver config values.

    Each field specifies the fraction of the base value used as the half-width
    of a uniform distribution.  For a parameter with base value ``v`` and
    fraction ``f``, the noisy value is sampled from
    ``v * Uniform(1 - f, 1 + f)``.

    A fraction of ``0.0`` (the default) means no noise.  Set
    ``default_fraction`` to apply the same noise level to every parameter at
    once; per-parameter fields override the default when set to a value > 0.

    Attributes:
        default_fraction: Global noise fraction applied to all parameters
            unless overridden by a per-parameter field.
        pos_gain_fraction: Override for ``IKConfig.pos_gain``.
        orientation_gain_fraction: Override for ``IKConfig.orientation_gain``.
        integration_dt_fraction: Override for ``IKConfig.integration_dt``.
        max_update_norm_fraction: Override for ``IKConfig.max_update_norm``.
        regularization_strength_fraction: Override for ``IKConfig.regularization_strength``.
        joints_update_scaling_fraction: Default fraction for all joints in
            ``IKConfig.joints_update_scaling``.  Individual joints can be
            overridden via ``per_joint_scaling_fractions``.
        per_joint_scaling_fractions: Optional per-joint fraction overrides.
            A dict mapping joint index (0-5) to a fraction.  Joints not
            present fall back to ``joints_update_scaling_fraction``, then
            ``default_fraction``.
    """

    default_fraction: float = 0.0
    pos_gain_fraction: float = 0.0
    orientation_gain_fraction: float = 0.0
    integration_dt_fraction: float = 0.0
    max_update_norm_fraction: float = 0.0
    regularization_strength_fraction: float = 0.0
    joints_update_scaling_fraction: float = 0.0
    per_joint_scaling_fractions: Dict[int, float] = field(default_factory=dict)


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


def _effective_fraction(
    per_param: float, noise: IKNoisePerturbation
) -> float:
    """Return per-param fraction if > 0, else fall back to default_fraction."""
    return per_param if per_param > 0.0 else noise.default_fraction


def _noisy_mul(value: float, fraction: float) -> float:
    """Multiply *value* by ``Uniform(1 - fraction, 1 + fraction)``."""
    if fraction == 0.0:
        return value
    return float(value * np.random.uniform(1.0 - fraction, 1.0 + fraction))


def perturb_ik_config(base: IKConfig, noise: IKNoisePerturbation) -> IKConfig:
    """Return a new IKConfig with each float field perturbed by multiplicative noise.

    For a parameter with base value ``v`` and effective fraction ``f``, the
    noisy value is ``v * Uniform(1 - f, 1 + f)``.  This keeps perturbations
    proportional to the parameter's scale (e.g. joint 3 at 0.01 gets ±0.001
    with 10% noise, while joint 0 at 1.0 gets ±0.1).

    Per-parameter fractions override ``default_fraction`` when set to > 0.
    For ``joints_update_scaling``, each joint can be individually overridden
    via ``per_joint_scaling_fractions``; joints not listed fall back to
    ``joints_update_scaling_fraction``, then ``default_fraction``.

    All results are clamped to stay positive.

    Args:
        base: The baseline IK solver configuration.
        noise: Multiplicative noise fractions.

    Returns:
        A new IKConfig instance with randomized values.
    """
    joint_default_frac = (
        noise.joints_update_scaling_fraction
        if noise.joints_update_scaling_fraction > 0.0
        else noise.default_fraction
    )
    noisy_scaling: List[float] = []
    for idx, s in enumerate(base.joints_update_scaling):
        frac = noise.per_joint_scaling_fractions.get(idx, joint_default_frac)
        noisy_scaling.append(max(1e-6, _noisy_mul(s, frac)))

    return IKConfig(
        tolerance=base.tolerance,
        regularization_threshold=base.regularization_threshold,
        regularization_strength=max(
            0.0,
            _noisy_mul(
                base.regularization_strength,
                _effective_fraction(noise.regularization_strength_fraction, noise),
            ),
        ),
        max_update_norm=max(
            1e-6,
            _noisy_mul(
                base.max_update_norm,
                _effective_fraction(noise.max_update_norm_fraction, noise),
            ),
        ),
        integration_dt=max(
            1e-6,
            _noisy_mul(
                base.integration_dt,
                _effective_fraction(noise.integration_dt_fraction, noise),
            ),
        ),
        pos_gain=max(
            1e-6,
            _noisy_mul(
                base.pos_gain,
                _effective_fraction(noise.pos_gain_fraction, noise),
            ),
        ),
        orientation_gain=max(
            1e-6,
            _noisy_mul(
                base.orientation_gain,
                _effective_fraction(noise.orientation_gain_fraction, noise),
            ),
        ),
        max_steps=base.max_steps,
        min_height=base.min_height,
        include_rotation_in_target_error_measure=base.include_rotation_in_target_error_measure,
        joints_update_scaling=noisy_scaling,
    )
