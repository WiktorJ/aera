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

home_offset (composable flag)
    Small random offsets are added to the home joint angles so the robot
    starts each episode from a slightly different configuration.  Enable via
    ``perturb_home=True`` on ``PerturbationConfig``.  This is orthogonal to
    the other modes and can be combined with any of them.

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

    # Home offset — start from a slightly perturbed home position:
    config = PerturbationConfig(perturb_home=True)
    go_home_perturbed(robot, config)  # instead of robot.go_home()
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Dict, List, Literal, Tuple

import numpy as np
from geometry_msgs.msg import Pose

from aera_semi_autonomous.control.ar4_mk3_interface_config import (
    ActuationConfig,
    Ar4Mk3InterfaceConfig,
    IKConfig,
)

if TYPE_CHECKING:
    from aera_semi_autonomous.control.ar4_mk3_robot_interface import (
        Ar4Mk3RobotInterface,
    )


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
class HomeOffsetPerturbation:
    """Additive noise for randomizing the robot's home joint positions.

    Each arm joint's home angle is offset by a sample from
    ``Uniform(-max_offset, +max_offset)`` (radians).

    Attributes:
        default_max_offset: Max offset applied to all joints (radians).
            Default ~0.05 rad ≈ 2.9°.
        per_joint_max_offsets: Optional per-joint overrides.  Dict mapping
            joint index (0-5) to a max offset in radians.  Joints not
            present fall back to ``default_max_offset``.
    """

    default_max_offset: float = 0.05
    per_joint_max_offsets: Dict[int, float] = field(default_factory=dict)


@dataclass
class ActuationPerturbation:
    """Per-episode sampling ranges for control-loop realism (see
    ``ActuationConfig`` for what each resolved value does).

    Ranges are deliberately conservative so the scripted IK + interpolation
    loops still converge inside their step budgets — the goal is realistic
    motion-profile *diversity*, not a faithful servo model. The arm's high
    position gains plus the IK's 700-step budget absorb mild lag/delay; push
    these much harder and grasp success starts to drop.

    Attributes:
        latency_steps_range: Inclusive (lo, hi) for the whole-step command
            delay. At the 0.002 s sim timestep, 0-4 steps ≈ 0-8 ms.
        command_lag_alpha_range: (lo, hi) for the first-order low-pass coeff.
            Smaller = laggier; ~0.2 gives a ~10 ms time constant, ~0.8 is
            nearly crisp, so the range spans clearly-laggy to almost-perfect.
        step_jitter_prob_range: (lo, hi) for the per-advance extra-settle-step
            probability.
    """

    latency_steps_range: tuple = (0, 4)
    command_lag_alpha_range: tuple = (0.2, 0.8)
    step_jitter_prob_range: tuple = (0.0, 0.1)


@dataclass
class SpeedPerturbation:
    """Per-episode motion-tempo sampling. One factor scales the whole arm's
    speed so the recorded action-deltas (and frame cadence) aren't locked to a
    single tempo.

    factor > 1 = faster/coarser moves; < 1 = slower/finer. The factor scales the
    IK step size up and the interpolation step counts down; IK max_steps scales
    inversely so slow episodes still have the budget to converge.

    factor_range is kept moderate: too fast and the IK overshoots / fails to
    converge, too slow and long moves exhaust the (scaled) step budget — either
    way costing usable demos.
    """

    factor_range: tuple = (0.7, 1.4)


@dataclass
class HoverHeightPerturbation:
    """Per-episode pre-grasp / pre-place hover height (the interface's
    ``above_target_offset``). Diversifies the descend geometry and the
    pre-grasp images instead of always hovering at the fixed 0.05 m."""

    offset_range: tuple = (0.04, 0.10)


@dataclass
class RecoveryPerturbation:
    """Per-episode grasp-time failure + recovery (sim2real plan #1+#2).

    Real grasps fail at grasp time, not mid-transport — once the object is in
    the jaws, friction holds it. Two failure modes are reproduced here, both
    ending in a correct grasp and, crucially, NEITHER ever pressing the object
    into the table (which would teach the policy to force the arm into a rigid
    object — a real-world disaster):

      - wrong_approach: the gripper lines up over the WRONG spot at hover height,
        then corrects laterally and descends cleanly. The mis-alignment lives
        only at hover, so the gripper never touches the object while off-target.
      - partial_grasp: a centred grasp with the jaw/object contact friction
        temporarily lowered (the kinematic lock is NOT engaged) so the block
        lifts a little then slides out from between the jaws and drops back onto
        the table — a real "too slippery to hold" slip, not a clean release —
        then the expert re-detects and re-grasps it.

    NOT probabilities: when ``PerturbationConfig.perturb_recovery`` is on, each
    enabled mode fires every episode (A/B by collecting twice). The toggles
    switch a mode off structurally; magnitudes are sampled per-episode.

    Attributes:
        wrong_approach: Enable the hover-misalign-then-correct failure.
        wrong_approach_offset_range: Lateral mis-alignment (m) at hover, sampled
            at a random heading.
        wrong_approach_hover_range: Height (m) above the object top for the
            mis-approach — kept well above the object so the descent is never
            triggered while off-target (no contact, no pressing).
        partial_grasp: Enable the marginal top-edge grasp that slips on lift.
        partial_grasp_lift_range: How far (m) the arm lifts before/while the
            block tips out — small, so it slips shortly after pickup.
        partial_grasp_slip_friction: Tangential contact friction applied to BOTH
            the jaws and the block during the slip attempt (MuJoCo takes the max
            of the two geoms' friction, so both must be lowered). A centred grasp
            at this friction lifts the block a little then lets it slide out and
            drop back onto the table. Verified in sim: ~0.45-0.6 slips reliably,
            >=0.7 holds, <=0.3 never lifts. The table-block friction is untouched
            so the block still settles normally, on the table, for the re-grasp.
        partial_grasp_pause_prob: Probability of pausing briefly after the jaws
            close, before the lift, so some slips look like a completed grasp
            that then loses the block (rather than a lift that began before the
            jaws finished closing). The rest get no pause for variety.
        partial_grasp_pause_steps: Length of that pause, in sim steps (~2 ms
            each).
        max_grasp_retries: Number of partial-grasp slips before the successful
            grasp.
    """

    wrong_approach: bool = True
    wrong_approach_offset_range: tuple = (0.018, 0.035)
    wrong_approach_hover_range: tuple = (0.05, 0.10)

    partial_grasp: bool = True
    partial_grasp_lift_range: tuple = (0.015, 0.035)
    partial_grasp_slip_friction: float = 0.55
    partial_grasp_pause_prob: float = 0.5
    partial_grasp_pause_steps: int = 40

    max_grasp_retries: int = 1


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
        perturb_home: Whether to perturb the home joint positions before the episode.
        home_offset: Noise configuration for the home position perturbation.
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

    perturb_home: bool = False
    home_offset: HomeOffsetPerturbation = field(default_factory=HomeOffsetPerturbation)

    ik_noise: IKNoisePerturbation = field(default_factory=IKNoisePerturbation)

    # Control-loop realism (latency / command lag / step jitter). Composable
    # flag like perturb_home — orthogonal to `mode`. When True, the collection
    # script samples an ActuationConfig per episode and passes it on the
    # interface config.
    perturb_actuation: bool = False
    actuation: ActuationPerturbation = field(default_factory=ActuationPerturbation)

    # Motion dynamics (composable, orthogonal to `mode`). perturb_speed scales
    # the per-episode arm tempo; perturb_hover_height varies the pre-grasp /
    # pre-place hover height. The collection script samples each per episode and
    # rewrites the interface config accordingly.
    perturb_speed: bool = False
    speed: SpeedPerturbation = field(default_factory=SpeedPerturbation)
    perturb_hover_height: bool = False
    hover_height: HoverHeightPerturbation = field(
        default_factory=HoverHeightPerturbation
    )

    # Recovery / off-manifold data (composable, orthogonal to `mode`). When on,
    # the collection loop injects deliberate missed grasps + corrective detours
    # so the policy sees recovery, not only clean successes. Enable/disable for
    # the whole run; A/B by collecting twice.
    perturb_recovery: bool = False
    recovery: RecoveryPerturbation = field(default_factory=RecoveryPerturbation)


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


def sample_wrong_approach_poses(
    object_pose: Pose, recovery: RecoveryPerturbation
) -> Tuple[Pose, Pose]:
    """Return ``(bad_hover, good_hover)`` for the wrong-approach failure.

    Both poses sit at the same sampled hover height above the object top;
    ``bad_hover`` is offset laterally (the wrong spot the arm lines up over),
    ``good_hover`` is directly above the object. The arm visits bad then good,
    recording a lateral correction — all at hover height, so the gripper never
    descends onto (and never touches) the object while mis-aligned. Orientation
    is the object's top-down grasp."""
    angle = np.random.uniform(0, 2 * np.pi)
    radius = np.random.uniform(*recovery.wrong_approach_offset_range)
    hover = np.random.uniform(*recovery.wrong_approach_hover_range)
    good = copy.deepcopy(object_pose)
    good.position.z += hover
    bad = copy.deepcopy(good)
    bad.position.x += radius * np.cos(angle)
    bad.position.y += radius * np.sin(angle)
    return bad, good


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


def perturb_home_qpos(
    home_qpos: np.ndarray,
    config: HomeOffsetPerturbation,
    num_arm_joints: int = 6,
) -> np.ndarray:
    """Return a copy of home_qpos with random additive offsets on arm joints.

    Each arm joint (indices 0 to ``num_arm_joints - 1``) is offset by
    ``Uniform(-max_offset, +max_offset)`` where ``max_offset`` comes from
    ``per_joint_max_offsets`` if present, else ``default_max_offset``.

    Args:
        home_qpos: The baseline home joint positions.
        config: Home offset perturbation configuration.
        num_arm_joints: Number of arm joints to perturb (default 6).

    Returns:
        A new array with perturbed joint positions.
    """
    result = home_qpos.copy()
    for i in range(min(num_arm_joints, len(result))):
        max_off = config.per_joint_max_offsets.get(i, config.default_max_offset)
        result[i] += np.random.uniform(-max_off, max_off)
    return result


def sample_actuation_config(noise: ActuationPerturbation) -> ActuationConfig:
    """Draw a per-episode ActuationConfig from the given ranges.

    Each episode gets one fixed (latency, lag, jitter) triple so the arm has a
    consistent "feel" within the trajectory; variety comes from resampling
    across episodes.
    """
    lo, hi = noise.latency_steps_range
    latency_steps = int(np.random.randint(lo, hi + 1))
    command_lag_alpha = float(np.random.uniform(*noise.command_lag_alpha_range))
    step_jitter_prob = float(np.random.uniform(*noise.step_jitter_prob_range))
    return ActuationConfig(
        latency_steps=latency_steps,
        command_lag_alpha=command_lag_alpha,
        step_jitter_prob=step_jitter_prob,
    )


def apply_speed_perturbation(
    interface_config: Ar4Mk3InterfaceConfig, noise: SpeedPerturbation
) -> Ar4Mk3InterfaceConfig:
    """Return a copy of `interface_config` with a per-episode tempo factor baked
    in: IK step size up, interpolation step counts down, IK budget up to keep
    slow episodes converging. See SpeedPerturbation."""
    s = float(np.random.uniform(*noise.factor_range))
    ik = interface_config.ik
    new_ik = replace(
        ik,
        integration_dt=ik.integration_dt * s,
        max_update_norm=ik.max_update_norm * s,
        max_steps=max(1, int(round(ik.max_steps / s))),
    )
    return replace(
        interface_config,
        ik=new_ik,
        go_home_interpolation_steps=max(
            1, int(round(interface_config.go_home_interpolation_steps / s))
        ),
        gripper_action_steps=max(
            1, int(round(interface_config.gripper_action_steps / s))
        ),
    )


def apply_hover_height_perturbation(
    interface_config: Ar4Mk3InterfaceConfig, noise: HoverHeightPerturbation
) -> Ar4Mk3InterfaceConfig:
    """Return a copy of `interface_config` with a per-episode pre-grasp/place
    hover height (``above_target_offset``)."""
    offset = float(np.random.uniform(*noise.offset_range))
    return replace(interface_config, above_target_offset=offset)


def go_home_perturbed(robot: Ar4Mk3RobotInterface, config: PerturbationConfig) -> bool:
    """Move the robot home, optionally with a perturbed home position.

    When ``config.perturb_home`` is True, the robot moves to a slightly
    offset version of the home joint configuration.  Otherwise this is
    equivalent to ``robot.go_home()``.

    Args:
        robot: The robot interface.
        config: Perturbation configuration.

    Returns:
        True if the robot reached the (perturbed) home position.
    """
    if not config.perturb_home:
        return robot.go_home()

    if not robot.go_home():
        return False

    home_qpos = robot.get_home_qpos()
    perturbed = perturb_home_qpos(home_qpos, config.home_offset)
    robot.teleport_to_qpos(perturbed)
    return True
