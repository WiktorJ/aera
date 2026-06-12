import dataclasses
import typing
from dataclasses import field

from aera.autonomous.envs.kinematic_grasp import GraspEngageConfig


@dataclasses.dataclass
class IKConfig:
    """Configuration for the IK solver."""

    tolerance: float = 1e-3
    regularization_threshold: float = 1e-5
    regularization_strength: float = 1e-4
    max_update_norm: float = 1.5
    integration_dt: float = 0.15
    pos_gain: float = 0.95
    orientation_gain: float = 1.1
    max_steps: int = 700
    min_height: float = 0.005
    include_rotation_in_target_error_measure: bool = False
    joints_update_scaling: typing.List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    )


@dataclasses.dataclass
class ActuationConfig:
    """Per-episode control-loop realism for demo collection.

    Makes the arm an imperfect tracker of the scripted IK targets so the
    recorded trajectories show realistic command lag / delay rather than the
    near-instant settling of a perfect position servo. The interface applies
    this to the arm actuators (act1..act6 only — the gripper is left crisp)
    right before each sim step; resolved values are sampled per episode by
    ``sample_actuation_config`` in trajectory_perturbation.py.

    Defaults are the identity (no effect), so an interface built without an
    explicit actuation config behaves exactly as before.

    Attributes:
        latency_steps: Whole-step delay between when the IK loop commands a
            ctrl and when the sim sees it (a ring buffer of depth
            latency_steps). 0 = no delay.
        command_lag_alpha: First-order low-pass coefficient on the (delayed)
            command, applied per sim step as
            ``applied += alpha * (commanded - applied)``. 1.0 = no lag; smaller
            = laggier (time constant ~ dt / alpha).
        step_jitter_prob: Per-advance probability of inserting one extra
            settle step (the arm coasts under the current applied ctrl before
            the next command), modelling an irregular control-loop tick. 0 =
            off.
    """

    latency_steps: int = 0
    command_lag_alpha: float = 1.0
    step_jitter_prob: float = 0.0


@dataclasses.dataclass
class Ar4Mk3InterfaceConfig:
    move_to_pos_tolerance: float = 1e-3
    above_target_offset: float = 0.05
    gripper_action_steps: int = 50
    go_home_interpolation_steps: int = 100
    home_qpos_error_tolerance: float = 1e-3
    gripper_pos_tolerance: float = 1e-3
    render_steps: bool = False
    ik: IKConfig = field(default_factory=IKConfig)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)
    # Gate for the kinematic grasp lock. Default permissive (old 5cm snap); the
    # collection scripts flip require_alignment on when injecting recovery data
    # so a deliberate near-miss genuinely fails to grab. Shared definition with
    # the eval env so collection and eval grasp under identical rules.
    grasp_engage: GraspEngageConfig = field(default_factory=GraspEngageConfig)
