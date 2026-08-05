import dataclasses
import typing
from dataclasses import field

from aera.autonomous.envs.kinematic_grasp import GraspEngageConfig
from aera.autonomous.envs.task_env_factory import COLLECTION_RECORD_EVERY


@dataclasses.dataclass
class IKConfig:
    """Configuration for the IK solver."""

    tolerance: float = 1e-3
    regularization_threshold: float = 1e-5
    regularization_strength: float = 1e-4
    max_update_norm: float = 1.5
    # The demonstrated arm's timescale. 0.15 ran the whole pick-and-place in
    # ~0.7 s at ~104 cm/s EEF — 5-20x a realistic collaborative arm — which left
    # the grasp descent commanding ~5% of the model's normalized output range.
    # 0.005 lands at 4.8-6.6 s / 12-15 cm/s. The response is strongly sublinear
    # (integration_dt only rate-limits while the max_update_norm clamp binds),
    # so this is ~30x the parameter for ~8x the time; measure, don't
    # extrapolate. EEF path length is unchanged, so only the timing shifts.
    integration_dt: float = 0.005
    pos_gain: float = 0.95
    orientation_gain: float = 1.1
    # Must scale with the slower dt or every episode aborts ("could not move
    # above target"). 3000 measured sufficient at dt=0.005 (3/3 seeds, no
    # aborts). Note perturb_recovery's ik perturbation scales integration_dt by
    # +/-10% WITHOUT scaling this, so the worst perturbed case has proportionally
    # less budget than the unperturbed path verified here.
    max_steps: int = 3000
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
    # Must stay BELOW the grasp squeeze it is supposed to deliver (0.6 mm), or
    # the close converges and exits before the jaws reach the block. At the old
    # 1 mm it was 10x the residual preload, which is the second of the two
    # stacked errors that left every scripted close short of first contact.
    gripper_pos_tolerance: float = 1e-4
    render_steps: bool = False
    # Record the depth cameras alongside RGB. Off: nothing downstream consumes
    # depth, and it is half the per-frame render cost plus a ~1.2 GB/episode RAM
    # spike at the slow arm's frame counts. See _record_step.
    record_depth: bool = False
    # Record one frame every N mj-steps instead of every one. Recording is ~99%
    # of a frame's cost (measured: 13.7 ms for the two camera renders against
    # 0.09 ms for mj_step + mj_forward, and the renders are per-call cost, not
    # per-pixel — 640x480 costs the same as 224x224), so collection wall-clock is
    # essentially raw_frames * 2 renders. The slow arm runs ~2500 mj-steps per
    # episode where the fast one ran ~400, and the transform's --skip then
    # discards most of them, so recording every step renders ~5x the frames that
    # reach the dataset.
    #
    # This is NOT a rate change: decimating here and decimating at transform are
    # exactly equivalent, and the deploy invariant absorbs it as
    # n_substeps == record_every * skip (see task_env_factory, CONTROL_RATE_SPEC).
    # 5 rather than 10 so the plan's own contingency stays cheap — if the batch
    # shows the descent is too coarse, re-transforming at --skip 1 gives a 10 ms
    # dataset without re-collecting.
    record_every: int = COLLECTION_RECORD_EVERY
    ik: IKConfig = field(default_factory=IKConfig)
    actuation: ActuationConfig = field(default_factory=ActuationConfig)
    # Gate for the kinematic grasp lock. Default permissive (old 5cm snap); the
    # collection scripts flip require_alignment on when injecting recovery data
    # so a deliberate near-miss genuinely fails to grab. Shared definition with
    # the eval env so collection and eval grasp under identical rules.
    grasp_engage: GraspEngageConfig = field(default_factory=GraspEngageConfig)
