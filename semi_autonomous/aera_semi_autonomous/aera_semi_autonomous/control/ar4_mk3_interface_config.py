import dataclasses
import typing
from dataclasses import field


# IKConfig defaults were tuned empirically via iterative one-parameter-at-a-time sweeps
# (20 trials per value, with domain randomization). Approximate full pick-and-place
# success rates at the chosen defaults:
#   pos_gain=0.95                              → ~50%
#   orientation_gain=1.5                       → ~80%
#   integration_dt=0.10                        → ~70%
#   max_update_norm=2.00                       → ~60% (plateau; higher values offer no gain)
#   regularization_strength=1e-4               → ~70%
#   joints_update_scaling=[1.2,1.0,0.85,0.02,0.7,0.3]
#     joint 0 (shoulder pan):  1.2  → ~65%
#     joint 1 (shoulder lift): 1.0  → ~50% (default is optimal)
#     joint 2 (elbow):         0.85 → ~65%
#     joint 3 (wrist rot):     0.02 → ~65%
#     joint 4 (wrist pitch):   0.7  → ~50%
#     joint 5 (wrist roll):    0.3  → ~65%
# Note: even with optimal defaults, convergence is not guaranteed for all poses.
@dataclasses.dataclass
class IKConfig:
    """Configuration for the IK solver."""

    tolerance: float = 1e-3
    regularization_threshold: float = 1e-5
    regularization_strength: float = 1e-4
    max_update_norm: float = 2.0
    integration_dt: float = 0.1
    pos_gain: float = 0.95
    orientation_gain: float = 1.5
    max_steps: int = 500
    min_height: float = 0.005
    include_rotation_in_target_error_measure: bool = False
    joints_update_scaling: typing.List[float] = field(
        default_factory=lambda: [1.2, 1.0, 0.85, 0.02, 0.7, 0.3]
    )


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
