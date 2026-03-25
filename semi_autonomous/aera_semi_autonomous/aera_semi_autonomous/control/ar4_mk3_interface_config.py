import dataclasses
import typing
from dataclasses import field


@dataclasses.dataclass
class IKConfig:
    """Configuration for the IK solver."""

    tolerance: float = 1e-3
    regularization_threshold: float = 1e-5
    regularization_strength: float = 1e-3
    max_update_norm: float = 0.75
    integration_dt: float = 0.1
    pos_gain: float = 0.95
    orientation_gain: float = 0.95
    max_steps: int = 500
    min_height: float = 0.005
    include_rotation_in_target_error_measure: bool = False
    joints_update_scaling: typing.List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 0.01, 1.0, 1.0]
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
