import dataclasses
import typing
from dataclasses import field

DEFAULT_CAMERA_CONFIG: typing.Dict[str, typing.Any] = {
    "width": 640,
    "height": 480,
    "fx": 525.0,
    "fy": 525.0,
    "cx": 320.0,
    "cy": 240.0,
}


@dataclasses.dataclass
class Ar4Mk3InterfaceConfig:
    camera_config: typing.Dict[str, typing.Any] = field(
        default_factory=lambda: DEFAULT_CAMERA_CONFIG
    )
    move_to_pos_tolerance: float = 1e-3
    above_target_offset: float = 0.05
    gripper_action_steps: int = 50
    go_home_interpolation_steps: int = 100
    home_qpos_error_tolerance: float = 1e-3
    gripper_pos_tolerance: float = 1e-3
    render_steps: bool = False
    ik_tolerance: float = 1e-3
    ik_regularization_threshold: float = 1e-5
    ik_regularization_strength: float = 1e-3
    ik_max_update_norm: float = 0.75
    ik_integration_dt: float = 0.1
    ik_pos_gain: float = 0.95
    ik_orientation_gain: float = 0.95
    ik_max_steps: int = 1000
    ik_min_height: float = 0.005
    ik_include_rotation_in_target_error_measure: bool = False
    ik_joints_update_scaling: typing.List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 0.01, 1.0, 1.0]
    )
