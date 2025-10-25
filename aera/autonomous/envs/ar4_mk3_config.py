import dataclasses
import typing
from dataclasses import field

import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.98,
    "azimuth": -133,
    "elevation": -26,
    "lookat": np.array([0, 0, 0]),
}


@dataclasses.dataclass
class Ar4Mk3EnvConfig:
    model_path: str
    n_substeps: int = 20
    gripper_extra_height: float = 0.2
    block_gripper: bool = False
    has_object: bool = True
    target_in_the_air: bool = False
    target_offset: tuple[float, float, float] = (0.0, -0.04, 0.01)
    obj_range: tuple[float, float] = (0.09, 0.08)
    obj_offset: tuple[float, float] = (0.0, -0.04)
    target_range: float = 0.13
    distance_threshold: float = 0.05
    reward_type: str = "sparse"
    object_size: tuple[float, float, float] = (0.012, 0.012, 0.062)
    use_eef_control: bool = False
    initial_qpos: dict = field(
        default_factory=lambda: {
            "robot0:slide0": 0.0,
            "robot0:slide1": 0.0,
            "robot0:slide2": 0.0,
        }
    )
    translation: typing.Optional[np.ndarray] = None
    quaterion: typing.Optional[np.ndarray] = None
    z_offset: float = 0.0
    distance_multiplier: float = 1.0
    default_camera_config: dict = field(default_factory=lambda: DEFAULT_CAMERA_CONFIG)
