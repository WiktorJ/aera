import dataclasses
from dataclasses import field
from typing import Optional, Sequence

import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.98,
    "azimuth": -133,
    "elevation": -26,
    "lookat": np.array([0, 0, 0]),
}


@dataclasses.dataclass
class MaterialConfig:
    rgba: Optional[Sequence[float]] = None
    specular: Optional[float] = None
    shininess: Optional[float] = None
    reflectance: Optional[float] = None


@dataclasses.dataclass
class LightConfig:
    pos: Optional[Sequence[float]] = None
    dir: Optional[Sequence[float]] = None
    diffuse: Optional[Sequence[float]] = None
    ambient: Optional[Sequence[float]] = None
    specular: Optional[Sequence[float]] = None
    active: Optional[bool] = None


@dataclasses.dataclass
class DynamicsConfig:
    damping: Optional[float] = None
    friction: Optional[Sequence[float]] = None
    mass: Optional[float] = None


@dataclasses.dataclass
class DomainRandConfig:
    robot_material: Optional[MaterialConfig] = None
    object_material: Optional[MaterialConfig] = None
    target_material: Optional[MaterialConfig] = None
    distractor1_material: Optional[MaterialConfig] = None
    distractor2_material: Optional[MaterialConfig] = None
    floor_material: Optional[MaterialConfig] = None
    wall_material: Optional[MaterialConfig] = None
    headlight: Optional[LightConfig] = None
    top_light: Optional[LightConfig] = None
    scene_light: Optional[LightConfig] = None
    object_dynamics: Optional[DynamicsConfig] = None


@dataclasses.dataclass
class Ar4Mk3EnvConfig:
    model_path: str
    n_substeps: int = 20
    gripper_extra_height: float = 0.2
    block_gripper: bool = False
    has_object: bool = True
    target_in_the_air: bool = False
    target_offset: tuple[float, float, float] = (0.0, -0.04, 0.02)
    obj_range: tuple[float, float] = (0.09, 0.08)
    obj_offset: tuple[float, float] = (0.0, -0.04)
    target_range: float = 0.13
    distance_threshold: float = 0.05
    reward_type: str = "sparse"
    object_size: tuple[float, float, float] = (0.012, 0.012, 0.012)
    use_eef_control: bool = False
    initial_qpos: dict = field(
        default_factory=lambda: {
            "robot0:slide0": 0.0,
            "robot0:slide1": 0.0,
            "robot0:slide2": 0.0,
        }
    )
    translation: Optional[np.ndarray] = None
    quaterion: Optional[np.ndarray] = None
    z_offset: float = 0.0
    distance_multiplier: float = 1.0
    domain_rand: Optional[DomainRandConfig] = None
    default_camera_config: dict = field(default_factory=lambda: DEFAULT_CAMERA_CONFIG)
