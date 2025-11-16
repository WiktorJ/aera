import dataclasses
from dataclasses import field
from typing import Optional, Sequence, Union

import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "distance": 0.98,
    "azimuth": -133,
    "elevation": -26,
    "lookat": np.array([0, 0, 0]),
}

T = np.array([0.6233588611899381, 0.05979687559388906, 0.7537742046170788])
Q = np.array(
    [
        -0.36336720179946663,
        -0.8203835174702869,
        0.22865474664402222,
        0.37769321910336584,
    ]
)

AVAILABLE_TEXTURES = [
    "blue-wood",
    "brass-ambra",
    "bread",
    "can",
    "ceramic",
    "cereal",
    "clay",
    "cream-plaster",
    "dark-wood",
    "dirt",
    "glass",
    "gray-felt",
    "gray-plaster",
    "gray-woodgrain",
    "green-wood",
    "lemon",
    "light-gray-floor-tile",
    "light-gray-plaster",
    "light-wood",
    "metal",
    "pink-plaster",
    "red-wood",
    "soda",
    "steel-brushed",
    "steel-scratched",
    "white-bricks",
    "white-plaster",
    "wood-tiles",
    "wood-varnished-panels",
    "yellow-plaster",
]


@dataclasses.dataclass
class MaterialConfig:
    texture_name: Optional[Union[str, Sequence[str]]] = None
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
    size: Optional[Sequence[float]] = None


@dataclasses.dataclass
class DomainRandConfig:
    object_material: Optional[MaterialConfig] = None
    target_material: Optional[MaterialConfig] = None
    distractor1_material: Optional[MaterialConfig] = None
    distractor2_material: Optional[MaterialConfig] = None
    object_distractor1_material: Optional[MaterialConfig] = None
    object_distractor2_material: Optional[MaterialConfig] = None
    floor_material: Optional[MaterialConfig] = None
    wall_material: Optional[MaterialConfig] = None
    base_link_material: Optional[MaterialConfig] = None
    link_1_material: Optional[MaterialConfig] = None
    link_2_material: Optional[MaterialConfig] = None
    link_3_material: Optional[MaterialConfig] = None
    link_4_material: Optional[MaterialConfig] = None
    link_5_material: Optional[MaterialConfig] = None
    link_6_material: Optional[MaterialConfig] = None
    gripper_base_link_material: Optional[MaterialConfig] = None
    gripper_jaw1_material: Optional[MaterialConfig] = None
    gripper_jaw2_material: Optional[MaterialConfig] = None
    headlight: Optional[LightConfig] = None
    top_light: Optional[LightConfig] = None
    scene_light: Optional[LightConfig] = None
    object_dynamics: Optional[DynamicsConfig] = None
    object_distractor1_dynamics: Optional[DynamicsConfig] = None
    object_distractor2_dynamics: Optional[DynamicsConfig] = None


@dataclasses.dataclass
class Ar4Mk3EnvConfig:
    model_path: str
    n_substeps: int = 20
    gripper_extra_height: float = 0.2
    block_gripper: bool = False
    has_object: bool = True
    target_in_the_air: bool = False
    absolute_state_actions: bool = False
    target_offset: tuple[float, float, float] = (0.0, -0.04, 0.03)
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
    translation: Optional[np.ndarray] = field(default_factory=lambda: T)
    quaterion: Optional[np.ndarray] = field(default_factory=lambda: Q)
    z_offset: float = 0.3
    distance_multiplier: float = 1.2
    domain_rand: Optional[DomainRandConfig] = None
    default_camera_config: dict = field(default_factory=lambda: DEFAULT_CAMERA_CONFIG)
    image_width: int = 224
    image_height: int = 224
