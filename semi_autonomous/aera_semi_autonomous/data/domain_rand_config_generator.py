import random
from typing import Tuple

import numpy as np

from aera.autonomous.envs.ar4_mk3_config import (
    AVAILABLE_TEXTURES,
    DomainRandConfig,
    DynamicsConfig,
    LightConfig,
    MaterialConfig,
)

NAMED_COLORS = {
    "red": (1, 0, 0, 1),
    "green": (0, 1, 0, 1),
    "blue": (0, 0, 1, 1),
    "yellow": (1, 1, 0, 1),
    "cyan": (0, 1, 1, 1),
    "magenta": (1, 0, 1, 1),
    "white": (1, 1, 1, 1),
    "black": (0, 0, 0, 1),
    "gray": (0.5, 0.5, 0.5, 1),
}


def generate_random_domain_rand_config() -> Tuple[DomainRandConfig, str, str]:
    """
    Generates a randomized DomainRandConfig for the AR4 MK3 environment.

    Returns:
        A tuple containing:
        - The generated DomainRandConfig.
        - The name of the color used for the object.
        - The name of the color used for the target.
    """
    # --- Color Selection ---
    object_color_name, target_color_name = random.sample(list(NAMED_COLORS.keys()), 2)
    object_rgba = NAMED_COLORS[object_color_name]
    target_rgba = NAMED_COLORS[target_color_name]
    # Make target semi-transparent
    target_rgba = (target_rgba[0], target_rgba[1], target_rgba[2], 0.5)

    # --- Material Randomization ---
    object_material = MaterialConfig(
        texture_name=random.choice(AVAILABLE_TEXTURES),
        rgba=object_rgba,
        specular=random.uniform(0.5, 1.0),
        shininess=random.uniform(0.5, 1.0),
        reflectance=random.uniform(0.0, 0.2),
    )
    target_material = MaterialConfig(rgba=target_rgba)
    floor_material = MaterialConfig(
        texture_name=random.choice(AVAILABLE_TEXTURES),
        specular=random.uniform(0.1, 0.8),
        shininess=random.uniform(0.1, 0.7),
    )
    wall_material = MaterialConfig(texture_name=random.choice(AVAILABLE_TEXTURES))

    # --- Light Randomization ---
    headlight = LightConfig(
        diffuse=np.random.uniform(0.6, 0.8, 3).tolist(),
        ambient=np.random.uniform(0.1, 0.3, 3).tolist(),
        specular=np.random.uniform(0.4, 0.6, 3).tolist(),
    )
    scene_light = LightConfig(
        active=True,
        pos=np.random.uniform([-1, -1, 2.5], [1, 1, 3.5]).tolist(),
        dir=np.random.uniform([-0.5, -0.5, -1.0], [0.5, 0.5, -0.8]).tolist(),
        diffuse=np.random.uniform(0.8, 1.0, 3).tolist(),
        ambient=np.random.uniform(0.3, 0.5, 3).tolist(),
        specular=np.random.uniform(0.8, 1.0, 3).tolist(),
    )

    # --- Dynamics Randomization ---
    object_dynamics = DynamicsConfig(
        size=np.random.uniform([0.01, 0.01, 0.01], [0.015, 0.015, 0.015]).tolist(),
        mass=random.uniform(0.05, 0.15),
        friction=np.random.uniform([1.5, 0.005, 0.005], [2.5, 0.015, 0.015]).tolist(),
        damping=random.uniform(0.005, 0.015),
    )

    domain_rand_config = DomainRandConfig(
        object_material=object_material,
        target_material=target_material,
        floor_material=floor_material,
        wall_material=wall_material,
        headlight=headlight,
        scene_light=scene_light,
        object_dynamics=object_dynamics,
    )

    return domain_rand_config, object_color_name, target_color_name
