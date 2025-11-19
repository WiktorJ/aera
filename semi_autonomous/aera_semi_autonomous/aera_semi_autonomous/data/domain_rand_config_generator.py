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
    (
        object_color_name,
        target_color_name,
        object_distractor1_color_name,
        object_distractor2_color_name,
        target_distractor1_color_name,
        target_distractor2_color_name,
    ) = np.random.choice(list(NAMED_COLORS.keys()), 6, replace=False)
    print(
        f"Object color: {object_color_name}\n"
        f"Target color: {target_color_name}\n"
        f"Object distractor 1 color: {object_distractor1_color_name}\n"
        f"Object distractor 2 color: {object_distractor2_color_name}\n"
        f"Target distractor 1 color: {target_distractor1_color_name}\n"
        f"Target distractor 2 color: {target_distractor2_color_name}"
    )

    object_rgba = NAMED_COLORS[object_color_name]
    target_rgba = NAMED_COLORS[target_color_name]
    object_distractor1_rgba = NAMED_COLORS[object_distractor1_color_name]
    object_distractor2_rgba = NAMED_COLORS[object_distractor2_color_name]
    target_distractor1_rgba = NAMED_COLORS[target_distractor1_color_name]
    target_distractor2_rgba = NAMED_COLORS[target_distractor2_color_name]

    # Make target and its distractors semi-transparent
    target_rgba = (target_rgba[0], target_rgba[1], target_rgba[2], 0.5)
    target_distractor1_rgba = (
        target_distractor1_rgba[0],
        target_distractor1_rgba[1],
        target_distractor1_rgba[2],
        0.5,
    )
    target_distractor2_rgba = (
        target_distractor2_rgba[0],
        target_distractor2_rgba[1],
        target_distractor2_rgba[2],
        0.5,
    )

    # --- Material Randomization ---
    object_material = MaterialConfig(
        texture_name=np.random.choice(AVAILABLE_TEXTURES),
        rgba=object_rgba,
        specular=np.random.uniform(0.5, 1.0),
        shininess=np.random.uniform(0.5, 1.0),
        reflectance=np.random.uniform(0.0, 0.2),
    )
    target_material = MaterialConfig(rgba=target_rgba)
    distractor1_material = MaterialConfig(rgba=target_distractor1_rgba)
    distractor2_material = MaterialConfig(rgba=target_distractor2_rgba)
    object_distractor1_material = MaterialConfig(
        texture_name=np.random.choice(AVAILABLE_TEXTURES),
        rgba=object_distractor1_rgba,
        specular=np.random.uniform(0.5, 1.0),
        shininess=np.random.uniform(0.5, 1.0),
        reflectance=np.random.uniform(0.0, 0.2),
    )
    object_distractor2_material = MaterialConfig(
        texture_name=np.random.choice(AVAILABLE_TEXTURES),
        rgba=object_distractor2_rgba,
        specular=np.random.uniform(0.5, 1.0),
        shininess=np.random.uniform(0.5, 1.0),
        reflectance=np.random.uniform(0.0, 0.2),
    )
    floor_material = MaterialConfig(
        texture_name=np.random.choice(AVAILABLE_TEXTURES),
        specular=np.random.uniform(0.1, 0.8),
        shininess=np.random.uniform(0.1, 0.7),
    )
    wall_material = MaterialConfig(texture_name=np.random.choice(AVAILABLE_TEXTURES))

    def _create_random_robot_part_material():
        return MaterialConfig(
            texture_name=np.random.choice(AVAILABLE_TEXTURES),
            specular=np.random.uniform(0.1, 0.8),
            shininess=np.random.uniform(0.1, 0.7),
        )

    base_link_material = _create_random_robot_part_material()
    link_1_material = _create_random_robot_part_material()
    link_2_material = _create_random_robot_part_material()
    link_3_material = _create_random_robot_part_material()
    link_4_material = _create_random_robot_part_material()
    link_5_material = _create_random_robot_part_material()
    link_6_material = _create_random_robot_part_material()
    gripper_base_link_material = _create_random_robot_part_material()
    gripper_jaw1_material = _create_random_robot_part_material()
    gripper_jaw2_material = _create_random_robot_part_material()

    # --- Light Randomization ---
    headlight = LightConfig(
        diffuse=np.random.uniform(0.4, 0.6, 3).tolist(),
        ambient=np.random.uniform(0.1, 0.2, 3).tolist(),
        specular=np.random.uniform(0.2, 0.4, 3).tolist(),
    )
    scene_light = LightConfig(
        active=True,
        pos=np.random.uniform([-1, -1, 2.5], [1, 1, 3.5]).tolist(),
        dir=np.random.uniform([-0.5, -0.5, -1.0], [0.5, 0.5, -0.8]).tolist(),
        diffuse=np.random.uniform(0.5, 0.7, 3).tolist(),
        ambient=np.random.uniform(0.2, 0.4, 3).tolist(),
        specular=np.random.uniform(0.5, 0.7, 3).tolist(),
    )
    top_light = LightConfig(
        active=True,
        pos=np.random.uniform([-1, -1, 1.5], [1, 1, 2.5]).tolist(),
        dir=np.random.uniform([-0.5, -0.5, -1.0], [0.5, 0.5, -0.8]).tolist(),
        diffuse=np.random.uniform(0.5, 0.7, 3).tolist(),
        ambient=np.random.uniform(0.2, 0.4, 3).tolist(),
        specular=np.random.uniform(0.5, 0.7, 3).tolist(),
    )

    # --- Dynamics Randomization ---
    def _create_random_dynamics_config():
        return DynamicsConfig(
            size=np.random.uniform([0.01, 0.01, 0.01], [0.015, 0.015, 0.015]).tolist(),
            mass=np.random.uniform(0.05, 0.15),
            friction=np.random.uniform(
                [1.5, 0.005, 0.005], [2.5, 0.015, 0.015]
            ).tolist(),
            damping=np.random.uniform(0.005, 0.015),
        )

    object_dynamics = _create_random_dynamics_config()
    object_distractor1_dynamics = _create_random_dynamics_config()
    object_distractor2_dynamics = _create_random_dynamics_config()

    domain_rand_config = DomainRandConfig(
        object_material=object_material,
        target_material=target_material,
        distractor1_material=distractor1_material,
        distractor2_material=distractor2_material,
        object_distractor1_material=object_distractor1_material,
        object_distractor2_material=object_distractor2_material,
        floor_material=floor_material,
        wall_material=wall_material,
        base_link_material=base_link_material,
        link_1_material=link_1_material,
        link_2_material=link_2_material,
        link_3_material=link_3_material,
        link_4_material=link_4_material,
        link_5_material=link_5_material,
        link_6_material=link_6_material,
        gripper_base_link_material=gripper_base_link_material,
        gripper_jaw1_material=gripper_jaw1_material,
        gripper_jaw2_material=gripper_jaw2_material,
        headlight=headlight,
        top_light=top_light,
        scene_light=scene_light,
        object_dynamics=object_dynamics,
        object_distractor1_dynamics=object_distractor1_dynamics,
        object_distractor2_dynamics=object_distractor2_dynamics,
    )

    return domain_rand_config, object_color_name, target_color_name
