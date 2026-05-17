import logging
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from aera.autonomous.envs.ar4_mk3_config import (
    AVAILABLE_TEXTURES,
    CameraConfig,
    DomainRandConfig,
    DynamicsConfig,
    LightConfig,
    MaterialConfig,
)

# Scene-camera anchor poses captured with
# aera/autonomous/simulation/examples/camera_pose_probe.py at the extreme
# acceptable viewpoints. New samples are drawn from the convex hull of these
# anchors (Dirichlet weights) so every sampled view is a blend of validated
# poses — avoids the corner-of-bounding-box failure mode of per-axis uniform
# sampling.
_SCENE_CAMERA_ANCHORS_POS = np.array(
    [
        [-0.8757, 0.3022, -0.3331],
        [-0.7761, 0.3093, -0.1889],
        [-0.0191, 0.0995, 0.1635],
        [-0.0066, 0.0303, 0.1957],
        [-0.4911, 0.7102, -0.4701],
        [0.5281, 1.0338, -0.0993],
        [0.3697, -0.2488, 0.2982],
        [0.4279, 0.1996, 0.1583],
    ]
)
_SCENE_CAMERA_ANCHORS_EULER = np.array(
    [
        [-0.2365, -0.2919, -0.0304],
        [0.0546, -0.3750, -0.0030],
        [0.0838, -0.0109, 0.2789],
        [-0.0566, 0.1260, 0.2819],
        [-0.8656, 0.1769, 0.4676],
        [-0.4151, -0.5173, 2.4690],
        [0.0944, -0.3660, 2.8664],
        [0.0313, 0.1657, 2.9724],
    ]
)
# Pre-convert to quaternions so we can do proper rotation averaging across
# the z-euler wraparound (anchors 1-5 have rz≈0, anchors 6-8 have rz≈2.5-3.0).
_SCENE_CAMERA_ANCHOR_ROTS = Rotation.from_euler(
    "xyz", _SCENE_CAMERA_ANCHORS_EULER
)


def _sample_scene_camera_pose() -> Tuple[list, list]:
    """Sample a (pos_offset, rot_offset_euler) inside the convex hull of the
    validated anchor poses.

    Position is a Dirichlet-weighted blend of anchor positions; rotation is
    the corresponding weighted quaternion mean (handles wraparound correctly).
    """
    n = len(_SCENE_CAMERA_ANCHORS_POS)
    weights = np.random.dirichlet(np.ones(n))
    pos = (weights[:, None] * _SCENE_CAMERA_ANCHORS_POS).sum(axis=0)
    rot = _SCENE_CAMERA_ANCHOR_ROTS.mean(weights=weights)
    return pos.tolist(), rot.as_euler("xyz").tolist()

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


def _generate_camera_configs(
    randomize_cameras: bool,
) -> Tuple[Optional[CameraConfig], Optional[CameraConfig]]:
    if not randomize_cameras:
        return None, None
    pos_offset, rot_offset_euler = _sample_scene_camera_pose()
    default_camera = CameraConfig(
        pos_offset=pos_offset,
        rot_offset_euler=rot_offset_euler,
    )
    # Minimal wrist-camera perturbation: ~±2mm position, ~±0.5deg rotation.
    gripper_camera = CameraConfig(
        pos_offset=np.random.uniform(-0.005, 0.005, 3).tolist(),
        rot_offset_euler=np.random.uniform(-0.01, 0.01, 3).tolist(),
    )
    return default_camera, gripper_camera


def generate_random_domain_rand_config(
    randomize_cameras: bool = False,
) -> Tuple[DomainRandConfig, str, str]:
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
    logging.getLogger(__name__).debug(
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
    target_rgba = (target_rgba[0], target_rgba[1], target_rgba[2], 0.9)
    target_distractor1_rgba = (
        target_distractor1_rgba[0],
        target_distractor1_rgba[1],
        target_distractor1_rgba[2],
        0.9,
    )
    target_distractor2_rgba = (
        target_distractor2_rgba[0],
        target_distractor2_rgba[1],
        target_distractor2_rgba[2],
        0.9,
    )

    # --- Material Randomization ---
    object_material = MaterialConfig(
        rgba=object_rgba,
    )
    target_material = MaterialConfig(rgba=target_rgba)
    distractor1_material = MaterialConfig(rgba=target_distractor1_rgba)
    distractor2_material = MaterialConfig(rgba=target_distractor2_rgba)
    object_distractor1_material = MaterialConfig(
        rgba=object_distractor1_rgba,
    )
    object_distractor2_material = MaterialConfig(
        rgba=object_distractor2_rgba,
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

    # --- Camera Randomization (opt-in) ---
    default_camera, gripper_camera = _generate_camera_configs(randomize_cameras)

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
        default_camera=default_camera,
        gripper_camera=gripper_camera,
    )

    return domain_rand_config, object_color_name, target_color_name
