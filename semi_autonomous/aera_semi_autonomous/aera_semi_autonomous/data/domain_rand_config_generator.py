import colorsys
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

# Physically-motivated material classes for each texture in AVAILABLE_TEXTURES.
# Sampling specular/shininess/reflectance per-class (instead of one global range)
# avoids "varnished plaster" or "matte mirror" combinations that read as fake
# even when the albedo is plausible.
_TEXTURE_CLASSES = {
    # Polished metal — sharper highlight, more reflectance.
    "brass-ambra": "metal_polished",
    "metal": "metal_polished",
    # Glazed ceramic — bright highlight.
    "ceramic": "ceramic",
    # Glass — highest spec / reflectance of the set.
    "glass": "glass",
    # Printed packaging (food / drink labels) — semi-gloss print finish.
    "bread": "printed",
    "can": "printed",
    "cereal": "printed",
    "lemon": "printed",
    "soda": "printed",
    # ambientCG matte / semi-matte wood.
    "wood049": "wood", "wood051": "wood", "wood058": "wood",
    "wood066": "wood", "wood067": "wood", "wood092": "wood",
    "wood094": "wood", "wood095": "wood",
    # ambientCG varnished / lacquered wood floor.
    "woodfloor043": "wood_varnished", "woodfloor051": "wood_varnished",
    "woodfloor064": "wood_varnished", "woodfloor070": "wood_varnished",
    # ambientCG brushed / scratched metal.
    "metal046b": "metal_brushed", "metal048a": "metal_brushed",
    "metal049a": "metal_brushed", "metal055a": "metal_brushed",
    "metal061b": "metal_brushed", "metal063": "metal_brushed",
    "metalplates006": "metal_brushed", "metalplates013": "metal_brushed",
    # ambientCG painted / plastered walls.
    "plaster001": "plaster", "plaster002": "plaster",
    "plaster003": "plaster", "plaster007": "plaster",
    "paintedplaster015": "plaster", "paintedplaster016": "plaster",
    "paintedplaster017": "plaster",
    # ambientCG brick + concrete — rough matte.
    "bricks075a": "matte_rough", "bricks097": "matte_rough",
    "bricks101": "matte_rough", "bricks102": "matte_rough",
    "bricks104": "matte_rough",
    "concrete034": "matte_rough", "concrete046": "matte_rough",
    "concrete047a": "matte_rough", "concrete048": "matte_rough",
    # ambientCG tile — slight sheen.
    "tiles040": "tile", "tiles078": "tile", "tiles107": "tile",
    "tiles132a": "tile", "tiles133a": "tile", "tiles138": "tile",
    # ambientCG fabric.
    "fabric030": "fabric", "fabric061": "fabric", "fabric066": "fabric",
    "fabric081c": "fabric", "fabric083": "fabric",
    # ambientCG marble — glazed-ceramic-like.
    "marble012": "ceramic", "marble016": "ceramic", "marble021": "ceramic",
    # ambientCG plastic.
    "plastic006": "plastic", "plastic010": "plastic",
    "plastic012a": "plastic", "plastic013a": "plastic",
    "plastic015a": "plastic",
}

# (specular_range, shininess_range, reflectance_range) per class.
_CLASS_MATERIAL_RANGES = {
    "wood":            ((0.05, 0.20), (0.10, 0.35), (0.00, 0.00)),
    "wood_varnished":  ((0.25, 0.45), (0.35, 0.60), (0.00, 0.05)),
    "metal_brushed":   ((0.35, 0.60), (0.30, 0.55), (0.05, 0.15)),
    "metal_polished":  ((0.50, 0.75), (0.50, 0.75), (0.10, 0.25)),
    "plaster":         ((0.00, 0.10), (0.05, 0.15), (0.00, 0.00)),
    "ceramic":         ((0.40, 0.65), (0.50, 0.75), (0.05, 0.12)),
    "glass":           ((0.60, 0.85), (0.65, 0.85), (0.10, 0.25)),
    "fabric":          ((0.00, 0.08), (0.05, 0.15), (0.00, 0.00)),
    "matte_rough":     ((0.00, 0.10), (0.05, 0.15), (0.00, 0.00)),
    "tile":            ((0.15, 0.35), (0.25, 0.45), (0.00, 0.05)),
    "printed":         ((0.15, 0.40), (0.25, 0.50), (0.00, 0.00)),
    # Covers matte 3D-printed parts at the low end and smooth molded plastic
    # at the high end. No reflectance — plastic is dielectric and the renderer
    # already gets the highlight from specular+shininess.
    "plastic":         ((0.10, 0.45), (0.20, 0.55), (0.00, 0.00)),
}

# Per-class plausible tile density in repeats per world meter. Only meaningful
# for materials whose geom uses texuniform=true (currently floor + walls), so
# the generator only emits texrepeat for those surfaces. Each range encodes the
# physical scale of the pattern: plaster/glass are near-uniform (low repeat),
# bricks/tiles/fabric have small recognizable units (higher repeat).
_CLASS_TEXREPEAT_PER_METER = {
    "wood":            (0.5, 1.2),
    "wood_varnished":  (0.4, 0.8),
    "metal_brushed":   (1.0, 2.5),
    "metal_polished":  (1.0, 2.5),
    "plaster":         (0.3, 0.8),
    "ceramic":         (1.5, 3.0),
    "glass":           (0.3, 0.8),
    "fabric":          (1.5, 3.0),
    "matte_rough":     (1.5, 3.5),
    "tile":            (1.5, 3.5),
    "printed":         (0.8, 2.0),
    # Molded/printed plastic surfaces — features are large, low repeat.
    "plastic":         (0.5, 1.5),
}

assert set(_TEXTURE_CLASSES.keys()) == set(AVAILABLE_TEXTURES), (
    "_TEXTURE_CLASSES is out of sync with AVAILABLE_TEXTURES: "
    f"{set(AVAILABLE_TEXTURES) ^ set(_TEXTURE_CLASSES.keys())}"
)

# Per-class tint policy: (P(apply tint), saturation range, value range) in HSV.
# Hue is always uniform [0, 1). The tint is multiplied against the texture by
# MuJoCo, so the photographic grain/structure is preserved while the dominant
# color shifts — this is what recovers the old hand-picked "red-wood/blue-wood"
# variety from the realistic ambientCG photo set. None disables tinting (the
# texture's own color is the look — bread label, brass, glass, marble).
# Saturation is capped <1 and value is bounded above 0.5 so we don't end up
# with cartoon-bright or near-black materials.
_CLASS_TINT_POLICY = {
    "wood":            (0.6, (0.2, 0.7), (0.5, 1.0)),
    "wood_varnished":  (0.6, (0.2, 0.7), (0.5, 1.0)),
    "plaster":         (0.7, (0.0, 0.8), (0.6, 1.0)),
    "fabric":          (0.7, (0.1, 0.9), (0.5, 1.0)),
    "tile":            (0.5, (0.1, 0.6), (0.6, 1.0)),
    "matte_rough":     (0.4, (0.1, 0.5), (0.5, 1.0)),
    "plastic":         (0.7, (0.2, 0.9), (0.6, 1.0)),
    "metal_brushed":   None,
    "metal_polished":  None,
    "ceramic":         None,
    "glass":           None,
    "printed":         None,
}

assert set(_CLASS_TINT_POLICY.keys()) == set(_CLASS_MATERIAL_RANGES.keys()), (
    "_CLASS_TINT_POLICY out of sync with _CLASS_MATERIAL_RANGES: "
    f"{set(_CLASS_MATERIAL_RANGES.keys()) ^ set(_CLASS_TINT_POLICY.keys())}"
)


def _sample_tint_rgba(cls: str) -> Optional[Tuple[float, float, float, float]]:
    policy = _CLASS_TINT_POLICY.get(cls)
    if policy is None:
        return None
    p_tint, s_range, v_range = policy
    if np.random.random() > p_tint:
        return None
    h = float(np.random.uniform(0.0, 1.0))
    s = float(np.random.uniform(*s_range))
    v = float(np.random.uniform(*v_range))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b), 1.0)


def _sample_material_for_texture(
    texture_name: str, randomize_texrepeat: bool = False
) -> MaterialConfig:
    """Sample specular/shininess/reflectance from the physical class the texture
    belongs to, so e.g. plaster never gets a mirror highlight.

    For tintable classes (wood, plaster, fabric, ...) also samples an rgba tint
    that the runtime multiplies against the texture — this is how we recover
    color variety on top of realistic photo textures.

    If randomize_texrepeat is True, also sample a class-appropriate texrepeat
    (units: repeats per world meter). Only pass True for materials whose geom
    uses texuniform=true — otherwise the value's meaning shifts to per-bbox
    repeats and the sampled scale is wrong.
    """
    cls = _TEXTURE_CLASSES[texture_name]
    spec_r, shin_r, refl_r = _CLASS_MATERIAL_RANGES[cls]
    texrepeat = None
    if randomize_texrepeat:
        rate = float(np.random.uniform(*_CLASS_TEXREPEAT_PER_METER[cls]))
        texrepeat = [rate, rate]
    return MaterialConfig(
        texture_name=texture_name,
        rgba=_sample_tint_rgba(cls),
        specular=float(np.random.uniform(*spec_r)),
        shininess=float(np.random.uniform(*shin_r)),
        reflectance=float(np.random.uniform(*refl_r)),
        texrepeat=texrepeat,
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
    floor_material = _sample_material_for_texture(
        np.random.choice(AVAILABLE_TEXTURES), randomize_texrepeat=True
    )
    wall_material = _sample_material_for_texture(
        np.random.choice(AVAILABLE_TEXTURES), randomize_texrepeat=True
    )

    def _create_random_robot_part_material():
        return _sample_material_for_texture(np.random.choice(AVAILABLE_TEXTURES))

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
