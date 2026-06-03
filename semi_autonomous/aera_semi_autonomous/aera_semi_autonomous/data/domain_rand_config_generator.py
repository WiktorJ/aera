import colorsys
import json
import logging
from pathlib import Path
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
    PropConfig,
    TableConfig,
    WallArtConfig,
)

# Fixed table height matches scene.xml: room floor at z=-0.75, work surface
# (top of `floor` geom) at z=0. Randomizing the height would require moving
# the arm base, so kept constant in v1.
_TABLE_HEIGHT = 0.75

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


# Color-temperature buckets for light tinting. Mixing per-channel uniforms (the
# old approach) collapses to near-achromatic; sampling one HSV tint and reusing
# it across diffuse/ambient/specular keeps the warm/cool character coherent.
# Saturation is kept low — real bulbs are off-white, not orange/blue.
_LIGHT_TEMP_BUCKETS = (
    (0.40, (0.05, 0.12), (0.05, 0.30)),  # warm (tungsten-ish)
    (0.40, (0.55, 0.65), (0.05, 0.30)),  # cool (daylight-ish)
    (0.20, (0.00, 0.00), (0.00, 0.00)),  # neutral
)


def _sample_light_tint() -> Tuple[float, float]:
    """Pick a (hue, saturation) for one light. Value is set per-channel later."""
    r = np.random.random()
    cum = 0.0
    for p, h_range, s_range in _LIGHT_TEMP_BUCKETS:
        cum += p
        if r < cum:
            return (
                float(np.random.uniform(*h_range)),
                float(np.random.uniform(*s_range)),
            )
    return 0.0, 0.0


def _tinted_rgb(h: float, s: float, v: float) -> list:
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return [float(r), float(g), float(b)]


# Per-scene lighting "mood" — sampled once and applied coherently to all lights.
# The dominant axis here is the diffuse-to-ambient ratio: high ambient washes
# shadows out, low ambient + strong diffuse gives crisp directional shadows.
# Fields are: (probability, diffuse_range, ambient_range, specular_range,
#              intensity_scale_range, p_active_aux).
# - diffuse can exceed 1.0 in MuJoCo; we let it go to 1.4 for overexposed looks.
# - p_active_aux is applied to the two positional lights so some scenes are
#   lit by just the headlight + one aux for stronger key-light shadows.
_LIGHT_MOODS = (
    # "studio": bright, even, soft shadows. Flat product-shot look.
    (0.30, (0.9, 1.3), (0.20, 0.40), (0.30, 0.60), (1.0, 1.3), 0.95),
    # "key": single strong directional, very low ambient — dramatic shadows.
    (0.35, (1.0, 1.5), (0.02, 0.10), (0.50, 1.00), (1.0, 1.4), 0.65),
    # "overcast": diffuse but not bright, moderate ambient, weak spec.
    (0.15, (0.6, 0.9), (0.15, 0.30), (0.10, 0.30), (0.8, 1.1), 0.90),
    # "dim": challenging low-light. Forces the policy to handle bad exposure.
    (0.05, (0.3, 0.6), (0.02, 0.12), (0.10, 0.40), (0.6, 1.0), 0.80),
    # "harsh": very bright key + very low fill — sharp shadows, blown highlights.
    (0.15, (1.2, 1.5), (0.01, 0.06), (0.60, 1.00), (1.1, 1.5), 0.50),
)


def _sample_lighting_mood():
    """Returns (diffuse_range, ambient_range, specular_range, scale_range,
    p_active_aux) for the whole scene. Single sample shared by all lights."""
    r = np.random.random()
    cum = 0.0
    for p, *cfg in _LIGHT_MOODS:
        cum += p
        if r < cum:
            return cfg
    return list(_LIGHT_MOODS[0][1:])


def _sample_headlight(mood) -> LightConfig:
    """MuJoCo's headlight has no position — it's tied to the camera. Tint it
    and pick an overall intensity from the scene mood. Headlight ambient is
    deliberately scaled down vs the positional lights so it doesn't flood out
    the directional shadows from scene_light."""
    diff_r, amb_r, spec_r, scale_r, _ = mood
    h, s = _sample_light_tint()
    scale = float(np.random.uniform(*scale_r))
    return LightConfig(
        diffuse=_tinted_rgb(h, s, np.clip(np.random.uniform(*diff_r) * scale * 0.9, 0, 1.5)),
        ambient=_tinted_rgb(h, s, np.clip(np.random.uniform(*amb_r) * scale * 0.6, 0, 1)),
        specular=_tinted_rgb(h, s, np.clip(np.random.uniform(*spec_r) * scale * 0.7, 0, 1)),
    )


def _sample_positional_light(
    mood,
    table_center: Tuple[float, float],
    distance_range: Tuple[float, float],
    elevation_range_deg: Tuple[float, float],
) -> LightConfig:
    """Sample a scene/top light with full spherical positioning around the table.

    Position is `table_center + distance * (cos(el)cos(az), cos(el)sin(az), sin(el))`
    where elevation is sampled directly in degrees from the horizon. This lets
    us actually produce low-angle lights that cast long shadows — the previous
    z+radius scheme was z-dominated (z 2.5–3.5 over radius 0.6–1.6 ⇒ elevation
    always ~65–80°), so shadows were always nearly vertical regardless of az.

    Direction is computed to point at a jittered point near the table so the
    directional light's shadow angle matches the sampled elevation/azimuth.

    Ranges come from the shared lighting mood so all lights in the scene agree
    on whether it's a bright studio or a moody key-lit shot.
    """
    diff_r, amb_r, spec_r, scale_r, p_active = mood
    h, s = _sample_light_tint()
    scale = float(np.random.uniform(*scale_r))
    az = float(np.random.uniform(0.0, 2.0 * np.pi))
    # Bias elevation low: sample a Beta(1.6, 3.0) and map onto the range. Mean
    # sits ~1/3 of the way up, so most samples are low-angle (long arm shadows)
    # while we still see the occasional overhead noon-style sample.
    t = float(np.random.beta(1.6, 3.0))
    el_lo, el_hi = elevation_range_deg
    el = (el_lo + t * (el_hi - el_lo)) * np.pi / 180.0
    distance = float(np.random.uniform(*distance_range))
    pos = [
        float(table_center[0] + distance * np.cos(el) * np.cos(az)),
        float(table_center[1] + distance * np.cos(el) * np.sin(az)),
        float(distance * np.sin(el)),
    ]
    # Aim at the arm's working volume (above the table, z≈0.3) so the directional
    # shadow frustum covers the arm — not just the table surface where shadows
    # would be hidden under the arm's silhouette.
    target = np.array(
        [
            table_center[0] + np.random.uniform(-0.10, 0.10),
            table_center[1] + np.random.uniform(-0.10, 0.10),
            float(np.random.uniform(0.20, 0.40)),
        ]
    )
    direction = target - np.array(pos)
    direction = (direction / np.linalg.norm(direction)).tolist()
    return LightConfig(
        active=bool(np.random.random() < p_active),
        pos=pos,
        dir=direction,
        diffuse=_tinted_rgb(h, s, np.clip(np.random.uniform(*diff_r) * scale, 0, 1.5)),
        ambient=_tinted_rgb(h, s, np.clip(np.random.uniform(*amb_r) * scale, 0, 1)),
        specular=_tinted_rgb(h, s, np.clip(np.random.uniform(*spec_r) * scale, 0, 1)),
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


# --- PLA-printed object / target appearance --------------------------------
#
# The policy is deployed on real 3D-PLA-printed blocks and printed target
# plates. Rendering them as flat pure-primary slabs (the old behavior) is the
# largest sim-real gap left in the scene — everything else already goes through
# the textured material pipeline. These helpers give the blocks/plates a
# filament-realistic color, an FDM layer-line texture, and a plastic highlight,
# with a wide distribution around the PLA look for robustness.

# Grayscale layer-line textures, declared in scene.xml and generated by
# scripts/generate_pla_textures.py. Tinted per-episode via rgba-multiply, so one
# texture covers every filament color. Kept out of AVAILABLE_TEXTURES on purpose
# — these are object-only and must never be picked for floors/walls/robot links.
PLA_LINE_TEXTURES = ("pla_lines_fine", "pla_lines_medium", "pla_lines_coarse")

# HSV hue centers for the chromatic named colors. The *name* must survive because
# it's injected into the policy's language prompt ("pick the {color} block"); we
# only make the rendered rgba look like filament, not the label. Blue is pulled a
# hair off pure-(0,0,1) violet toward a more typical filament blue.
_PLA_HUE = {
    "red": 0.00, "yellow": 0.13, "green": 0.33,
    "cyan": 0.50, "blue": 0.62, "magenta": 0.83,
}
# Near-neutral filaments don't have a meaningful hue — drive them by value only.
# (saturation_range, value_range).
_PLA_NEUTRAL = {
    "white": ((0.0, 0.05), (0.85, 0.97)),
    "gray":  ((0.0, 0.08), (0.42, 0.58)),
    "black": ((0.0, 0.10), (0.06, 0.16)),
}


def _pla_filament_rgba(color_name: str) -> Tuple[float, float, float, float]:
    """Map a NAMED_COLORS label to a 3D-PLA-filament-plausible opaque rgba.

    Real PLA is never a pure (1,0,0)-style primary: it's saturated-but-not-maxed
    and slightly dark. We jitter S/V per episode (and the hue a hair) so repeated
    'red' blocks aren't identical, while staying well inside the color's name so
    the language prompt stays truthful.
    """
    if color_name in _PLA_NEUTRAL:
        s_range, v_range = _PLA_NEUTRAL[color_name]
        h = float(np.random.uniform(0.0, 1.0))  # irrelevant at ~0 saturation
        s = float(np.random.uniform(*s_range))
        v = float(np.random.uniform(*v_range))
    else:
        h = (_PLA_HUE[color_name] + float(np.random.uniform(-0.02, 0.02))) % 1.0
        s = float(np.random.uniform(0.65, 0.95))
        v = float(np.random.uniform(0.55, 0.85))
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return (float(r), float(g), float(b), 1.0)


def _sample_pla_object_material(
    color_name: str, is_plate: bool = False
) -> MaterialConfig:
    """Material for a printed block (is_plate=False) or printed target plate.

    Wide DR around the PLA look: a filament-colored base, a layer-line texture
    applied most (but not all) of the time so the policy also sees clean prints,
    and a matte->semi-gloss highlight. The grayscale texture is tinted by `rgba`
    via MuJoCo's texture*rgba multiply, so one texture serves every color. Plates
    run a touch more matte with a denser line pitch (bigger, flatter surface than
    the ~24mm cubes).
    """
    rgba = _pla_filament_rgba(color_name)
    if is_plate:
        spec_r, shin_r, rep_r = (0.05, 0.35), (0.10, 0.45), (1.5, 3.0)
    else:
        spec_r, shin_r, rep_r = (0.08, 0.55), (0.15, 0.60), (1.0, 2.2)
    texture_name = None
    texrepeat = None
    if np.random.random() < 0.65:
        texture_name = str(np.random.choice(PLA_LINE_TEXTURES))
        rate = float(np.random.uniform(*rep_r))
        texrepeat = [rate, rate]
    # Occasional faint sheen from a glossier filament or polished plate. PLA is a
    # dielectric so reflectance stays low even at the top of the range.
    reflectance = (
        float(np.random.uniform(0.0, 0.05)) if np.random.random() < 0.15 else 0.0
    )
    return MaterialConfig(
        texture_name=texture_name,
        rgba=rgba,
        specular=float(np.random.uniform(*spec_r)),
        shininess=float(np.random.uniform(*shin_r)),
        reflectance=reflectance,
        texrepeat=texrepeat,
    )


# --- Static prop / clutter sampling -----------------------------------------
#
# Slot pool + asset registry are declared in
# aera/autonomous/simulation/mujoco/ar4_mk3/props.xml (generated by
# scripts/generate_props_xml.py). The manifest below describes every asset in
# the pool. The runtime mutates each slot's body_pos / body_quat / geom_dataid
# / geom_matid / geom_rgba to realize a PropConfig.

NUM_PROP_SLOTS = 30
PROP_COUNT_RANGE = (20, 30)  # inclusive both ends — randint upper is +1
PROP_THEMES = ("workshop", "office", "lab", "kitchen")
# Per-slot zone weights. Shelf dominates because the shelf unit is always in
# the camera frame and reads as background context; floor is kept low because
# even one floor stool fills a large chunk of the frame, and the (-x, -y)
# quadrant saturates visually after 1-2 placements.
P_SHELF_ZONE = 0.70
P_TABLE_ZONE = 0.20
P_FLOOR_ZONE = 0.10
# Largest object that's allowed on the table — anything taller would loom over
# the workspace and bias the policy toward false-positive object detection.
TABLE_MAX_DIM = 0.20
# Cap for shelf items. Set higher than TABLE_MAX_DIM because shelf clutter is
# background dressing — a slightly tall bottle looks fine on a shelf level,
# but anything bigger risks clipping into the next level. Bumping this from
# 0.20 doubled the per-theme shelf pool, which the new higher slot count needs
# to avoid running out of unique assets and leaving slots inactive.
SHELF_MAX_DIM = 0.25
# Floor is reserved for furniture-scale items (stools, tiered shelves, large
# dish racks, big flower vases). The current pool packs around two size bands:
# kitchenware tops out near 0.40 m (pots, baskets, mug trees, small trays,
# utensil racks) and reads as scattered debris on the floor; furniture starts
# at ~0.495 m (tiered shelves) and goes up to ~0.94 m (tall stools). 0.49 sits
# in the gap and cleanly separates the two.
FLOOR_MIN_DIM = 0.49
# Background shelf geometry — these MUST match the geoms in scene.xml. The
# shelf unit sits against the y=-2 wall, shifted toward the -x corner where
# the render camera's principal ray hits. Slab tops are 0.015m above the
# slab centers in the XML.
SHELF_X_RANGE = (-1.66, -0.74)  # 0.04m inset from the side panels
SHELF_Y_RANGE = (-1.96, -1.74)  # within the 0.26m shelf depth
SHELF_LEVEL_TOPS = (-0.385, 0.115, 0.615)  # top surface z of each level
# Floor placement is restricted to the (-x, -y) quadrant — that's where the
# render camera's view cone touches the ground. Sampling the full floor
# scattered large items behind the camera where they're never seen.
FLOOR_X_RANGE = (-1.80, -0.50)
FLOOR_Y_RANGE = (-1.80, 0.00)
# Active manipulation workspace on the table — derived from the spawn ranges in
# generate_random_domain_rand_config (x∈[-0.14,0.12], y∈[-0.54,-0.24]) with a
# safety margin so the prop's bounding box never reaches across the boundary.
# Props are excluded if their footprint OVERLAPS this rect (not just their
# center), so the buffer compounds with each prop's xy-radius.
_WORKSPACE_X = (-0.14, 0.12)
_WORKSPACE_Y = (-0.54, -0.24)
_WORKSPACE_MARGIN = 0.10  # extra clearance beyond geometric exclusion

_PROPS_DIR = (
    Path(__file__).resolve().parents[4]
    / "aera" / "autonomous" / "simulation" / "props"
)
_MANIFEST_PATH = _PROPS_DIR / "_manifest.json"
# Sidecar emitted by scripts/generate_props_xml.py listing only the assets it
# actually compiled into props.xml. Without this filter the sampler would draw
# from the full 400+ manifest and most picks would hit a mesh the model never
# loaded, leaving the slot invisible.
_SCENE_ASSETS_PATH = _PROPS_DIR / "_scene_assets.json"


def _load_prop_manifest() -> dict[str, list[dict]]:
    """Read the prop manifest once and index entries by theme, restricted to
    assets that the current props.xml actually declares.

    Falls back to an empty pool if either the manifest or the scene-assets
    sidecar is missing — in that case the sampler returns all-inactive
    PropConfigs and the scene looks the way it did before this feature landed."""
    log = logging.getLogger(__name__)
    if not _MANIFEST_PATH.exists():
        log.warning(
            "Prop manifest not found at %s — prop slots will stay hidden. "
            "Run scripts/fetch_props.py && scripts/normalize_props.py to enable.",
            _MANIFEST_PATH,
        )
        return {t: [] for t in PROP_THEMES}
    raw = json.loads(_MANIFEST_PATH.read_text())

    allowed_ids: Optional[set[str]] = None
    compiled_aabbs: dict[str, dict] = {}
    if _SCENE_ASSETS_PATH.exists():
        sidecar = json.loads(_SCENE_ASSETS_PATH.read_text())
        allowed_ids = set(sidecar["asset_ids"])
        compiled_aabbs = sidecar.get("compiled_aabbs", {})
    else:
        log.warning(
            "Scene-assets sidecar not found at %s — sampler will draw from the "
            "full manifest using on-disk OBJ AABBs (props will float because "
            "MuJoCo recenters meshes on compile). Run "
            "scripts/generate_props_xml.py to refresh.",
            _SCENE_ASSETS_PATH,
        )

    by_theme: dict[str, list[dict]] = {t: [] for t in PROP_THEMES}
    for entry in raw["props"]:
        if allowed_ids is not None and entry["id"] not in allowed_ids:
            continue
        # Replace the manifest's on-disk AABB with the post-compile AABB so
        # the sampler's "sit on the surface" math matches what's actually
        # rendered. Falls back to manifest values if the sidecar is stale.
        if entry["id"] in compiled_aabbs:
            ca = compiled_aabbs[entry["id"]]
            entry = {
                **entry,
                "aabb_min": ca["aabb_min"],
                "aabb_max": ca["aabb_max"],
                "size": [
                    ca["aabb_max"][i] - ca["aabb_min"][i] for i in range(3)
                ],
            }
        for theme in entry.get("themes", []):
            if theme in by_theme:
                by_theme[theme].append(entry)
    return by_theme


_PROPS_BY_THEME = _load_prop_manifest()


def _sample_prop_pose(
    asset: dict,
    zone: str,
    table_center: Tuple[float, float],
    table_half_size: Tuple[float, float],
    occupied: list[Tuple[float, float, float, float]],
) -> Optional[Tuple[float, float, float]]:
    """Reject-sample a (x, y, z) for `asset` inside `zone`.

    Returns None if no non-overlapping placement found in `_MAX_TRIES`. The
    caller treats that as "skip this slot" so we never silently stack two
    props on top of each other.

    z is computed so the asset's mesh bottom (aabb_min[2]) rests on the
    surface — floor at z=-0.75, table top at z=0, shelf at SHELF_LEVEL_TOPS.

    Overlap is tracked in 3D: occupied entries are (x, y, z, radius) and a
    candidate only collides with previous placements at the same height
    (|dz| < _Z_OVERLAP). Without this, items on different shelf levels would
    spuriously block each other.
    """
    _MAX_TRIES = 25
    _Z_OVERLAP = 0.05
    # Approximate the asset by a circle in the XY plane for overlap checks —
    # cheap, conservative, and good enough for visual clutter (the alternative,
    # rotated-AABB intersection, would buy us 5% denser packing for 20x code).
    radius = max(asset["size"][0], asset["size"][1]) / 2 + 0.05
    aabb_min_z = asset["aabb_min"][2]
    asset_hx = asset["size"][0] / 2
    asset_hy = asset["size"][1] / 2

    for _ in range(_MAX_TRIES):
        if zone == "floor":
            x = float(np.random.uniform(*FLOOR_X_RANGE))
            y = float(np.random.uniform(*FLOOR_Y_RANGE))
            # Carve out the table footprint + arm reach so floor clutter
            # doesn't intersect the work area or block the arm's view of
            # its own workspace.
            if -1.0 <= x <= 1.0 and -1.0 <= y <= 0.5:
                continue
            # Carve out the shelf footprint (against the y=-2 wall, -x side)
            # so floor furniture doesn't clip into the shelf side panels.
            if -1.75 <= x <= -0.65 and y <= -1.55:
                continue
            z = -0.75 - aabb_min_z
        elif zone == "shelf":
            level_z = float(SHELF_LEVEL_TOPS[
                np.random.randint(len(SHELF_LEVEL_TOPS))
            ])
            margin = 0.03
            x = float(np.random.uniform(
                SHELF_X_RANGE[0] + margin + asset_hx,
                SHELF_X_RANGE[1] - margin - asset_hx,
            ))
            y = float(np.random.uniform(
                SHELF_Y_RANGE[0] + margin + asset_hy,
                SHELF_Y_RANGE[1] - margin - asset_hy,
            ))
            z = level_z - aabb_min_z
        else:  # table
            cx, cy = table_center
            hx, hy = table_half_size
            margin = 0.05
            x = float(np.random.uniform(cx - hx + margin, cx + hx - margin))
            y = float(np.random.uniform(cy - hy + margin, cy + hy - margin))
            # Workspace exclusion: reject if the prop's *footprint* (center ±
            # half-size) overlaps the active-arm rect — not just the center —
            # plus a fixed safety margin. With center-only the corners of
            # large props could still poke into the grasp area.
            ex_x_lo = _WORKSPACE_X[0] - asset_hx - _WORKSPACE_MARGIN
            ex_x_hi = _WORKSPACE_X[1] + asset_hx + _WORKSPACE_MARGIN
            ex_y_lo = _WORKSPACE_Y[0] - asset_hy - _WORKSPACE_MARGIN
            ex_y_hi = _WORKSPACE_Y[1] + asset_hy + _WORKSPACE_MARGIN
            if ex_x_lo <= x <= ex_x_hi and ex_y_lo <= y <= ex_y_hi:
                continue
            z = 0.0 - aabb_min_z

        too_close = False
        for ox, oy, oz, orad in occupied:
            if abs(z - oz) > _Z_OVERLAP:
                continue
            if (x - ox) ** 2 + (y - oy) ** 2 < (radius + orad) ** 2:
                too_close = True
                break
        if too_close:
            continue
        occupied.append((x, y, z, radius))
        return x, y, z
    return None


def _yaw_quat() -> list[float]:
    """Random yaw around +Z only. Pitch/roll randomization would tip props
    sideways which looks unnatural (e.g. a mug lying on its side)."""
    yaw = float(np.random.uniform(0.0, 2.0 * np.pi))
    return [float(np.cos(yaw / 2)), 0.0, 0.0, float(np.sin(yaw / 2))]


def _sample_props(
    table_center: Tuple[float, float],
    table_half_size: Tuple[float, float],
) -> list[PropConfig]:
    """Pick a theme, then fill NUM_PROP_SLOTS entries (some active, rest off)."""
    theme = str(np.random.choice(PROP_THEMES))
    pool = _PROPS_BY_THEME[theme]
    # `pool` can be empty if the manifest is missing or the theme genuinely has
    # zero assets — either way: nothing to place.
    n_active = int(np.random.randint(PROP_COUNT_RANGE[0], PROP_COUNT_RANGE[1] + 1))
    if not pool:
        n_active = 0

    occupied: list[Tuple[float, float, float, float]] = []
    slots: list[PropConfig] = []
    # props.xml declares one body per asset_id, so an asset can appear at most
    # once per scene — we track and skip duplicates here rather than have the
    # runtime silently no-op the second occurrence (which would leave the slot
    # count below n_active).
    used_ids: set[str] = set()
    # Pre-partition the theme pool by zone-suitability. Shelf/table accept
    # small items only; floor accepts items large enough to read as furniture.
    zone_pools = {
        "shelf": [a for a in pool if max(a["size"]) <= SHELF_MAX_DIM],
        "table": [a for a in pool if max(a["size"]) <= TABLE_MAX_DIM],
        "floor": [a for a in pool if max(a["size"]) >= FLOOR_MIN_DIM],
    }
    zone_weights = {"shelf": P_SHELF_ZONE, "table": P_TABLE_ZONE, "floor": P_FLOOR_ZONE}
    for i in range(NUM_PROP_SLOTS):
        if i >= n_active:
            slots.append(PropConfig(active=False))
            continue
        r = np.random.random()
        cum = 0.0
        zone = "shelf"
        for zname, w in zone_weights.items():
            cum += w
            if r < cum:
                zone = zname
                break
        candidates_all = zone_pools[zone]
        if not candidates_all:
            # Theme has nothing for the chosen zone — fall through the others
            # in priority order so the slot still places something visible.
            for fallback in ("shelf", "table", "floor"):
                if fallback != zone and zone_pools[fallback]:
                    zone = fallback
                    candidates_all = zone_pools[fallback]
                    break
        candidates = [a for a in candidates_all if a["id"] not in used_ids]
        if not candidates:
            # Theme exhausted — every remaining asset is already placed.
            slots.append(PropConfig(active=False))
            continue
        asset = candidates[np.random.randint(len(candidates))]
        pose = _sample_prop_pose(
            asset, zone, table_center, table_half_size, occupied
        )
        if pose is None:
            slots.append(PropConfig(active=False))
            continue
        used_ids.add(asset["id"])
        x, y, z = pose
        slots.append(PropConfig(
            active=True,
            asset_id=asset["id"],
            pos=[x, y, z],
            quat=_yaw_quat(),
        ))
    return slots


# --- Wall art (paintings / boards) ----------------------------------------
#
# Sits on wall_x_neg (x≈-2.0) — the wall that's empty from the default render
# camera POV, opposite the side that holds the shelf. The wall_art geom is
# declared in scene.xml with default alpha=0; this sampler decides each episode
# whether to show a painting, a board, or nothing.

# Texture names must match `<texture name="painting_*">` entries in scene.xml.
# Re-run scripts/download_paintings.py and update both this list and scene.xml
# if the painting pool changes.
WALL_PAINTING_TEXTURES = (
    "painting_9", "painting_11", "painting_874", "painting_14655",
    "painting_30368", "painting_44065", "painting_61158", "painting_64754",
    "painting_65821", "painting_87088", "painting_90048", "painting_97933",
    "painting_111377", "painting_146701", "painting_181777",
)
# Solid-color boards. Each entry is the central rgba; the sampler perturbs
# slightly so two whiteboards don't look identical. Whiteboard / corkboard /
# chalkboard cover the three common "thing hanging on an office wall" looks.
_BOARD_PRESETS = (
    (0.95, 0.95, 0.92),  # whiteboard
    (0.55, 0.40, 0.25),  # corkboard
    (0.15, 0.30, 0.20),  # chalkboard
)
# Wall plane lives at x=-2.00 with 0.01m thickness, so the inside face is at
# x=-1.99. The wall_art geom is a (rotated) plane with zero thickness sitting
# 5mm in front of the wall — close enough to read as "on" the wall, far enough
# to avoid z-fighting with the wall's inside face.
_WALL_ART_X = -1.985
_WALL_ART_HALF_THICKNESS = 0.0  # unused for plane geom, kept for schema
# Restrict y to the camera-visible portion of wall_x_neg. The render camera
# sits at ~(+x, +y, +z) and looks toward the (-x, -y) corner where the shelf
# is, so its principal ray hits wall_x_neg roughly around y in [-1.6, -0.2];
# anything outside this strip is off-frame and wastes a DR slot. (Originally
# allowed y up to +1.5 — paintings sampled there were rendered correctly but
# behind/beside the camera so the user couldn't see them.)
# z is computed dynamically from the sampled half-height so the slab always
# clears the floor (>0.1m gap) and the ceiling (<1.65m top), regardless of
# whether it's a small painting or a tall whiteboard.
_WALL_ART_Y_RANGE = (-1.6, -0.2)
_WALL_ART_Z_FLOOR_CLEAR = 0.1
_WALL_ART_Z_CEILING_TOP = 1.65
# Painting size (half-extents in wall-plane). The wall is ~2.6m from the
# render camera; at the lower bound (0.4 half-height ⇒ 0.8m tall) the painting
# spans roughly 17° vertically, which is small-but-noticeable. Smaller than
# this and the painting disappears against the randomized wall texture
# (originally 0.25 lower-bound = 0.5m ⇒ 11° = invisible most of the time).
# Aspect ratio is sampled separately and mapped onto (hy, hz).
_PAINTING_HEIGHT_RANGE = (0.40, 0.70)  # half-extent ⇒ rendered 0.8-1.4m tall
_PAINTING_ASPECT_RANGE = (0.7, 1.6)    # width / height
# Boards run a touch larger on average — a whiteboard / pinboard reads as
# bigger than a framed painting in a real office.
_BOARD_HEIGHT_RANGE = (0.50, 0.80)
_BOARD_ASPECT_RANGE = (1.0, 2.0)

P_WALL_ART_PAINTING = 0.60
P_WALL_ART_BOARD = 0.25
# Remaining 0.15 is "blank wall" — the policy still needs to handle the empty
# case, otherwise it learns to expect art every episode.


def _sample_wall_art() -> WallArtConfig:
    r = np.random.random()
    if r < P_WALL_ART_PAINTING:
        kind = "painting"
    elif r < P_WALL_ART_PAINTING + P_WALL_ART_BOARD:
        kind = "board"
    else:
        return WallArtConfig(active=False)

    if kind == "painting":
        height_h = float(np.random.uniform(*_PAINTING_HEIGHT_RANGE))
        aspect = float(np.random.uniform(*_PAINTING_ASPECT_RANGE))
        texture = str(np.random.choice(WALL_PAINTING_TEXTURES))
        rgba = (1.0, 1.0, 1.0, 1.0)
    else:
        height_h = float(np.random.uniform(*_BOARD_HEIGHT_RANGE))
        aspect = float(np.random.uniform(*_BOARD_ASPECT_RANGE))
        texture = None
        base = _BOARD_PRESETS[np.random.randint(len(_BOARD_PRESETS))]
        # ±0.05 jitter per channel — keeps the three presets distinguishable
        # while breaking exact-color repetition across episodes.
        jitter = np.random.uniform(-0.05, 0.05, 3)
        rgba = (
            float(np.clip(base[0] + jitter[0], 0, 1)),
            float(np.clip(base[1] + jitter[1], 0, 1)),
            float(np.clip(base[2] + jitter[2], 0, 1)),
            1.0,
        )
    width_h = height_h * aspect
    y = float(np.random.uniform(*_WALL_ART_Y_RANGE))
    z_lo = _WALL_ART_Z_FLOOR_CLEAR + height_h
    z_hi = _WALL_ART_Z_CEILING_TOP - height_h
    z = float(np.random.uniform(z_lo, z_hi))
    return WallArtConfig(
        active=True,
        pos=[_WALL_ART_X, y, z],
        half_size=[_WALL_ART_HALF_THICKNESS, width_h, height_h],
        texture_name=texture,
        rgba=list(rgba),
    )


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

    # --- Material Randomization ---
    # The three pickable blocks and three target plates are all 3D-PLA-printed
    # parts in the real world, so they share the filament/layer-line appearance
    # sampler. Plates are opaque now (was alpha=0.9) — the real target is a solid
    # printed pad, not a translucent slab. `is_plate` only nudges the finish; the
    # named color still drives the prompt-facing color label.
    object_material = _sample_pla_object_material(object_color_name)
    object_distractor1_material = _sample_pla_object_material(
        object_distractor1_color_name
    )
    object_distractor2_material = _sample_pla_object_material(
        object_distractor2_color_name
    )
    target_material = _sample_pla_object_material(target_color_name, is_plate=True)
    distractor1_material = _sample_pla_object_material(
        target_distractor1_color_name, is_plate=True
    )
    distractor2_material = _sample_pla_object_material(
        target_distractor2_color_name, is_plate=True
    )
    floor_material = _sample_material_for_texture(
        np.random.choice(AVAILABLE_TEXTURES), randomize_texrepeat=True
    )
    wall_material = _sample_material_for_texture(
        np.random.choice(AVAILABLE_TEXTURES), randomize_texrepeat=True
    )
    table_material = _sample_material_for_texture(
        np.random.choice(AVAILABLE_TEXTURES), randomize_texrepeat=True
    )

    # --- Table Geometry ---
    # The arm's initial gripper site sits at ~(0, -0.37, 0.47) (measured via
    # mj_forward on the default model), so objects/distractors/goal spawn in
    # x ∈ [-0.14, 0.12], y ∈ [-0.54, -0.24]. The table is centered between
    # the arm base (at the origin) and that workspace, with half-sizes chosen
    # so even the smallest sample covers both the arm base and the full spawn
    # region with margin.
    center_x = float(np.random.uniform(-0.05, 0.05))
    center_y = float(np.random.uniform(-0.30, -0.15))
    top_hx = float(np.random.uniform(0.30, 0.55))
    top_hy = float(np.random.uniform(0.50, 0.75))
    # Visual-only thickness — collision is handled by the deep `table_collision`
    # box, so the slab can stay slim without risking tunneling.
    top_hz = float(np.random.uniform(0.01, 0.025))
    ped_hx = float(np.random.uniform(0.15, top_hx - 0.10))
    ped_hy = float(np.random.uniform(0.15, top_hy - 0.10))
    ped_hz = (_TABLE_HEIGHT - 2 * top_hz) / 2
    ped_z = -(2 * top_hz) - ped_hz
    table = TableConfig(
        top_half_size=[top_hx, top_hy, top_hz],
        top_pos=[center_x, center_y, -top_hz],
        pedestal_half_size=[ped_hx, ped_hy, ped_hz],
        pedestal_pos=[center_x, center_y, ped_z],
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
    table_center = (center_x, center_y)
    lighting_mood = _sample_lighting_mood()
    headlight = _sample_headlight(lighting_mood)
    # scene_light is directional in scene.xml — the *only* light in the scene
    # that casts shadows (top_light is a point light, headlight has no shadow
    # casting). Elevation is biased low (see _sample_positional_light) so most
    # samples produce long, prominent arm shadows that sweep when the arm moves.
    # Distance kept moderate so the shadow map's per-meter resolution stays high.
    scene_light = _sample_positional_light(
        lighting_mood,
        table_center,
        distance_range=(1.5, 3.0),
        elevation_range_deg=(15.0, 70.0),
    )
    # top_light stays mostly overhead — acts as the "ceiling fixture" fill.
    top_light = _sample_positional_light(
        lighting_mood,
        table_center,
        distance_range=(1.5, 2.5),
        elevation_range_deg=(55.0, 88.0),
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

    # --- Background prop sampling ---
    props = _sample_props(
        table_center=table_center,
        table_half_size=(top_hx, top_hy),
    )
    wall_art = _sample_wall_art()

    domain_rand_config = DomainRandConfig(
        object_material=object_material,
        target_material=target_material,
        distractor1_material=distractor1_material,
        distractor2_material=distractor2_material,
        object_distractor1_material=object_distractor1_material,
        object_distractor2_material=object_distractor2_material,
        floor_material=floor_material,
        wall_material=wall_material,
        table_material=table_material,
        table=table,
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
        props=props,
        wall_art=wall_art,
    )

    return domain_rand_config, object_color_name, target_color_name
