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

# T/Q calibrated for use_geometric_lookat=True. Picked to reproduce the exact
# same default rendered view (lookat, azim, elev, distance) as the legacy
# (T, Q) above under the geometric ray origin, so flipping the flag without
# overriding translation/quaterion leaves the baseline view unchanged.
T_GEOMETRIC = np.array(
    [-0.40970032586752225, 0.25031248388773203, 0.8542931529018112]
)
Q_GEOMETRIC = np.array(
    [
        -0.4537194255805848,
        -0.8725520599992705,
        0.08176549604975476,
        -0.16157347894252677,
    ]
)

AVAILABLE_TEXTURES = [
    # Legacy hand-picked textures kept for exotic classes that the bulk
    # ambientCG downloader doesn't cover (printed packaging, polished metal,
    # glass, glazed ceramic). These are ~1K or smaller; replace when 2K+
    # equivalents are sourced.
    "brass-ambra",
    "bread",
    "can",
    "ceramic",
    "cereal",
    "glass",
    "lemon",
    "metal",
    "soda",
    # ambientCG 2K CC0 textures (scripts/download_textures.py).
    "wood049", "wood051", "wood058", "wood066", "wood067",
    "wood092", "wood094", "wood095",
    "woodfloor043", "woodfloor051", "woodfloor064", "woodfloor070",
    "metal046b", "metal048a", "metal049a", "metal055a", "metal061b", "metal063",
    "metalplates006", "metalplates013",
    "plaster001", "plaster002", "plaster003", "plaster007",
    "paintedplaster015", "paintedplaster016", "paintedplaster017",
    "bricks075a", "bricks097", "bricks101", "bricks102", "bricks104",
    "concrete034", "concrete046", "concrete047a", "concrete048",
    "tiles040", "tiles078", "tiles107", "tiles132a", "tiles133a", "tiles138",
    "fabric030", "fabric061", "fabric066", "fabric081c", "fabric083",
    "marble012", "marble016", "marble021",
    "plastic006", "plastic010", "plastic012a", "plastic013a", "plastic015a",
]

# Pickable-block visual presets: each is an (edge-shape, half-size) pair. A
# preset realizes one `pla_block_p<i>` mesh (the edge OBJ from
# scripts/generate_pla_block_mesh.py at that scale) and a `<base>_visual_p<i>`
# geom per block body in ar4_mk3.xml. The domain randomizer picks one per block
# per episode (the index into this tuple is the *_block_variant field on
# DomainRandConfig); the runtime shows that geom, hides the rest, and scales the
# collision BOX to the preset's half-size (box geoms scale at runtime; meshes do
# not, which is why size variation is discrete). Edge shapes cycle over 6 values
# and sizes over 5, so both are evenly covered. Keep in sync with the XML geoms.
_PLA_BLOCK_EDGES = (
    "chamfer_s", "chamfer_m", "chamfer_l", "fillet_s", "fillet_m", "fillet_l",
)
_PLA_BLOCK_SIZES = (0.0095, 0.0110, 0.0120, 0.0135, 0.0150)  # half-extent (m)
PLA_BLOCK_PRESETS = tuple(
    (_PLA_BLOCK_EDGES[i % len(_PLA_BLOCK_EDGES)],
     _PLA_BLOCK_SIZES[i % len(_PLA_BLOCK_SIZES)])
    for i in range(15)
)


@dataclasses.dataclass
class MaterialConfig:
    texture_name: Optional[Union[str, Sequence[str]]] = None
    rgba: Optional[Sequence[float]] = None
    specular: Optional[float] = None
    shininess: Optional[float] = None
    reflectance: Optional[float] = None
    texrepeat: Optional[Sequence[float]] = None


@dataclasses.dataclass
class LightConfig:
    pos: Optional[Sequence[float]] = None
    dir: Optional[Sequence[float]] = None
    diffuse: Optional[Sequence[float]] = None
    ambient: Optional[Sequence[float]] = None
    specular: Optional[Sequence[float]] = None
    active: Optional[bool] = None


@dataclasses.dataclass
class CameraConfig:
    pos_offset: Optional[Sequence[float]] = None
    rot_offset_euler: Optional[Sequence[float]] = None


@dataclasses.dataclass
class DynamicsConfig:
    damping: Optional[float] = None
    friction: Optional[Sequence[float]] = None
    mass: Optional[float] = None
    size: Optional[Sequence[float]] = None


@dataclasses.dataclass
class PropConfig:
    """One prop slot in the scene. `asset_id` resolves to a manifest entry
    declared in props.xml; the runtime points the slot geom at that asset's
    mesh + material and places the body at (pos, quat). Inactive slots are
    hidden (rgba alpha=0)."""
    active: bool = False
    asset_id: Optional[str] = None
    pos: Optional[Sequence[float]] = None
    quat: Optional[Sequence[float]] = None  # MuJoCo (w, x, y, z)


@dataclasses.dataclass
class WallArtConfig:
    """Decoration slot on the wall_x_neg side of the room.

    Three sampled variants:
      - inactive: hidden (alpha=0). No texture, no rgba.
      - painting: `texture_name` set to one of the painting_* textures; rgba is
        white so the texture renders unmodified.
      - board: no texture; rgba carries a whiteboard/corkboard/chalkboard tint.

    Position/size are always set when active so an episode can move + resize
    the slab without changing the scene XML.
    """
    active: bool = False
    pos: Optional[Sequence[float]] = None        # (x≈-1.99, y, z)
    half_size: Optional[Sequence[float]] = None  # (0.005, hy, hz)
    texture_name: Optional[str] = None
    rgba: Optional[Sequence[float]] = None


@dataclasses.dataclass
class CableSegmentConfig:
    """One wiring-harness tube segment (a visual-only capsule declared in
    ar4_mk3.xml, parented to a link body). `pos_offset` is ADDED to the geom's
    compiled position to jitter routing — the runtime caches the base position
    so this never accumulates across episodes."""
    geom_name: str
    active: bool = True
    rgba: Optional[Sequence[float]] = None
    radius: Optional[float] = None
    pos_offset: Optional[Sequence[float]] = None


@dataclasses.dataclass
class CableArcConfig:
    """One arched cable, rendered as a chain of capsule sub-segments
    (`<base_name>_s0`.._sN, declared in ar4_mk3.xml and parented to a link).
    The runtime bends the chain into a quadratic arc through start -> midpoint
    + apex_offset -> end (all in the parent link frame), so the cable bows /
    floats off the arm instead of hugging it. apex_offset is the sampled bow
    vector (direction = outward, magnitude = how far it floats)."""
    base_name: str
    n_segments: int
    start: Sequence[float]
    end: Sequence[float]
    apex_offset: Sequence[float]
    active: bool = True
    rgba: Optional[Sequence[float]] = None
    radius: Optional[float] = None


@dataclasses.dataclass
class ArmCableConfig:
    """The arm's wiring harness for one episode. This is the only DR axis that
    perturbs the robot's *shape* (silhouette) rather than its surface
    appearance — see _sample_arm_cables for the rationale. `segments` are short
    straight tube bridges; `arcs` are the long runs that bow off the arm."""
    segments: list[CableSegmentConfig] = field(default_factory=list)
    arcs: list[CableArcConfig] = field(default_factory=list)


@dataclasses.dataclass
class ArmDynamicsConfig:
    """Per-episode physics/actuation overrides for the 6 arm joints — the only
    DR axis that changes how the arm *moves* (tracking stiffness, friction,
    inertia) rather than how it looks. Every field is a length-6 list indexed by
    arm joint (joint_1..joint_6 / act1..act6); a None field leaves the compiled
    value untouched. Values are absolute (already resolved by the sampler), so
    the runtime is a pure setter and nothing accumulates across episodes.

    - kp / kv: position-actuator gains. The runtime writes kp to
      actuator_gainprm[:,0] and (-kp, -kv) to actuator_biasprm[:,1:3], since a
      MuJoCo position actuator computes force = kp*(ctrl - qpos) - kv*qvel.
    - damping / armature / frictionloss: written to dof_damping / dof_armature /
      dof_frictionloss for each joint's DoF.
    - force_limit: symmetric actuator_forcerange ([-f, f]) torque saturation.
    """
    kp: Optional[Sequence[float]] = None
    kv: Optional[Sequence[float]] = None
    damping: Optional[Sequence[float]] = None
    armature: Optional[Sequence[float]] = None
    frictionloss: Optional[Sequence[float]] = None
    force_limit: Optional[Sequence[float]] = None


@dataclasses.dataclass
class TableConfig:
    top_half_size: Optional[Sequence[float]] = None
    top_pos: Optional[Sequence[float]] = None
    pedestal_half_size: Optional[Sequence[float]] = None
    pedestal_pos: Optional[Sequence[float]] = None


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
    table_material: Optional[MaterialConfig] = None
    table: Optional[TableConfig] = None
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
    # Index into PLA_BLOCK_PRESETS choosing the (edge-shape, size) visual preset
    # for each pickable block. The runtime shows that mesh geom and scales the
    # collision box to the preset's size. None leaves the XML default visible.
    object_block_variant: Optional[int] = None
    object_distractor1_block_variant: Optional[int] = None
    object_distractor2_block_variant: Optional[int] = None
    default_camera: Optional["CameraConfig"] = None
    gripper_camera: Optional["CameraConfig"] = None
    # Fixed-length list: one entry per prop slot declared in props.xml. Slots
    # the sampler chose to skip carry active=False and are rendered transparent.
    props: Optional[list[PropConfig]] = None
    wall_art: Optional[WallArtConfig] = None
    # Wiring-harness tubes on the arm — the only DR axis that varies the robot's
    # geometry/silhouette instead of just its materials. None leaves the XML
    # default (all tubes visible, black) untouched.
    arm_cables: Optional[ArmCableConfig] = None
    # Per-joint actuator gains / joint friction / inertia for the arm — varies
    # how the arm tracks commands (stiffness, lag, stiction). None leaves the
    # compiled (CAD-tuned) dynamics untouched.
    arm_dynamics: Optional[ArmDynamicsConfig] = None


@dataclasses.dataclass
class Ar4Mk3EnvConfig:
    model_path: str | None = None
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
    # When True, the lookat ground-plane ray is cast from the camera's true
    # world position (-R.T @ T) instead of the raw extrinsic translation T.
    # The legacy (False) behavior makes the (T, Q) -> (lookat, az, el, dist)
    # map non-surjective and is preserved as default for backward compatibility
    # with existing datasets/policies.
    use_geometric_lookat: bool = False
    include_images_in_obs: bool = False
    domain_rand: Optional[DomainRandConfig] = None
    default_camera_config: dict = field(default_factory=lambda: DEFAULT_CAMERA_CONFIG)
    image_width: int = 224
    image_height: int = 224
    show_grip_overlay: bool = True

    def __post_init__(self):
        # When the geometric lookat is enabled and the user hasn't explicitly
        # overridden translation/quaterion, swap in the geometric-mode
        # calibration so the default rendered view stays identical to legacy.
        if self.use_geometric_lookat:
            if self.translation is T or np.array_equal(self.translation, T):
                self.translation = T_GEOMETRIC
            if self.quaterion is Q or np.array_equal(self.quaterion, Q):
                self.quaterion = Q_GEOMETRIC
