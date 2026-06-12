"""Kinematic grasp lock shared by the data-collection interface and the eval
env.

MuJoCo contact-based grasping of the small PLA blocks is unstable (blocks squirt
out of the jaws, slip mid-lift), which made physical-grasp trajectory collection
yield only a few percent usable episodes. The fix is a kinematic lock: at grasp
time we snapshot the held object's pose in the gripper frame and the jaw qpos,
then re-apply both every sim step so there is zero relative motion between
gripper and object regardless of arm dynamics.

This lives in `aera` (not the semi_autonomous interface) so BOTH consumers can
share the exact same mechanism:

- The collection interface (`Ar4Mk3RobotInterface`) calls `engage()` explicitly
  once its scripted close completes, and `enforce()` after every sim step.
- The eval env (`Ar4Mk3Env`, behind the `kinematic_grasp` flag) infers the
  engage/release moments from the policy's gripper command and calls the same
  methods. Sharing one implementation means eval reproduces collection's grasp
  behavior exactly instead of drifting from it.
"""

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np


@dataclass
class GraspEngageConfig:
    """Gate deciding when the kinematic lock attaches an object.

    The default reproduces the original permissive behavior: a single 5 cm
    grip-site-to-object-center snap with no alignment check, so existing
    datasets and policies are unchanged unless this is opted into.

    Set ``require_alignment=True`` for the demanding gate — the gripper must
    actually be aligned over the object (jaws straddling it laterally, at the
    right height) to grab. A small mis-aligned approach then genuinely fails to
    grasp, which is what makes realistic grasp-failure / recovery data possible
    instead of the 5 cm snap turning every near-miss into a perfect grab.

    The lock is shared by data collection and eval, so enabling this makes both
    attach objects under identical, physically-meaningful rules.

    Tolerances are world-frame because the grasp is top-down: ``lateral_tol`` is
    the max horizontal (xy) offset between the grip site and the object center,
    ``height_tol`` the max vertical (z) offset. Defaults are sized against the
    24 mm block / ~14 mm jaw travel so a clean grasp (sub-cm lateral, ~12 mm
    vertical) passes while a ~20 mm near-miss fails.
    """

    require_alignment: bool = False
    max_distance: float = 0.05  # coarse center-distance bound, always applied
    lateral_tol: float = 0.012  # max world-xy grip->object offset (m)
    height_tol: float = 0.030  # max world-z grip->object offset (m)


class KinematicGraspLock:
    """Snap a grasped object (and the jaw qpos) rigidly to the gripper.

    Holds a reference to the env's `model`/`data` (both persist across resets in
    MuJoCo — `mj_resetData` mutates `data` in place — so the references stay
    valid). Stateless until `engage()` records a held object; `enforce()` is a
    no-op while nothing is held.
    """

    def __init__(
        self,
        model,
        data,
        grasp_object_names: Sequence[str],
        gripper_body_name: str = "gripper_base_link",
        gripper_joint_names: Sequence[str] = (
            "gripper_jaw1_joint",
            "gripper_jaw2_joint",
        ),
        engage_config: Optional[GraspEngageConfig] = None,
    ):
        self.model = model
        self.data = data
        self.grasp_object_names = tuple(grasp_object_names)
        self.gripper_body_name = gripper_body_name
        self.gripper_joint_names = list(gripper_joint_names)
        # Default = permissive (old 5cm snap, no alignment) so behavior is
        # unchanged unless a caller opts into the demanding gate.
        self.engage_config = engage_config or GraspEngageConfig()

        self._held_object_name: Optional[str] = None
        self._held_relpose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._held_jaw_qpos: Optional[np.ndarray] = None
        self._held_jaw_qpos_indices: Optional[np.ndarray] = None
        self._held_jaw_dof_indices: Optional[np.ndarray] = None

    @property
    def is_held(self) -> bool:
        return self._held_object_name is not None

    @property
    def held_object(self) -> Optional[str]:
        return self._held_object_name

    def _slide_indices(self, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
        """qpos/dof index arrays for slide joints (the jaws are 1-DoF each)."""
        qpos_idx, dof_idx = [], []
        for name in joint_names:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            qpos_idx.append(self.model.jnt_qposadr[jid])
            dof_idx.append(self.model.jnt_dofadr[jid])
        return np.array(qpos_idx), np.array(dof_idx)

    def engage(
        self, max_distance: Optional[float] = None
    ) -> Tuple[Optional[str], Optional[str], float]:
        """Lock the closest known object near the grip site, subject to the gate.

        Returns (locked_name, closest_name, closest_dist): locked_name is None
        if the closest object failed the gate (nothing locked). Snapshots the
        object's pose in the gripper frame and the current jaw qpos on success.

        ``max_distance`` overrides the config's coarse center-distance bound when
        given; otherwise the config value is used. When the config has
        ``require_alignment`` set, the object must additionally be laterally and
        vertically aligned with the grip site (jaws actually straddling it), so a
        small mis-aligned approach fails to grab.
        """
        cfg = self.engage_config
        if max_distance is None:
            max_distance = cfg.max_distance

        grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip")
        grip_pos = self.data.site_xpos[grip_site_id]

        best_name: Optional[str] = None
        best_dist = float("inf")
        for obj_name in self.grasp_object_names:
            body_id = self.model.body(obj_name).id
            dist = float(np.linalg.norm(self.data.xpos[body_id] - grip_pos))
            if dist < best_dist:
                best_dist = dist
                best_name = obj_name

        if best_name is None or best_dist > max_distance:
            return None, best_name, best_dist

        # Demanding gate: the gripper must straddle the object (small horizontal
        # offset) at the object's height (small vertical offset). World-frame
        # because the grasp is top-down. A near-miss violates the lateral bound
        # and so fails to lock — the physical basis for realistic recovery data.
        if cfg.require_alignment:
            obj_pos = self.data.xpos[self.model.body(best_name).id]
            lateral = float(np.linalg.norm(obj_pos[:2] - grip_pos[:2]))
            vertical = float(abs(obj_pos[2] - grip_pos[2]))
            if lateral > cfg.lateral_tol or vertical > cfg.height_tol:
                return None, best_name, best_dist

        body1_id = self.model.body(self.gripper_body_name).id
        body2_id = self.model.body(best_name).id
        p1 = self.data.xpos[body1_id]
        q1 = self.data.xquat[body1_id]
        p2 = self.data.xpos[body2_id]
        q2 = self.data.xquat[body2_id]

        # Express body2's pose in body1's frame: rel_pos = R(q1)^T (p2 - p1),
        # rel_quat = q1^-1 * q2.
        q1_inv = np.empty(4)
        mujoco.mju_negQuat(q1_inv, q1)
        rel_pos = np.empty(3)
        mujoco.mju_rotVecQuat(rel_pos, p2 - p1, q1_inv)
        rel_quat = np.empty(4)
        mujoco.mju_mulQuat(rel_quat, q1_inv, q2)

        self._held_object_name = best_name
        self._held_relpose = (rel_pos.copy(), rel_quat.copy())
        self._held_jaw_qpos_indices, self._held_jaw_dof_indices = self._slide_indices(
            self.gripper_joint_names
        )
        self._held_jaw_qpos = self.data.qpos[self._held_jaw_qpos_indices].copy()
        return best_name, best_name, best_dist

    def enforce(self) -> None:
        """Re-apply the held object's world pose and the pinned jaw qpos.

        Called after every sim step so any physics drift during integration is
        overwritten before the next step or render. No-op while nothing is held.
        """
        if self._held_object_name is None or self._held_relpose is None:
            return
        rel_pos, rel_quat = self._held_relpose

        body1_id = self.model.body(self.gripper_body_name).id
        p1 = self.data.xpos[body1_id]
        q1 = self.data.xquat[body1_id]

        # World pose: p_obj = p1 + R(q1) * rel_pos, q_obj = q1 * rel_quat
        rotated = np.empty(3)
        mujoco.mju_rotVecQuat(rotated, rel_pos, q1)
        p_obj = p1 + rotated
        q_obj = np.empty(4)
        mujoco.mju_mulQuat(q_obj, q1, rel_quat)

        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{self._held_object_name}:joint"
        )
        qpos_addr = self.model.jnt_qposadr[joint_id]
        dof_addr = self.model.jnt_dofadr[joint_id]
        self.data.qpos[qpos_addr : qpos_addr + 3] = p_obj
        self.data.qpos[qpos_addr + 3 : qpos_addr + 7] = q_obj
        self.data.qvel[dof_addr : dof_addr + 6] = 0.0

        # Pin the jaw positions too: fast arm motion can otherwise push a jaw
        # past its frictionloss threshold and make it slide, which manifests as
        # the jaws "jumping" relative to the kinematically-held object.
        if self._held_jaw_qpos is not None:
            self.data.qpos[self._held_jaw_qpos_indices] = self._held_jaw_qpos
            self.data.qvel[self._held_jaw_dof_indices] = 0.0

        # Refresh derived quantities so render and downstream reads see the
        # corrected pose this frame, not next step.
        mujoco.mj_forward(self.model, self.data)

    def release(self) -> None:
        """Clear the lock so the object is back under physics."""
        self._held_object_name = None
        self._held_relpose = None
        self._held_jaw_qpos = None
        self._held_jaw_qpos_indices = None
        self._held_jaw_dof_indices = None
