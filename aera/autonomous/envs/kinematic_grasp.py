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

    With ``require_alignment`` (the default), the gripper must actually be
    positioned to grasp the object before the lock engages, so the kinematic
    weld can't grant a "free" grasp the real arm wouldn't get. This makes sim
    eval predictive (a misaligned policy fails in sim as it would on hardware)
    and only lets well-aligned grasps become "success" in collected data. Set
    it False to restore the old permissive 5 cm snap (e.g. to A/B, or to eval an
    older policy under the conditions it was trained on).

    The gate is checked in the gripper's tool frame, because the real grasp
    envelope is strongly anisotropic: the jaws close along their pinch axis
    (small tolerance) but the object can sit far along the finger axis (large
    tolerance). Tolerances are calibrated to the measured physical (non-locked)
    grasp envelope of the 24 mm block (the binding / largest case): it holds to
    ~6 mm pinch / ~20 mm finger / ~27 mm tool-height offset and fails beyond, so
    the gate sits just inside those. The lock is shared by collection and eval,
    so both attach objects under identical, physically-meaningful rules.

    Attributes:
        require_alignment: Enable the demanding tool-frame gate (default on).
        max_distance: Coarse grip-site-to-object-centre bound, always applied.
        pinch_tol: Max offset along the jaws' pinch axis (gripper local x).
        finger_tol: Max offset along the finger axis (gripper local y).
        height_tol: Max offset along the tool approach axis (gripper local z).
    """

    require_alignment: bool = True
    max_distance: float = 0.05  # coarse center-distance bound, always applied
    pinch_tol: float = 0.007    # gripper-local x (jaws close across this)
    finger_tol: float = 0.020   # gripper-local y (along the jaws)
    height_tol: float = 0.027   # gripper-local z (tool approach axis)


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
        # Demanding alignment gate is on by default; pass an engage_config with
        # require_alignment=False for the old permissive 5cm snap.
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

        # Demanding gate: the object must be within the gripper's real grasp
        # envelope, checked in the gripper's TOOL FRAME because that envelope is
        # anisotropic — tight across the jaws' pinch axis, loose along the
        # fingers, moderate in height. The grip-site->object offset is rotated
        # into the gripper body frame and bounded per axis, so a near-miss fails
        # to lock and the kinematic weld can't grant a grasp the real arm
        # wouldn't get. (Calibrated to the measured non-locked grasp envelope;
        # see GraspEngageConfig. The one case it can't catch is a small pinch-
        # axis miss that the closing jaws shove back to centre — no engage-time
        # check sees past that recentering; only an actual test-lift would.)
        if cfg.require_alignment:
            obj_pos = self.data.xpos[self.model.body(best_name).id]
            gripper_q = self.data.xquat[self.model.body(self.gripper_body_name).id]
            q_inv = np.empty(4)
            mujoco.mju_negQuat(q_inv, gripper_q)
            local = np.empty(3)
            mujoco.mju_rotVecQuat(local, obj_pos - grip_pos, q_inv)
            if (
                abs(local[0]) > cfg.pinch_tol
                or abs(local[1]) > cfg.finger_tol
                or abs(local[2]) > cfg.height_tol
            ):
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
