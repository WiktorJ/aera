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
        close_depth_tol: Slack on the close-depth gate (engage's
            ``close_ctrl_target``): the commanded jaw target may stop this far
            short of the candidate's surface (``-pinch_half_width``) and still
            count as a committed close. Matches the collection preload (0.5 mm),
            so a demo-faithful close command passes with exactly that margin.
        require_pinch_contact: Only engage while BOTH jaw contact pads are in
            contact with the candidate, with contact normals roughly along the
            pinch axis — i.e. the object is physically pinched between the
            jaws right now. This is what stops the weld from gluing a block to
            the *outside* of the jaws (front / side / below the fingertips):
            the centre-offset tolerances above allow offsets larger than the
            pads' own extent, so without this gate a closed gripper brushing a
            block can weld it mid-air. It also means eval engagement completes
            only after the jaws have physically closed onto the block (the
            close command alone no longer welds), so the welded pose is the
            genuinely pinched pose — recentred by contact physics exactly as
            in collection's scripted close — not a snapshot taken while the
            jaws were still open.
        pinch_normal_align: Min |cos| between a pad-object contact normal and
            the pinch axis for that contact to count as pinching. Rejects the
            fully-closed jaws pressing down on a block's top face (normal is
            vertical) while accepting tilted/diagonal but genuine pinches.
    """

    require_alignment: bool = True
    max_distance: float = 0.05  # coarse center-distance bound, always applied
    pinch_tol: float = 0.007    # gripper-local x (jaws close across this)
    finger_tol: float = 0.020   # gripper-local y (along the jaws)
    height_tol: float = 0.027   # gripper-local z (tool approach axis)
    close_depth_tol: float = 0.0005  # jaw-travel slack for the close-depth gate
    require_pinch_contact: bool = True
    pinch_normal_align: float = 0.5  # |cos| >= this vs pinch axis, ~60 deg cone


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
        jaw_contact_geom_names: Sequence[str] = (
            "gripper_jaw1_contact",
            "gripper_jaw2_contact",
        ),
        engage_config: Optional[GraspEngageConfig] = None,
    ):
        self.model = model
        self.data = data
        self.grasp_object_names = tuple(grasp_object_names)
        self.gripper_body_name = gripper_body_name
        self.gripper_joint_names = list(gripper_joint_names)
        self.jaw_contact_geom_names = tuple(jaw_contact_geom_names)
        # Demanding alignment gate is on by default; pass an engage_config with
        # require_alignment=False for the old permissive 5cm snap.
        self.engage_config = engage_config or GraspEngageConfig()

        self._held_object_name: Optional[str] = None
        self._held_relpose: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._held_jaw_qpos: Optional[np.ndarray] = None
        self._held_jaw_qpos_indices: Optional[np.ndarray] = None
        self._held_jaw_dof_indices: Optional[np.ndarray] = None
        # Deferred-pin bookkeeping (see maybe_pin_jaws).
        self._pin_prev_jaw_qpos: Optional[np.ndarray] = None
        self._pin_calls: int = 0

    @property
    def is_held(self) -> bool:
        return self._held_object_name is not None

    @property
    def jaws_pinned(self) -> bool:
        """True once the jaw qpos is being enforced (immediately for
        ``engage(pin_jaws=True)``, after :meth:`maybe_pin_jaws` otherwise)."""
        return self._held_jaw_qpos is not None

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

    def _pinch_half_width(self, obj_name: str) -> Optional[float]:
        """Half-width of the object's collision box across the pinch dimension
        (the shorter horizontal extent — the side the yaw-aligned jaws close
        on). None if the object has no geom named after it."""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, obj_name)
        if geom_id == -1:
            return None
        size = self.model.geom_size[geom_id]
        return float(min(size[0], size[1]))

    def jaws_pinching(self, obj_name: str) -> bool:
        """True while the object is physically pinched between the jaws.

        Requires a current contact between EACH jaw contact pad and the
        object, with the contact normal within ``pinch_normal_align`` of the
        pinch axis (gripper-local x, the jaws' slide axis). One-sided touches,
        or the closed jaws pressing on the block's top face (vertical
        normals), don't count. Reads ``data.contact``, so it reflects the last
        stepped/forwarded state.
        """
        obj_body_id = self.model.body(obj_name).id
        pad_ids = []
        for name in self.jaw_contact_geom_names:
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid == -1:
                raise ValueError(f"Jaw contact geom '{name}' not found in model")
            pad_ids.append(gid)

        gripper_xmat = self.data.xmat[
            self.model.body(self.gripper_body_name).id
        ].reshape(3, 3)
        pinch_axis = gripper_xmat[:, 0]

        pinching = {gid: False for gid in pad_ids}
        for i in range(self.data.ncon):
            con = self.data.contact[i]
            for gid in pad_ids:
                if con.geom1 == gid:
                    other = con.geom2
                elif con.geom2 == gid:
                    other = con.geom1
                else:
                    continue
                if self.model.geom_bodyid[other] != obj_body_id:
                    continue
                normal = con.frame[:3]
                align = abs(float(np.dot(normal, pinch_axis)))
                if align >= self.engage_config.pinch_normal_align:
                    pinching[gid] = True
        return all(pinching.values())

    def engage(
        self,
        max_distance: Optional[float] = None,
        pin_jaws: bool = True,
        close_ctrl_target: Optional[float] = None,
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

        ``pin_jaws=True`` (collection: engage is called after a scripted close
        has physically completed, so the current jaw qpos is the closed-on-block
        pose) freezes the jaws at their current qpos immediately.
        ``pin_jaws=False`` (eval: engage is retried on every control step while
        the policy commands a close; with the pinch-contact gate it succeeds
        once the jaws have stalled against the object, but they may still be
        settling into their preload) leaves the jaws under actuator control so
        they finish closing onto the welded object; the caller then pins them
        via :meth:`maybe_pin_jaws` once they settle. Without that deferral the
        jaws could be frozen slightly off their final closed-on-block pose —
        an image the training demos never contain.

        ``close_ctrl_target`` (eval): the commanded jaw target, in jaw-qpos
        units. When given, the candidate only locks if the command reaches its
        surface — ``close_ctrl_target >= -(pinch_half_width + close_depth_tol)``
        — i.e. the policy committed to a genuine close ON THIS object, not a
        twitch off full-open. The bound is per-candidate because the grasp
        command scales with block width (collection closes to
        ``-(half_width - preload)``), so no flat threshold fits every size.
        None (collection) skips the gate: there the scripted close has already
        physically happened.
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

        # Close-depth gate: the commanded jaw target must reach this candidate's
        # surface (within close_depth_tol slack) before the weld is granted.
        if close_ctrl_target is not None:
            half_width = self._pinch_half_width(best_name)
            if half_width is not None and close_ctrl_target < -(
                half_width + cfg.close_depth_tol
            ):
                return None, best_name, best_dist

        # Pinch-contact gate: the candidate must be physically pinched between
        # both jaw pads right now. In eval (engage retried every control step
        # while the close command holds) this defers the weld until the jaws
        # have actually closed onto the block, so a block outside/below the
        # jaws can never be welded and the snapshot pose is the real pinched
        # pose. See GraspEngageConfig.require_pinch_contact.
        if cfg.require_pinch_contact and not self.jaws_pinching(best_name):
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
        self._held_jaw_qpos = (
            self.data.qpos[self._held_jaw_qpos_indices].copy() if pin_jaws else None
        )
        self._pin_prev_jaw_qpos = None
        self._pin_calls = 0
        return best_name, best_name, best_dist

    def maybe_pin_jaws(
        self, settle_tol: float = 1e-5, max_wait_calls: int = 25
    ) -> bool:
        """Pin the jaws once their closing motion has physically settled.

        Intended to be called once per control step after an
        ``engage(pin_jaws=False)``. The jaws settle when consecutive samples
        move less than ``settle_tol`` (they've stalled against the held object
        or reached their target); ``max_wait_calls`` force-pins after that many
        calls so contact jitter can't leave them unpinned forever. Returns True
        once the jaws are pinned (or already were). No-op while nothing is held.
        """
        if self._held_object_name is None:
            return False
        if self._held_jaw_qpos is not None:
            return True
        qpos = self.data.qpos[self._held_jaw_qpos_indices].copy()
        self._pin_calls += 1
        settled = self._pin_prev_jaw_qpos is not None and bool(
            np.max(np.abs(qpos - self._pin_prev_jaw_qpos)) < settle_tol
        )
        if settled or self._pin_calls >= max_wait_calls:
            self._held_jaw_qpos = qpos
            self._pin_prev_jaw_qpos = None
            return True
        self._pin_prev_jaw_qpos = qpos
        return False

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
        self._pin_prev_jaw_qpos = None
        self._pin_calls = 0
