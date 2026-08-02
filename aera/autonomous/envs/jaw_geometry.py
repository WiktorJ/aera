"""Single source of truth for where the gripper jaws actually touch an object.

Three places used to model this independently — the scripted expert's close
target, the kinematic lock's close-depth gate, and the eval metric's
``close_shallow`` reason — and all three assumed the pads make contact when the
jaw qpos reaches ``-half_width``. They don't: the contact pads stand proud of
the jaw body's own origin, so first contact happens ``pad_inner_offset`` earlier
(a constant +0.4 mm on the current model). The error was small enough to look
like a rounding detail and large enough to eat the entire 0.5 mm grasp preload,
which is why a scripted close never registered a single pad contact.

Everything here is derived from the MuJoCo model rather than hardcoded, so a
geometry change in ``ar4_mk3.xml`` cannot silently desync the call sites again.

Sign convention (``ar4_mk3.xml``: ``gripper_jaw{1,2}_joint range="-0.014 0"``):
each jaw is a symmetric slide joint, ``qpos = 0`` fully closed and
``qpos = -0.014`` fully open. Larger (less negative) qpos = more closed.
"""

from typing import Sequence

import mujoco

# Actuator limits for act8/act9 (ar4_mk3.xml ctrlrange="-0.014 0").
GRIPPER_JAW_QPOS_MIN = -0.014
GRIPPER_JAW_QPOS_MAX = 0.0

# How far past first contact a close must go to produce a registerable pinch.
# Measured on a 24 mm block: first contact at -11.6 mm, but the lock only
# engages for commands >= -11.0 mm, because the close is tolerance-limited and
# stops slightly short of whatever it was asked for.
DEFAULT_GRASP_SQUEEZE = 0.0006

JAW_CONTACT_GEOM_NAMES = ("gripper_jaw1_contact", "gripper_jaw2_contact")


def pad_inner_offset(
    model, geom_names: Sequence[str] = JAW_CONTACT_GEOM_NAMES
) -> float:
    """Distance each contact pad's inner face sits from its jaw's origin.

    ``|geom_pos_x| - geom_size_x`` per pad: the pad is a thin box offset inward
    along the pinch axis, so its inner face leads the jaw origin by this much.
    On the current model both pads give 0.0014 - 0.001 = **0.0004 m**, which
    reproduces the measured pad gap exactly: ``gap(mm) = 0.8 - 2*qpos(mm)``.

    Raises:
        ValueError: if a pad geom is missing, or the two pads disagree (an
            asymmetric gripper would break the symmetric-jaw math below).
    """
    offsets = []
    for name in geom_names:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if gid == -1:
            raise ValueError(f"Jaw contact geom '{name}' not found in model")
        offsets.append(abs(float(model.geom_pos[gid][0])) - float(model.geom_size[gid][0]))

    if abs(offsets[0] - offsets[1]) > 1e-9:
        raise ValueError(
            f"Jaw contact pads are asymmetric ({offsets[0]:.6f} vs {offsets[1]:.6f} m); "
            "the symmetric-jaw pinch model does not apply."
        )
    return offsets[0]


def first_contact_qpos(model, half_width: float) -> float:
    """Jaw qpos at which the pads first touch an object of this half-width.

    The pads close symmetrically, so the gap between their inner faces is
    ``2 * (pad_inner_offset - qpos)``; setting that equal to ``2 * half_width``
    gives ``qpos = pad_inner_offset - half_width``.

    Not clamped to the actuator range on purpose: a block too wide to pinch
    returns a qpos below ``GRIPPER_JAW_QPOS_MIN``, and callers should see that
    rather than have it quietly clipped into a reachable-looking target.
    """
    return pad_inner_offset(model) - half_width


def engage_qpos(
    model, half_width: float, squeeze: float = DEFAULT_GRASP_SQUEEZE
) -> float:
    """Jaw qpos target that closes far enough onto the object to actually grip.

    ``first_contact_qpos + squeeze`` — past first contact by enough that a
    tolerance-limited close still lands in contact with registerable
    penetration. Used both as the scripted expert's close target and as the
    threshold the eval gates judge a commanded close against, so the demo and
    the gate can no longer disagree about what "closed on it" means.
    """
    return first_contact_qpos(model, half_width) + squeeze


def is_pinchable(model, half_width: float, squeeze: float = DEFAULT_GRASP_SQUEEZE) -> bool:
    """Whether the jaws can reach a grip on this object at all.

    Two bounds, and the open end is the one that actually binds: the pads must
    be able to open *wider* than the object (``first_contact_qpos`` reachable)
    and still close onto it (``engage_qpos`` reachable). At full open the pad
    gap is only 28.8 mm, so the 30 mm distractor preset needs a first contact
    at -14.6 mm against a -14.0 mm limit — the jaws cannot straddle it at all.
    That is what makes it safe as a distractor-only preset. Checking the squeeze
    end alone would call it pinchable, since -14.6 + 0.6 lands exactly on the
    limit.
    """
    return (
        first_contact_qpos(model, half_width) >= GRIPPER_JAW_QPOS_MIN
        and engage_qpos(model, half_width, squeeze) <= GRIPPER_JAW_QPOS_MAX
    )
