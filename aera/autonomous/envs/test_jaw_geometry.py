"""Tests for the shared jaw-contact geometry.

These pin the numbers that three call sites (scripted close target, the lock's
close-depth gate, the eval metric's ``close_shallow``) now all derive from, and
that were previously wrong by a constant +0.4 mm in all three at once.
"""

import mujoco
import numpy as np
import pytest

from aera.autonomous.envs.ar4_mk3_config import _PLA_BLOCK_SIZES
from aera.autonomous.envs.jaw_geometry import (
    DEFAULT_GRASP_SQUEEZE,
    GRIPPER_JAW_QPOS_MAX,
    GRIPPER_JAW_QPOS_MIN,
    engage_qpos,
    first_contact_qpos,
    is_pinchable,
    pad_inner_offset,
)

_MODEL_PATH = "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml"


@pytest.fixture(scope="module")
def model():
    return mujoco.MjModel.from_xml_path(_MODEL_PATH)


def test_pad_inner_offset_matches_the_measured_gap_law(model):
    # Measured on the real model: gap(mm) = 0.8 - 2*qpos(mm), i.e. the pads'
    # inner faces lead the jaw origins by 0.4 mm each. This is the constant the
    # old -(half_width) model omitted.
    assert pad_inner_offset(model) == pytest.approx(0.0004, abs=1e-9)


def test_pad_gap_is_affine_in_qpos_in_the_real_model(model):
    # Derives the law from the simulator rather than trusting the constant:
    # drive both jaws to a qpos, forward, and measure the actual world-frame
    # distance between the pads' inner faces.
    data = mujoco.MjData(model)
    jaw_qadr = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
        for n in ("gripper_jaw1_joint", "gripper_jaw2_joint")
    ]
    pad_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)
        for n in ("gripper_jaw1_contact", "gripper_jaw2_contact")
    ]

    for q in (-0.014, -0.010, -0.006, 0.0):
        data.qpos[jaw_qadr[0]] = q
        data.qpos[jaw_qadr[1]] = q
        mujoco.mj_forward(model, data)
        # Pinch axis is the gripper body's local x; project both pad centres
        # onto it and subtract each pad's own half-thickness to reach the faces.
        gripper_xmat = data.xmat[
            model.body("gripper_base_link").id
        ].reshape(3, 3)
        pinch_axis = gripper_xmat[:, 0]
        projections = [float(np.dot(data.geom_xpos[g], pinch_axis)) for g in pad_ids]
        half_thickness = float(model.geom_size[pad_ids[0]][0])
        gap = abs(projections[0] - projections[1]) - 2 * half_thickness
        assert gap == pytest.approx(0.0008 - 2 * q, abs=1e-6)


@pytest.mark.parametrize(
    "half_width, expected_mm",
    [
        (0.0095, -9.1),   # 19 mm
        (0.0110, -10.6),  # 22 mm
        (0.0120, -11.6),  # 24 mm — the measured sweep's block
        (0.0135, -13.1),  # 27 mm, distractor-only
    ],
)
def test_first_contact_for_each_preset(model, half_width, expected_mm):
    assert first_contact_qpos(model, half_width) * 1000 == pytest.approx(
        expected_mm, abs=1e-6
    )


def test_engage_target_clears_the_measured_engagement_threshold(model):
    # The 24 mm sweep: commands at -11.5 mm registered zero pad contacts and the
    # lock refused; -11.0 mm gave 8 contacts and engaged. The default squeeze
    # must land on the engaging side of that boundary.
    target = engage_qpos(model, 0.0120)
    assert target * 1000 == pytest.approx(-11.0, abs=1e-6)
    assert target > -0.0115, "target must not fall back into the never-engages band"


def test_engage_is_first_contact_plus_squeeze(model):
    for half_width in _PLA_BLOCK_SIZES:
        assert engage_qpos(model, half_width) == pytest.approx(
            first_contact_qpos(model, half_width) + DEFAULT_GRASP_SQUEEZE
        )


def test_graspable_presets_are_reachable_and_the_30mm_one_is_not(model):
    graspable = [h for h in _PLA_BLOCK_SIZES if h <= 0.012]
    assert len(graspable) == 3  # 19 / 22 / 24 mm
    for half_width in graspable:
        assert is_pinchable(model, half_width)
        assert GRIPPER_JAW_QPOS_MIN <= engage_qpos(model, half_width) <= GRIPPER_JAW_QPOS_MAX

    # The 30 mm preset is unpinchable BY CONSTRUCTION, which is what lets it be
    # used as a distractor. At full open the pad gap is 28.8 mm, so the jaws
    # cannot even straddle it.
    assert first_contact_qpos(model, 0.0150) < GRIPPER_JAW_QPOS_MIN
    assert not is_pinchable(model, 0.0150)


def test_asymmetric_or_missing_pads_fail_loudly(model):
    with pytest.raises(ValueError, match="not found"):
        pad_inner_offset(model, geom_names=("gripper_jaw1_contact", "nope"))

    # A geometry edit that breaks jaw symmetry must not be silently averaged
    # away — the whole point of deriving from the model is to catch that.
    import copy

    skewed = copy.copy(model)
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "gripper_jaw2_contact")
    original = float(skewed.geom_pos[gid][0])
    skewed.geom_pos[gid][0] = original - 0.001
    try:
        with pytest.raises(ValueError, match="asymmetric"):
            pad_inner_offset(skewed)
    finally:
        skewed.geom_pos[gid][0] = original


def test_jaw_actuators_are_force_limited(model):
    # A full-close command leaves a ~10 mm position error against kp=10000, so
    # without a force limit the jaws drive ~102 N into the block: 0.7 mm of
    # visible interpenetration and a buzzing contact. Every other actuator in
    # the model has a forcerange; these were the only ones that didn't.
    for name in ("act8", "act9"):
        act_id = model.actuator(name).id
        assert model.actuator_forcelimited[act_id], f"{name} must be force-limited"
        lo, hi = model.actuator_forcerange[act_id]
        assert hi == pytest.approx(-lo), f"{name} force limit should be symmetric"
        # Lower bound: the jaw joints carry frictionloss=2, and at +/-5 N the
        # jaws no longer complete the travel onto the smallest block at all.
        # Upper bound: past ~15 N the pads visibly press into the block.
        assert 7.0 <= hi <= 15.0, f"{name} forcerange {hi} N outside the measured window"
