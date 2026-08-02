"""Minimal tests for the crucial fixture-free helpers in transform_skip_dataset."""

import numpy as np

from aera.autonomous.openpi.scripts.transform_skip_dataset import (
    _build_output_repo_id,
    _gripper_moved,
    _parse_image_from_sample,
)


def test_build_output_repo_id_naming():
    # A wrong name silently writes/uploads the wrong dataset, so pin the scheme.
    assert _build_output_repo_id("org/name", 5, False, None) == "org/name_skip5"
    assert _build_output_repo_id("org/name", 5, True, None) == "org/name_skip5_delta"
    assert _build_output_repo_id("org/name", 3, True, "custom") == "org/name_custom"


def test_parse_image_float_chw_to_uint8_hwc():
    # LeRobot stores float32 CHW; the pipeline (and obs-aug) needs uint8 HWC.
    chw = np.zeros((3, 2, 2), dtype=np.float32)
    chw[0] = 1.0  # full red channel
    out = _parse_image_from_sample(chw)
    assert out.dtype == np.uint8
    assert out.shape == (2, 2, 3)
    assert (out[..., 0] == 255).all()
    assert (out[..., 1] == 0).all() and (out[..., 2] == 0).all()


def _frame(joints, gripper):
    return np.array(list(joints) + [gripper], dtype=np.float32)


def test_gripper_guard_reads_the_state_not_the_action():
    # The case that motivates the guard: mid-close ramp. The jaws are physically
    # travelling (state -0.0140 -> -0.0136) while a binarized action holds the
    # same "closed" label on both frames. Judging on the action would call this
    # frame static and — since the interface parks the arm during the ramp —
    # delete the whole grasp window.
    joints = [0.1] * 6
    moved = _gripper_moved(
        state_now=_frame(joints, -0.0136),
        state_prev=_frame(joints, -0.0140),
        action_now=_frame(joints, 0.0),
        action_prev=_frame(joints, 0.0),
        num_joint_dims=6,
        gripper_eps=0.0002,
    )
    assert moved


def test_gripper_guard_ignores_a_parked_jaw():
    joints = [0.1] * 6
    assert not _gripper_moved(
        state_now=_frame(joints, -0.01050),
        state_prev=_frame(joints, -0.01049),
        action_now=_frame(joints, 0.0),
        action_prev=_frame(joints, 0.0),
        num_joint_dims=6,
        gripper_eps=0.0002,
    )


def test_gripper_guard_falls_back_to_the_action_without_state():
    joints = [0.1] * 6
    assert _gripper_moved(
        state_now=None,
        state_prev=None,
        action_now=_frame(joints, 0.0),
        action_prev=_frame(joints, -0.014),
        num_joint_dims=6,
        gripper_eps=0.0002,
    )
