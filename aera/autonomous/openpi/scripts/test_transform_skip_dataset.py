"""Minimal tests for the crucial fixture-free helpers in transform_skip_dataset."""

from unittest import mock

import numpy as np

from aera.autonomous.envs.task_env_factory import COLLECTION_RECORD_EVERY
from aera.autonomous.openpi.scripts.transform_skip_dataset import (
    _binarize_gripper_action,
    _build_output_repo_id,
    _gripper_moved,
    _parse_image_from_sample,
    _resolve_record_every,
)


def _meta_with(info):
    """Patch LeRobotDatasetMetadata to return a stub carrying `info`."""
    return mock.patch(
        "aera.autonomous.openpi.scripts.transform_skip_dataset.LeRobotDatasetMetadata",
        return_value=mock.Mock(info=info),
    )


def test_build_output_repo_id_naming():
    # A wrong name silently writes/uploads the wrong dataset, so pin the scheme.
    assert _build_output_repo_id("org/name", 5, False, None) == "org/name_skip5"
    assert _build_output_repo_id("org/name", 5, True, None) == "org/name_skip5_delta"
    assert _build_output_repo_id("org/name", 3, True, "custom") == "org/name_custom"


def test_build_output_repo_id_names_the_net_decimation():
    """The name carries record_every * skip, i.e. the deploy n_substeps.

    Naming a record_every=5 / skip=2 dataset "skip2" would tell the deploy side
    to replay a 50 Hz dataset at 250 Hz.
    """
    assert (
        _build_output_repo_id("org/name", 2, True, None, 5) == "org/name_skip10_delta"
    )
    # The two ways of reaching net 10 are the same dataset, so the same name.
    assert _build_output_repo_id("org/name", 2, True, None, 5) == _build_output_repo_id(
        "org/name", 10, True, None, 1
    )


def test_record_every_comes_from_the_dataset():
    with _meta_with({"record_every": 5}):
        assert _resolve_record_every("org/name", None) == 5


def test_record_every_flag_overrides_the_dataset():
    # A dataset built before the key existed has to be tellable.
    with _meta_with({"record_every": 5}):
        assert _resolve_record_every("org/name", 3) == 3


def test_record_every_falls_back_when_the_dataset_predates_the_key():
    with _meta_with({}):
        assert _resolve_record_every("org/name", None) == COLLECTION_RECORD_EVERY


def test_record_every_survives_an_unreadable_dataset():
    # Resolution feeds a log line and a name; it must never fail the transform.
    with mock.patch(
        "aera.autonomous.openpi.scripts.transform_skip_dataset.LeRobotDatasetMetadata",
        side_effect=FileNotFoundError("not cached"),
    ):
        assert _resolve_record_every("org/name", None) == COLLECTION_RECORD_EVERY
        assert _resolve_record_every("org/name", 2) == 2


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


# --- --binarize-gripper ------------------------------------------------------

# The written values are float32, so compare at that precision rather than
# against the float64 literals.
OPEN = np.float32(-0.014)
CLOSED = np.float32(0.0)


def _binarized(gripper, threshold=-0.013):
    return _binarize_gripper_action(_frame([0.1] * 6, gripper), 6, threshold)[6]


def test_binarize_maps_either_side_of_the_threshold():
    # Just closed of the threshold -> full close; just open of it -> full open.
    assert _binarized(-0.0129) == CLOSED
    assert _binarized(-0.0131) == OPEN
    # The threshold itself is "not above" -> open, so the mapping is total.
    assert _binarized(-0.013) == OPEN


def test_binarize_collapses_the_whole_close_ramp_to_two_values():
    # The ramp is what motivates this: the recorded action is the measured next
    # jaw qpos, so a close writes a run of mid-values that teach partial closes.
    ramp = [-0.014, -0.0138, -0.0125, -0.011, -0.0105, -0.005, 0.0]
    out = [_binarized(g) for g in ramp]
    assert set(out) == {OPEN, CLOSED}
    # And it stays monotone: open frames first, then closed, no flapping.
    assert out == sorted(out)


def test_binarize_is_width_invariant():
    # The whole point: three block sizes stall the jaws at three different
    # qpos values, and all three must produce the SAME "closed" command so the
    # policy never has to regress half-width from pixels.
    holding = [-0.0095, -0.011, -0.012]  # 19 / 22 / 24 mm blocks
    assert {_binarized(g) for g in holding} == {CLOSED}


def test_binarize_leaves_the_arm_joints_untouched():
    joints = [0.1, -0.2, 0.3, -0.4, 0.5, -0.6]
    out = _binarize_gripper_action(_frame(joints, -0.007), 6, -0.013)
    np.testing.assert_allclose(out[:6], joints)


def test_binarized_action_is_static_while_the_state_still_moves():
    # The interaction that made T3 necessary. Across a close ramp the binarized
    # action is constant, so an action-based idle filter would drop every frame
    # of the grasp window; the state-based guard keeps them.
    joints = [0.1] * 6
    a_prev = _binarize_gripper_action(_frame(joints, -0.0125), 6, -0.013)
    a_now = _binarize_gripper_action(_frame(joints, -0.0110), 6, -0.013)
    assert a_prev[6] == a_now[6]  # constant label...
    assert _gripper_moved(                     # ...but the world moved
        state_now=_frame(joints, -0.0110),
        state_prev=_frame(joints, -0.0125),
        action_now=a_now,
        action_prev=a_prev,
        num_joint_dims=6,
        gripper_eps=0.0002,
    )
