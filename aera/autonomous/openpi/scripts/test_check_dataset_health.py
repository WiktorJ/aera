"""Tests for the fixture-free helpers in check_dataset_health.

These pin the gripper-structure parsing, which is what checks 5-7 rest on: a
wrong episode profile turns the pre-training gate into a rubber stamp.
"""

import numpy as np

from aera.autonomous.openpi.scripts.check_dataset_health import (
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    _hysteretic_closed,
    profile_episode,
)


def _episode(cmd, state=None):
    cmd = np.asarray(cmd, dtype=float)
    state = np.asarray(state if state is not None else cmd, dtype=float)
    heights = np.full(len(cmd), 0.1)
    return profile_episode(cmd, state, heights, eps=1e-5)


def test_hysteresis_ignores_jitter_across_a_single_threshold():
    # Unbinarized data records the measured jaw qpos, which jitters across any
    # fixed line. A bare `cmd > -0.013` test would flip on every sample here and
    # report four grasp cycles; inside the band the state must simply hold.
    jitter = np.array([-0.0131, -0.0129, -0.0131, -0.0129, -0.0131])
    closed = _hysteretic_closed(jitter)
    assert int(np.count_nonzero(closed[1:] != closed[:-1])) <= 1
    assert int(np.count_nonzero(jitter[1:] > -0.013) != 0)  # naive test would chatter


def test_hysteresis_needs_the_full_band_to_reopen():
    cmd = np.array([GRIPPER_OPEN, GRIPPER_CLOSED, -0.0132, GRIPPER_OPEN])
    # -0.0132 sits inside the band, so the gripper stays "closed" until the
    # command drops past the release level.
    assert list(_hysteretic_closed(cmd)) == [False, True, True, False]


def test_clean_episode_has_one_cycle_and_no_leading_closed():
    cmd = [GRIPPER_OPEN] * 5 + [GRIPPER_CLOSED] * 5 + [GRIPPER_OPEN] * 3
    prof = _episode(cmd)
    assert prof.leading_closed == 0
    assert prof.transitions == 2  # one close, one release
    assert prof.close_at == 5
    assert len(prof.release_heights_m) == 1


def test_jaws_closed_at_episode_start_is_counted():
    # qpos0 has both jaws at 0, so every episode opens with a closed->open ramp
    # that binarizes to "commanded closed" while nothing is held.
    cmd = [GRIPPER_CLOSED] * 4 + [GRIPPER_OPEN] * 5 + [GRIPPER_CLOSED] * 4
    assert _episode(cmd).leading_closed == 4


def test_recovery_episode_shows_extra_cycles():
    # partial_grasp: close, lift, release while aloft, then re-grasp.
    cmd = ([GRIPPER_OPEN] * 3 + [GRIPPER_CLOSED] * 3 + [GRIPPER_OPEN] * 3
           + [GRIPPER_CLOSED] * 3 + [GRIPPER_OPEN] * 2)
    prof = _episode(cmd)
    assert prof.transitions == 4  # > 2 is the recovery signature
    assert len(prof.release_heights_m) == 2


def test_grasp_window_spans_the_ramp_not_just_its_end():
    # On unbinarized data the recorded action is the measured jaw qpos, so it
    # only crosses the closed threshold once the jaws are nearly shut: the
    # window has to expand backwards or it reports the tail of the ramp.
    ramp = np.linspace(GRIPPER_OPEN, -0.011, 9)
    state = np.concatenate([[GRIPPER_OPEN] * 4, ramp, [-0.011] * 4])
    prof = _episode(state)  # command == state, as in an unbinarized dataset
    assert prof.close_at is not None
    assert prof.grasp_window >= 8, prof.grasp_window


def test_window_is_scale_free_against_open_jaw_jitter():
    # An open jaw jitters against its limit stop by the same order as the tail
    # of a close ramp, so an absolute epsilon would bleed into it.
    rng = np.random.default_rng(0)
    jitter = GRIPPER_OPEN + rng.normal(0, 2e-5, 20)
    ramp = np.linspace(GRIPPER_OPEN, -0.0105, 6)
    state = np.concatenate([jitter, ramp, [-0.0105] * 10])
    prof = _episode(state)
    assert 4 <= prof.grasp_window <= 12, prof.grasp_window
