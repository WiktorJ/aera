"""Regression tests for the shared collection/eval env-config factory.

These exist because four train/eval mismatches were found by hand, one at a
time, by reading _build_env against collect_trajectories — and nothing prevented
a fifth. The point of the factory is that a new divergence has to be declared in
EVAL_ONLY_FIELDS rather than appearing silently, and these tests are what make
that true.
"""

import dataclasses

import numpy as np
import pytest

from aera.autonomous.envs.ar4_mk3_config import Q_GEOMETRIC, T_GEOMETRIC
from aera.autonomous.envs.task_env_factory import (
    COLLECTION_DR_FLAGS,
    EVAL_ONLY_FIELDS,
    build_phase_prompts,
    build_prompt,
    build_task_env_config,
)

_MODEL_PATH = "/tmp/scene.xml"  # never loaded; the factory only stores the path


def _collection_config():
    return build_task_env_config(_MODEL_PATH, domain_rand=None)


def _eval_config(**overrides):
    return build_task_env_config(
        _MODEL_PATH,
        domain_rand=None,
        eval_overrides={
            "n_substeps": 10,
            "absolute_state_actions": False,
            "include_images_in_obs": True,
            "kinematic_grasp": True,
            "relative_action_scale": 1.0,
            **overrides,
        },
    )


def test_collection_and_eval_differ_only_in_eval_only_fields():
    # THE test this module exists for.
    collection, evaluation = _collection_config(), _eval_config()
    differing = set()
    for f in dataclasses.fields(collection):
        a = getattr(collection, f.name)
        b = getattr(evaluation, f.name)
        same = (
            np.array_equal(a, b)
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray)
            else a == b
        )
        if not same:
            differing.add(f.name)
    assert differing <= EVAL_ONLY_FIELDS, (
        f"eval diverges from collection on non-eval-only fields: "
        f"{sorted(differing - EVAL_ONLY_FIELDS)}"
    )


def test_eval_cannot_override_a_task_defining_field():
    # The guard rail: silently redefining the task is what the factory prevents.
    for field in ("randomize_object_yaw", "use_geometric_lookat", "reward_type"):
        with pytest.raises(ValueError, match="eval-only"):
            build_task_env_config(
                _MODEL_PATH, domain_rand=None, eval_overrides={field: False}
            )


def test_camera_parameterization_is_geometric():
    # E2 mismatch 2, the subtle one: the camera DR offsets are validated
    # against the GEOMETRIC parameterization, so randomize_cameras without
    # use_geometric_lookat lands the camera on the far side of the scene —
    # strictly worse than no camera DR at all. __post_init__ swaps T/Q for
    # their geometric counterparts, so both flags move together by construction.
    config = _collection_config()
    assert config.use_geometric_lookat is True
    np.testing.assert_allclose(config.translation, T_GEOMETRIC)
    np.testing.assert_allclose(config.quaterion, Q_GEOMETRIC)


def test_object_yaw_is_randomized():
    # E2 mismatch 3: eval left this off, so every eval block spawned
    # axis-aligned while training saw random yaw. That makes eval EASIER than
    # training and hides the diagonal-grasp failure mode.
    assert _collection_config().randomize_object_yaw is True
    assert _eval_config().randomize_object_yaw is True


def test_dr_flags_match_what_collection_enables():
    # E2 mismatches 1 and 4: eval called the generator bare, which left
    # randomize_cameras False.
    assert COLLECTION_DR_FLAGS == {
        "randomize_cameras": True,
        "randomize_arm_dynamics": True,
    }


def test_prompt_is_lowercase_and_unpunctuated():
    # E2 mismatch 4: eval built "Pick the ... target." — a different string to
    # the tokenizer than anything in training.
    prompt = build_prompt("yellow", "red")
    assert prompt == "pick the yellow block and place it on the red target"
    assert not prompt.endswith(".")
    assert prompt[0].islower()


def test_phase_prompts():
    assert build_phase_prompts("yellow", "red") == (
        "pick the yellow block",
        "place on the red target",
    )
