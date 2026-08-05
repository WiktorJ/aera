"""One definition of the pick-and-place task environment, shared by collection
and eval.

Collection and eval used to build ``Ar4Mk3EnvConfig`` independently, and they
drifted. Four mismatches were found by reading one against the other — camera
parameterization, camera randomization, object yaw, and the prompt string — each
of which quietly made the scored eval a different task from the one the policy
was trained on. Nothing prevented a fifth.

So the rule is: **eval env config == collection env config**, with eval layering
only its own knobs on top. This module holds the collection values as the single
definition; anything eval-specific goes through ``eval_overrides``. The prompt
template and the DR-generator call live here too, for the same reason.

The eval-only knobs are the ones with no collection counterpart: ``n_substeps``
(collection steps the sim directly and never calls ``env.step``),
``kinematic_grasp`` (collection drives the lock explicitly),
``relative_action_scale`` / ``absolute_state_actions`` (action decoding),
``include_images_in_obs`` (collection records images through the interface), and
``obs_image_aug`` (eval-time sensor realism).
"""

import dataclasses
from typing import Any, Optional

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig, Q, T

# Exactly what collection records (lowercase, no trailing period). A capitalized
# or punctuated variant is a different string to the tokenizer, and eval used to
# send one.
PROMPT_TEMPLATE = "pick the {object_color} block and place it on the {target_color} target"
PICK_PROMPT_TEMPLATE = "pick the {object_color} block"
PLACE_PROMPT_TEMPLATE = "place on the {target_color} target"

# The DR axes collection turns on. Both are opt-in flags on the generator, and
# eval called it with bare defaults — so every DR-on eval scene sat at the one
# fixed default camera pose while training saw random anchor-hull poses.
COLLECTION_DR_FLAGS = {
    "randomize_cameras": True,
    "randomize_arm_dynamics": True,
}

# --- Deploy rate ------------------------------------------------------------
#
# One definition, because SuiteConfig and run_policy_on_env.Args used to carry
# separate copies that drifted (replan 10 vs 5, max_episode_steps 1000 vs 400,
# n_substeps 3 vs 20). suite.py imports run_policy_on_env, so the duplication
# could not be collapsed by pointing one at the other; they both point here.
#
# THE INVARIANT: n_substeps == record_every * skip. An action delta spans
# record_every * skip * 2 ms of motion and one env.step integrates
# n_substeps * 2 ms, so they must be equal or the arm executes each delta at the
# wrong speed.
#
# Collection records one frame every COLLECTION_RECORD_EVERY mj-steps rather
# than every mj-step, because ~99% of a recorded frame's cost is the two camera
# renders (measured: 13.7 ms of renders against 0.09 ms of physics) and the
# transform then throws most of those frames away. Decimating at collection and
# at transform are exactly equivalent — the transform pairs state[t] with
# state[t + skip] in *recorded* frames, so the delta spans record_every * skip
# raw steps and the frame stride matches — so only the product is a learning
# choice. The split is a cost/flexibility choice: a smaller record_every costs
# more to collect but leaves room to re-transform at a finer rate without
# re-collecting.
COLLECTION_RECORD_EVERY = 5
DATASET_SKIP = 2
DEPLOY_N_SUBSTEPS = COLLECTION_RECORD_EVERY * DATASET_SKIP
# <option timestep="0.002"> in ar4_mk3.xml. Here so rate arithmetic reads as
# rate arithmetic instead of a bare 0.002 at each call site.
MJ_TIMESTEP_S = 0.002
# Inference every 4 env steps = 80 ms at n_substeps=10.
DEPLOY_REPLAN_STEPS = 4
# ~3x the demonstrated episode length (250-330 env steps at n_substeps=10).
DEPLOY_MAX_EPISODE_STEPS = 1000

# Fields eval is allowed to differ on. Anything else diverging is a bug, and
# test_task_env_factory asserts exactly that.
EVAL_ONLY_FIELDS = frozenset(
    {
        "n_substeps",
        "kinematic_grasp",
        "relative_action_scale",
        "include_images_in_obs",
        "absolute_state_actions",
        "obs_image_aug",
        "obs_image_aug_strength",
    }
)


def build_task_env_config(
    model_path: str,
    domain_rand: Any = None,
    *,
    use_geometric_lookat: bool = True,
    randomize_object_yaw: bool = True,
    eval_overrides: Optional[dict] = None,
) -> Ar4Mk3EnvConfig:
    """Build the task env config, collection's values by definition.

    Args:
        model_path: Absolute path to scene.xml.
        domain_rand: A config from ``generate_random_domain_rand_config``, or
            None for the un-randomized dev/visualization scene. Note None is
            *not* an in-distribution eval condition — collection has no clean
            episodes, so it is the most OOD point in appearance space and must
            never be reported as a scored number.
        use_geometric_lookat: Exposed only because collection carries it as a
            flag. The default is the value collection runs; eval must not set
            it, and it must not be turned off while camera DR is on.
        randomize_object_yaw: Same — a collection flag whose default here is
            what collection actually runs. Eval leaving it off made every eval
            block spawn axis-aligned while training saw random yaw, which made
            eval easier than training and hid the diagonal-grasp failure.
        eval_overrides: Eval-only knobs (see ``EVAL_ONLY_FIELDS``). Passing any
            other key is rejected, because that would be eval silently
            redefining the task.
    """
    config = Ar4Mk3EnvConfig(
        model_path=model_path,
        reward_type="sparse",
        use_eef_control=False,
        # T/Q are reinterpreted as T_GEOMETRIC/Q_GEOMETRIC by __post_init__ when
        # use_geometric_lookat is set. The two must move together: the camera DR
        # offsets are validated against the geometric parameterization, so
        # applying them to the other base lands the camera on the far side of
        # the scene (measured: azimuth 209 deg vs 264 deg for the same offset).
        translation=T,
        quaterion=Q,
        use_geometric_lookat=use_geometric_lookat,
        distance_multiplier=1.2,
        z_offset=0.3,
        domain_rand=domain_rand,
        randomize_object_yaw=randomize_object_yaw,
    )

    if eval_overrides:
        unknown = set(eval_overrides) - EVAL_ONLY_FIELDS
        if unknown:
            raise ValueError(
                f"eval_overrides may only set eval-only knobs, got {sorted(unknown)}. "
                "Everything else must match collection — that is the point of "
                "this factory. If a field genuinely needs to differ, add it to "
                "EVAL_ONLY_FIELDS with a reason."
            )
        config = dataclasses.replace(config, **eval_overrides)

    return config


def build_prompt(object_color: str, target_color: str) -> str:
    """The combined task prompt, in collection's exact format."""
    return PROMPT_TEMPLATE.format(object_color=object_color, target_color=target_color)


def build_phase_prompts(object_color: str, target_color: str) -> tuple[str, str]:
    """(pick, place) prompts for two-phase prompting."""
    return (
        PICK_PROMPT_TEMPLATE.format(object_color=object_color),
        PLACE_PROMPT_TEMPLATE.format(target_color=target_color),
    )
