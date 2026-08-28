#!/usr/bin/env python3
"""
Run a trained policy on the AR4 MK3 environment.

This script connects to a running policy server, initializes the AR4 MK3 environment,
and runs evaluation episodes. It records videos of the rollouts and reports
success rates.

Example usage:
1. Start the policy server:
    python aera/autonomous/openpi/scripts/serve_policy.py --checkpoint.config pi0_fast_ar4_mk3_low_mem_finetune --checkpoint.dir <path_to_checkpoint>

2. Run this script:
    python semi_autonomous/aera_semi_autonomous/scripts/run_policy_on_env.py --prompt "pick the red block and place it on the green target"

3. Two-phase prompting (dataset-faithful):
    python .../run_policy_on_env.py --two-phase-prompt --pick-color red --place-color green
   Press Enter in the terminal to transition from pick to place.
"""

import collections
import dataclasses
import logging
import os
import pathlib
import select
import sys
from typing import Any

import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tyro

from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera.autonomous.envs.task_env_factory import (
    COLLECTION_DR_FLAGS,
    DEPLOY_MAX_EPISODE_STEPS,
    DEPLOY_N_SUBSTEPS,
    DEPLOY_REPLAN_STEPS,
    build_phase_prompts,
    build_prompt,
    build_task_env_config,
)
from aera.autonomous.openpi.eval import metrics as _metrics
from aera_semi_autonomous.data.domain_rand_config_generator import (
    generate_random_domain_rand_config,
)
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# Environment constants
ENV_RESOLUTION = 256  # resolution used for rendered images
ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIPPER_JOINT_NAME = "gripper_jaw1_joint"
GRIPPER_OPEN_ACTION = 1.0  # Normalized gripper action: -1 closed, +1 open

# Two-phase prompting
PHASE_PICK = "pick"
PHASE_PLACE = "place"


@dataclasses.dataclass
class Args:
    """Arguments for running the policy on the environment."""

    # --- Model server parameters ---
    host: str = "0.0.0.0"
    port: int = 8000
    prompt: str = "pick the yellow block and place it on the red target"

    # --- Evaluation parameters ---
    # Rate defaults are shared with SuiteConfig via task_env_factory. They used
    # to be separate copies here and drifted badly (replan 5 vs 10,
    # max_episode_steps 400 vs 1000, n_substeps 20 vs 3), so an ad-hoc run of
    # this script silently evaluated at a different rate than the scored suite.
    replan_steps: int = DEPLOY_REPLAN_STEPS
    num_episodes: int = 1
    max_episode_steps: int = DEPLOY_MAX_EPISODE_STEPS
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize
    resize_size: int = 224

    # --- Environment parameters ---
    domain_rand: bool = False
    headless: bool = False
    # Mirror the data-collection kinematic grasp lock into eval so success isn't
    # confounded by unstable MuJoCo contact-grasp physics (the demos are
    # collected with this lock). Disable to eval under raw friction grasping.
    kinematic_grasp: bool = True
    # mj-steps integrated per env.step (= per policy action). MUST match the
    # `--skip` the trained dataset was built with: the action delta spans
    # `skip * 0.002 s` of motion, and one env.step integrates
    # `n_substeps * 0.002 s`, so they have to be equal for the arm to move at
    # the rate the policy expects. Override for an older checkpoint (e.g. a
    # skip=3 one → n_substeps=3).
    n_substeps: int = DEPLOY_N_SUBSTEPS
    # Scale applied to the policy's relative arm-joint action in the env. The
    # dataset stores joint deltas in radians (Unnormalize restores physical
    # units), so the policy output is applied directly at 1.0. Use a smaller
    # value only for a policy whose arm output is normalized to ~[-1, 1].
    relative_action_scale: float = 1.0
    # Apply the shared sensor-realism image augmentation to eval renders so
    # sim-eval matches the augmented training distribution. Off by default.
    obs_image_aug: bool = False
    obs_image_aug_strength: float = 1.0

    # --- Two-phase prompting ---
    # When enabled, the prompt starts as "pick the {color} block" and switches
    # to "place on the {color} target" once the user presses Enter in the
    # terminal. This matches how the training data is annotated (separate
    # pick/place segments) and avoids the confusion of a combined prompt.
    two_phase_prompt: bool = False
    pick_color: str = "yellow"  # Used when two_phase_prompt and not domain_rand
    place_color: str = "red"  # Used when two_phase_prompt and not domain_rand

    # --- Safety ---
    skip_decode_failures: bool = False  # When enabled, handle likely FAST decode failures (all-identical action chunk)
    replan_steps_on_failure: int = 1  # Steps to execute from a failed decode chunk before replanning

    # --- Utils ---
    video_out_path: str = "data/ar4_mk3/videos"
    seed: int = 7


def _is_decode_failure(action_chunk: np.ndarray) -> bool:
    """Detect likely FAST tokenizer decode failures.

    When decoding fails, the server returns the unnormalized mean action
    (zeros in normalized space passed through Unnormalize), resulting in
    all rows of the action chunk being identical.
    """
    return np.allclose(action_chunk, action_chunk[0], atol=1e-5)


def _find_model_path() -> str | None:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    candidates = [
        os.path.join(
            project_root,
            "aera",
            "autonomous",
            "simulation",
            "mujoco",
            "ar4_mk3",
            "scene.xml",
        ),
        "aera/autonomous/simulation/mujoco/ar4_mk3/scene.xml",
    ]
    return next((os.path.abspath(p) for p in candidates if os.path.exists(p)), None)


def _resolve_prompts(args: Args) -> tuple[str, str, Any]:
    """Resolve domain rand config and (pick_prompt, place_prompt).

    For non-two-phase mode, both prompts are identical (the combined prompt).
    """
    if args.domain_rand:
        # Same generator call as collection (see COLLECTION_DR_FLAGS). Calling
        # it bare left randomize_cameras off, so every DR-on eval scene sat at
        # the single fixed default camera pose while training saw random
        # anchor-hull poses.
        domain_rand_config, object_color, target_color = (
            generate_random_domain_rand_config(**COLLECTION_DR_FLAGS)
        )
    else:
        # Dev/visualization only, never a scored number: collection has no
        # clean episodes, so this scene differs from every training episode on
        # every visual axis at once — the most OOD point in appearance space,
        # not a baseline that "should" be easier.
        domain_rand_config = None
        object_color = args.pick_color
        target_color = args.place_color

    if args.two_phase_prompt:
        pick_prompt, place_prompt = build_phase_prompts(object_color, target_color)
    elif args.domain_rand:
        # Collection's exact string. This used to be capitalized and
        # punctuated, i.e. a different string to the tokenizer than anything
        # the policy was trained on.
        pick_prompt = place_prompt = build_prompt(object_color, target_color)
    else:
        pick_prompt = place_prompt = args.prompt

    return pick_prompt, place_prompt, domain_rand_config


def _build_env(args: Args, model_path: str, domain_rand_config: Any) -> Ar4Mk3PickAndPlaceEnv:
    # Everything task-defining comes from the shared factory, so eval cannot
    # drift from collection again; only the eval-only knobs are layered on.
    env_config = build_task_env_config(
        model_path,
        domain_rand_config,
        eval_overrides={
            "n_substeps": args.n_substeps,
            "absolute_state_actions": False,
            "include_images_in_obs": True,
            "kinematic_grasp": args.kinematic_grasp,
            "relative_action_scale": args.relative_action_scale,
            "obs_image_aug": args.obs_image_aug,
            "obs_image_aug_strength": args.obs_image_aug_strength,
        },
    )
    return Ar4Mk3PickAndPlaceEnv(render_mode="rgb_array", config=env_config)


def _setup_display() -> dict:
    matplotlib.use("TkAgg")
    plt.ion()
    fig, (ax_env, ax_gripper) = plt.subplots(1, 2, figsize=(10, 5))
    ax_env.set_title("AR4 Mk3 Env")
    ax_env.axis("off")
    ax_gripper.set_title("AR4 Mk3 Gripper")
    ax_gripper.axis("off")
    return {
        "fig": fig,
        "ax_env": ax_env,
        "ax_gripper": ax_gripper,
        "im_env": None,
        "im_gripper": None,
    }


def _update_display(display: dict, img: np.ndarray, gripper_img: np.ndarray) -> None:
    if display["im_env"] is None:
        display["im_env"] = display["ax_env"].imshow(img)
        display["im_gripper"] = display["ax_gripper"].imshow(gripper_img)
    else:
        display["im_env"].set_data(img)
        display["im_gripper"].set_data(gripper_img)
    display["fig"].canvas.draw_idle()
    display["fig"].canvas.flush_events()
    plt.pause(0.001)


def _stdin_pressed() -> bool:
    """Non-blocking check for any line on stdin (Enter pressed)."""
    if not sys.stdin or not sys.stdin.isatty():
        return False
    ready, _, _ = select.select([sys.stdin], [], [], 0)
    if not ready:
        return False
    sys.stdin.readline()
    return True


def _is_success(env: Ar4Mk3PickAndPlaceEnv) -> bool:
    """Task success: the manipulated object sits within distance_threshold of the
    goal. The env never sets `terminated` (continuing task), so we compute this
    directly from sim state — mirrors the collector's success check
    (collect_trajectories.run_pick_and_place_and_collect)."""
    object_pos = env._utils.get_site_xpos(env.model, env.data, "object0")
    return bool(np.linalg.norm(object_pos - env.goal) < env.distance_threshold)


def _build_warmup_action() -> np.ndarray:
    # With absolute_state_actions=False, arm actions are relative deltas, so
    # zeros mean "no movement". Gripper is normalized: -1 closed, +1 open.
    #
    # OPEN, and it must stay paired with the env starting with open jaws
    # (Ar4Mk3Env._open_jaws). Holding it closed through the settle window would
    # re-shut the jaws the training data now shows open at t=0 — reintroducing
    # at eval exactly the spurious leading "closed" frames that were just
    # removed from collection.
    warmup_action = np.zeros(7)
    warmup_action[-1] = GRIPPER_OPEN_ACTION
    return warmup_action


def _query_policy(
    client: _websocket_client_policy.WebsocketClientPolicy,
    img: np.ndarray,
    gripper_img: np.ndarray,
    state: np.ndarray,
    prompt: str,
    args: Args,
    last_successful_gripper_action: float,
) -> tuple[np.ndarray, float]:
    """Run inference and return (steps_to_enqueue, updated_last_gripper_action)."""
    element: dict[str, Any] = {
        "image": img,
        "gripper_image": gripper_img,
        "state": state,
        "prompt": prompt,
    }
    action_chunk = client.infer(element)["actions"]
    if args.skip_decode_failures and _is_decode_failure(action_chunk):
        logging.warning(
            "Detected FAST decode failure, applying limited steps with frozen gripper"
        )
        steps = action_chunk[: args.replan_steps_on_failure].copy()
        steps[:, 6] = last_successful_gripper_action
        return steps, last_successful_gripper_action
    return action_chunk[: args.replan_steps], action_chunk[0, 6]


def _denormalize_gripper(action: np.ndarray) -> np.ndarray:
    """Convert raw policy gripper output (-0.014=open, 0=closed) to env-normalized
    (-1=closed, +1=open)."""
    action = np.array(action)
    action[6] = (2.0 * action[6] / -0.014) - 1.0
    return action


def _save_episode_video(
    replay_images: list[np.ndarray],
    video_out_path: str,
    episode_idx: int,
    prompt: str,
    success: bool,
) -> None:
    suffix = "success" if success else "failure"
    task_segment = prompt.replace(" ", "_")[:50]
    video_path = (
        pathlib.Path(video_out_path)
        / f"rollout_{episode_idx}_{task_segment}_{suffix}.mp4"
    )
    imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
    logging.info(f"Saved video to {video_path}")


def _run_episode(
    args: Args,
    env: Ar4Mk3PickAndPlaceEnv,
    client: _websocket_client_policy.WebsocketClientPolicy,
    pick_prompt: str,
    place_prompt: str,
    episode_idx: int,
    display: dict | None,
) -> tuple[_metrics.EpisodeMetrics, list[np.ndarray], str]:
    """Runs a single evaluation episode. Returns (metrics, replay_images, final_prompt)."""
    logging.info(f"\nStarting episode {episode_idx + 1}/{args.num_episodes}")
    if args.two_phase_prompt:
        logging.info(f"Phase 1 (pick): {pick_prompt}")
        logging.info(f"Phase 2 (place): {place_prompt}")
        logging.info("Press Enter in the terminal to advance from pick -> place.")
    else:
        logging.info(f"Task: {pick_prompt}")

    obs, _ = env.reset(seed=args.seed + episode_idx)
    action_plan: collections.deque = collections.deque()
    replay_images: list[np.ndarray] = []
    tracker = _metrics.EpisodeTracker(env)
    tracker_started = False
    last_successful_gripper_action: float = 0.0
    phase = PHASE_PICK
    current_prompt = pick_prompt

    warmup_action = _build_warmup_action()
    gripper_qpos_addr = env.model.joint(GRIPPER_JOINT_NAME).qposadr[0]
    arm_qpos_indices = [env.model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES]

    for t in range(args.max_episode_steps):
        try:
            if t < args.num_steps_wait:
                obs, _, _, _, _ = env.step(warmup_action)
                continue

            # Capture the spawn geometry once, after the settle steps, so the
            # progress ratios are normalized against the real starting pose.
            if not tracker_started:
                tracker.start()
                tracker_started = True

            # Check for phase transition (two-phase prompting only).
            if (
                args.two_phase_prompt
                and phase == PHASE_PICK
                and _stdin_pressed()
            ):
                phase = PHASE_PLACE
                current_prompt = place_prompt
                action_plan.clear()  # Force replan with new prompt
                logging.info("Switched to place phase: %s", current_prompt)

            img = obs["default_camera_image"]
            gripper_img = obs["gripper_camera_image"]

            if display is not None:
                _update_display(display, img, gripper_img)

            arm_qpos = env.data.qpos[arm_qpos_indices]
            gripper_qpos = np.array([env.data.qpos[gripper_qpos_addr]])
            current_qpos = np.concatenate((arm_qpos, gripper_qpos))

            img = image_tools.convert_to_uint8(img)
            gripper_img = image_tools.convert_to_uint8(gripper_img)
            replay_images.append(img)

            if not action_plan:
                steps, last_successful_gripper_action = _query_policy(
                    client,
                    img,
                    gripper_img,
                    current_qpos,
                    current_prompt,
                    args,
                    last_successful_gripper_action,
                )
                action_plan.extend(steps)

            action = _denormalize_gripper(action_plan.popleft())
            obs, _, terminated, truncated, _ = env.step(action)
            tracker.update()

            if _is_success(env):
                break
            if terminated or truncated:
                break
        except Exception as e:
            logging.error(f"Caught exception: {e}", exc_info=True)
            break

    # If the episode ended during the settle window (e.g. immediate exception),
    # start() may not have run; produce a zeroed episode rather than crashing.
    if not tracker_started:
        tracker.start()
    return tracker.finalize(), replay_images, current_prompt


def _log_funnel(agg: dict[str, float], prefix: str) -> None:
    """Pretty-print the aggregated funnel + headline scalars."""
    if not agg:
        return
    n = int(agg.get("eval/num_episodes", 0))
    funnel = " / ".join(
        f"{stage}={agg.get(f'eval/funnel/{stage}_rate', 0.0) * 100:.0f}%"
        for stage in ("reached", "grasped", "lifted", "transported", "placed")
    )
    logging.info(f"Funnel ({prefix}, n={n}): {funnel}")
    logging.info(
        "  reach_progress=%.2f place_progress=%.2f grasp_drop_rate=%.2f",
        agg.get("eval/reach_progress_mean", 0.0),
        agg.get("eval/place_progress_mean", 0.0),
        agg.get("eval/grasp_drop_rate", 0.0),
    )


def run_on_env(args: Args) -> None:
    """Runs the policy on the AR4 MK3 environment."""
    # force=True: openpi installs a root handler at import, which would otherwise
    # make this a no-op and leave the root level at WARNING.
    logging.basicConfig(level=logging.INFO, force=True)
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    model_path = _find_model_path()
    if model_path is None:
        logging.error("Could not find AR4 MK3 model file.")
        return

    pick_prompt, place_prompt, domain_rand_config = _resolve_prompts(args)
    env = _build_env(args, model_path, domain_rand_config)
    display = None if args.headless else _setup_display()

    episode_metrics: list[_metrics.EpisodeMetrics] = []
    for episode_idx in range(args.num_episodes):
        ep, replay_images, final_prompt = _run_episode(
            args, env, client, pick_prompt, place_prompt, episode_idx, display
        )
        episode_metrics.append(ep)

        _save_episode_video(
            replay_images, args.video_out_path, episode_idx, final_prompt, ep.placed
        )

        logging.info(
            "Episode finished. reached=%s grasped=%s lifted=%s transported=%s "
            "placed=%s mode=%s (reach_progress=%.2f place_progress=%.2f "
            "attempts=%d close_cycles=%d press_steps=%d)",
            ep.reached, ep.grasped, ep.lifted, ep.transported, ep.placed,
            ep.failure_mode, ep.reach_progress, ep.place_progress,
            ep.grasp_attempt_count, ep.gripper_close_cycles, ep.press_steps,
        )
        _log_funnel(_metrics.aggregate(episode_metrics), prefix="so far")

    env.close()
    if display is not None:
        plt.ioff()
        plt.close("all")
    logging.info("Evaluation finished.")
    _log_funnel(_metrics.aggregate(episode_metrics), prefix="final")


if __name__ == "__main__":
    run_on_env(tyro.cli(Args))
