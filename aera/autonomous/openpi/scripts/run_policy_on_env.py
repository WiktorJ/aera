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
"""

import collections
import dataclasses
import logging
import os
import pathlib
from typing import Any

import cv2
import imageio
import numpy as np
import tyro

from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3PickAndPlaceEnv
from aera_semi_autonomous.data.domain_rand_config_generator import (
    generate_random_domain_rand_config,
)
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy

# Environment constants
ENV_RESOLUTION = 256  # resolution used for rendered images
ARM_JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
GRIPPER_JOINT_NAME = "gripper_jaw1_joint"


@dataclasses.dataclass
class Args:
    """Arguments for running the policy on the environment."""

    # --- Model server parameters ---
    host: str = "0.0.0.0"
    port: int = 8000
    prompt: str = "pick the red block and place it on the green target"

    # --- Evaluation parameters ---
    replan_steps: int = 5
    num_episodes: int = 1
    max_episode_steps: int = 400
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize
    resize_size: int = 224

    # --- Environment parameters ---
    domain_rand: bool = False
    headless: bool = False

    # --- Utils ---
    video_out_path: str = "data/ar4_mk3/videos"
    seed: int = 7


def run_on_env(args: Args) -> None:
    """Runs the policy on the AR4 MK3 environment."""
    logging.basicConfig(level=logging.INFO)
    np.random.seed(args.seed)
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    possible_model_paths = [
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
    model_path = next(
        (os.path.abspath(p) for p in possible_model_paths if os.path.exists(p)), None
    )
    if model_path is None:
        logging.error("Could not find AR4 MK3 model file.")
        return

    if args.domain_rand:
        domain_rand_config, object_color, target_color = (
            generate_random_domain_rand_config()
        )
        prompt = (
            f"Pick the {object_color} block and place it on the {target_color} target."
        )
    else:
        domain_rand_config = None
        prompt = args.prompt

    env_config = Ar4Mk3EnvConfig(
        model_path=model_path,
        reward_type="sparse",
        use_eef_control=False,  # Policy outputs joint positions
        domain_rand=domain_rand_config,  # Add domain rand config if needed
        absolute_state_actions=True,
        include_images_in_obs=True,
    )
    env = Ar4Mk3PickAndPlaceEnv(
        render_mode="rgb_array",
        config=env_config,
    )

    total_episodes, total_successes = 0, 0
    for episode_idx in range(args.num_episodes):
        logging.info(f"\nStarting episode {episode_idx + 1}/{args.num_episodes}")
        logging.info(f"Task: {prompt}")

        obs, info = env.reset(seed=args.seed + episode_idx)
        action_plan = collections.deque()
        replay_images = []
        done = False

        # Get initial joint positions for dummy action
        joint_names = ARM_JOINT_NAMES + [GRIPPER_JOINT_NAME]
        qpos_indices = [env.model.joint(name).qposadr[0] for name in joint_names]
        initial_qpos = env.data.qpos[qpos_indices]
        gripper_qpos_addr = env.model.joint(GRIPPER_JOINT_NAME).qposadr[0]
        arm_qpos_indices = [
            env.model.joint(name).qposadr[0] for name in ARM_JOINT_NAMES
        ]

        for t in range(args.max_episode_steps):
            try:
                if t < args.num_steps_wait:
                    obs, _, done, _, info = env.step(initial_qpos)
                    continue

                # Get observations
                img = obs["default_camera_image"]
                gripper_img = obs["gripper_camera_image"]

                if not args.headless:
                    # Display the image
                    display_img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                    cv2.imshow("AR4 Mk3 Env", display_img)
                    cv2.waitKey(1)

                # Get arm and gripper joint positions
                arm_qpos = env.data.qpos[arm_qpos_indices]
                gripper_qpos = np.array([env.data.qpos[gripper_qpos_addr]])
                current_qpos = np.concatenate((arm_qpos, gripper_qpos))

                # Preprocess images (rotate 180 deg to match training data).
                # TODO: This may or may not be needed. Investigage.
                # img = np.ascontiguousarray(img[::-1, ::-1])
                # img = image_tools.convert_to_uint8(
                #     image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                # )
                img = image_tools.convert_to_uint8(img)
                gripper_img = image_tools.convert_to_uint8(gripper_img)
                replay_images.append(img)

                if not action_plan:
                    # Prepare observations dict
                    element: dict[str, Any] = {
                        "image": img,
                        "gripper_image": gripper_img,
                        "state": current_qpos,
                        "prompt": prompt,
                    }
                    action_chunk = client.infer(element)["actions"]
                    action_plan.extend(action_chunk[: args.replan_steps])

                action = action_plan.popleft()
                obs, _, done, _, info = env.step(action)

                if done:
                    break
            except Exception as e:
                logging.error(f"Caught exception: {e}", exc_info=True)
                break

        if done:
            total_successes += 1
        total_episodes += 1

        # Save replay video
        suffix = "success" if done else "failure"
        task_segment = prompt.replace(" ", "_")[:50]
        video_path = (
            pathlib.Path(args.video_out_path)
            / f"rollout_{episode_idx}_{task_segment}_{suffix}.mp4"
        )
        imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=10)
        logging.info(f"Saved video to {video_path}")

        logging.info(f"Episode finished. Success: {done}")
        if total_episodes > 0:
            success_rate = total_successes / total_episodes * 100
            logging.info(
                f"Success rate so far: {success_rate:.1f}% ({total_successes}/{total_episodes})"
            )

    env.close()
    if not args.headless:
        cv2.destroyAllWindows()
    logging.info("Evaluation finished.")
    if total_episodes > 0:
        final_rate = total_successes / total_episodes * 100
        logging.info(f"Final success rate: {final_rate:.1f}%")


if __name__ == "__main__":
    run_on_env(tyro.cli(Args))
