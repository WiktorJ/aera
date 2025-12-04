"""
Script to convert data collected by TrajectoryDataCollector to LeRobot format.

This script reads the episode data saved in JSON format, processes images, states,
and actions, and converts it into a format compatible with LeRobot for training RL policies.

Usage:
python semi_autonomous/aera_semi_autonomous/scripts/convert_data_to_lerobot.py --data_dir /path/to/your/rl_training_data

The resulting dataset will be saved to `rl_training_data_lerobot` by default.
"""

import json
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from torch.utils import data
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset, CODEBASE_VERSION
from lerobot.utils.constants import HF_LEROBOT_HOME
from huggingface_hub import HfApi


def _process_rgb_image(image_hex: str) -> np.ndarray:
    """Process RGB image bytes and return as a NumPy array."""
    image_bytes = bytes.fromhex(image_hex)
    rgb_np = np.frombuffer(image_bytes, np.uint8)
    bgr_img = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)


def _process_depth_image(image_hex: str) -> np.ndarray:
    """Process depth image bytes and return as a NumPy array."""
    image_bytes = bytes.fromhex(image_hex)
    depth_array = np.frombuffer(image_bytes, dtype=np.float32)
    depth_image = depth_array.reshape((height, width))
    return np.expand_dims(depth_image, axis=-1)


def main(
    data_dir: str,
    output_dir: str | Path | None = None,
    frame_skip: int = 1,
    squeeze_gripper: bool = True,
    push_to_hub: bool = False,
):
    """
    Main function to convert trajectory data to LeRobot format.

    Args:
        data_dir: Path to the root directory containing episode data folders.
        output_dir: Path to save the converted dataset. Defaults to `{data_dir}_lerobot`.
        frame_skip: How many steps to skip between each observation. Defaults to 1 (no skip).
        squeeze_gripper: If True, squeeze the 2D gripper state/action into 1D.
        push_to_hub: If True, push the dataset to the Hugging Face Hub.
    """
    if output_dir is None:
        output_path = HF_LEROBOT_HOME / data_dir
    else:
        output_path = Path(output_dir)

    episode_dirs = sorted([p for p in Path(data_dir).iterdir() if p.is_dir()])
    if not episode_dirs:
        print(f"No episode directories found in {data_dir}")
        return

    # Load metadata from the first episode to define dataset features
    with open(episode_dirs[0] / "episode_data.json") as f:
        first_episode_data = json.load(f)
    metadata = first_episode_data["metadata"]
    height = metadata["image_height"]
    width = metadata["image_width"]

    # Calculate average FPS from all episodes
    total_points = 0
    total_duration = 0
    for episode_dir in episode_dirs:
        with open(episode_dir / "episode_data.json") as f:
            episode_data = json.load(f)
        total_points += len(episode_data["trajectory_data"])
        total_duration += episode_data.get("duration", 0)

    fps = round(total_points / total_duration) if total_duration > 0 else 30
    print(f"Calculated average FPS: {fps}")

    action_dim = 7 if squeeze_gripper else 8

    # Create LeRobot dataset and define features
    features = {
        "image": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        "gripper_image": {
            "dtype": "image",
            "shape": (height, width, 3),
            "names": ["height", "width", "channel"],
        },
        # "depth_image": {
        #     "dtype": "image",
        #     "shape": (height, width, 1),
        #     "names": ["height", "width", "channel"],
        # },
        "state": {
            "dtype": "float32",
            "shape": (action_dim,),  # 6 arm joints + 1/2 gripper joints
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (action_dim,),  # 6 arm joints + 1/2 gripper joints
            "names": ["actions"],
        },
        "granural_prompt": {
            "dtype": "string",
            "shape": (1,),
            "names": ["granular_prompt"],
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=f"Purple69/{output_path.name}",
        robot_type="AR4_MK3",
        features=features,
        fps=fps,
        image_writer_processes=5,
        image_writer_threads=10,
    )

    # Loop over raw episode data and write to the LeRobot dataset
    processed_prompts = set()
    for episode_dir in episode_dirs:
        print(f"Processing episode: {episode_dir.name}")
        with open(episode_dir / "episode_data.json") as f:
            episode_data = json.load(f)

        trajectory = episode_data["trajectory_data"]
        if not trajectory:
            continue

        for i in range(0, len(trajectory), frame_skip):
            step = trajectory[i]
            # Decode RGB image
            default_image = _process_rgb_image(
                step["observations"]["rgb_images"]["default"]
            )
            gripper_image = _process_rgb_image(
                step["observations"]["rgb_images"]["gripper_camera"]
            )
            # default_depth_image = _process_depth_image(
            #     step["observations"]["depth_images"]["default"]
            # )
            # gripper_depth_image = _process_depth_image(
            #     step["observations"]["depth_images"]["gripper_camera"]
            # )
            # State is current joint positions
            gripper_state = step["observations"]["gripper_state"]
            gripper_action = step["action"]["gripper_state"]
            if squeeze_gripper:
                gripper_state = [gripper_state[0]]
                gripper_action = [gripper_action[0]]
            state = np.array(
                step["observations"]["joint_state"] + gripper_state,
                dtype=np.float32,
            )
            action = np.array(
                step["action"]["joint_state"] + gripper_action,
                dtype=np.float32,
            )

            processed_prompts.add(step["prompt"])

            dataset.add_frame(
                {
                    "image": default_image,
                    "gripper_image": gripper_image,
                    # "depth_image": depth_image,
                    "state": state,
                    "actions": action,
                    "task": step["full_prompt"],
                    "granural_prompt": step["prompt"],
                    # "is_first": step["is_first"],
                    # "is_last": step["is_last"],
                    # "is_terminal": step["is_terminal"],
                    # "reward": step["default_reward"],
                }
            )
        dataset.save_episode()
    if push_to_hub:
        dataset.finalize()
        dataset.push_to_hub(
            tags=["aera", "ar4_mk3", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
    print(f"Finished converting dataset. Saved to {output_path}")
    print(f"Processed {processed_prompts} prompts")


if __name__ == "__main__":
    tyro.cli(main)
