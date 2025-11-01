"""
Script to convert data collected by TrajectoryDataCollector to LeRobot format.

This script reads the episode data saved in JSON format, processes images, states,
and actions, and converts it into a format compatible with LeRobot for training RL policies.

Usage:
python semi_autonomous/aera_semi_autonomous/scripts/convert_data_to_lerobot.py --data_dir /path/to/your/rl_training_data

To push the dataset to the Hugging Face Hub, use the --push_to_hub flag:
python semi_autonomous/aera_semi_autonomous/scripts/convert_data_to_lerobot.py --data_dir /path/to/your/rl_training_data --push_to_hub

Note: to run the script, you need to install lerobot, tyro and opencv-python:
`pip install lerobot tyro opencv-python`

The resulting dataset will be saved to the $HF_LEROBOT_HOME directory.
"""

import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

REPO_NAME = "aera-bot/ar4_mk3_pick_and_place"


def main(data_dir: str, *, push_to_hub: bool = False):
    """
    Main function to convert trajectory data to LeRobot format.

    Args:
        data_dir: Path to the root directory containing episode data folders.
        push_to_hub: If True, push the converted dataset to the Hugging Face Hub.
    """
    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

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

    # Create LeRobot dataset and define features
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="ar4_mk3",
        fps=fps,
        features={
            "image": {
                "dtype": "image",
                "shape": (height, width, 3),
                "names": ["height", "width", "channel"],
            },
            "depth_image": {
                "dtype": "image",
                "shape": (height, width, 1),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),  # 6 arm joints + 2 gripper joints
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # 6 arm joints + 2 gripper joints
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Loop over raw episode data and write to the LeRobot dataset
    for episode_dir in episode_dirs:
        print(f"Processing episode: {episode_dir.name}")
        with open(episode_dir / "episode_data.json") as f:
            episode_data = json.load(f)

        for step in episode_data["trajectory_data"]:
            # Decode RGB image
            rgb_bytes = bytes.fromhex(step["observations"]["rgb_image"])
            rgb_np = np.frombuffer(rgb_bytes, np.uint8)
            bgr_img = cv2.imdecode(rgb_np, cv2.IMREAD_COLOR)
            rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

            # Decode depth image
            depth_bytes = bytes.fromhex(step["observations"]["depth_image"])
            depth_array = np.frombuffer(depth_bytes, dtype=np.float32)
            depth_image = depth_array.reshape((height, width))
            depth_image = np.expand_dims(depth_image, axis=-1)

            # State is current joint positions
            state = np.array(
                step["observations"]["joint_state"]
                + step["observations"]["gripper_state"],
                dtype=np.float32,
            )

            # Action is the next set of joint positions
            action = np.array(
                step["action"]["joint_state"] + step["action"]["gripper_state"],
                dtype=np.float32,
            )

            dataset.add_frame(
                {
                    "image": rgb_img,
                    "depth_image": depth_image,
                    "state": state,
                    "actions": action,
                    "task": step["prompt"],
                    "is_first": step["is_first"],
                    "is_last": step["is_last"],
                    "is_terminal": step["is_terminal"],
                    "reward": step["default_reward"],
                }
            )
        dataset.save_episode()

    print(f"Finished converting dataset. Saved to {output_path}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        print(f"Pushing dataset to Hugging Face Hub: {REPO_NAME}")
        dataset.push_to_hub(
            tags=["ar4_mk3", "pick-and-place", "teleoperation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print("Push to hub complete.")


if __name__ == "__main__":
    tyro.cli(main)
