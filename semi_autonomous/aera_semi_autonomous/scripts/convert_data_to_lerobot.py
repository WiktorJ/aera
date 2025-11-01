"""
Script to convert data collected by TrajectoryDataCollector to LeRobot format.

This script reads the episode data saved in JSON format, processes images, states,
and actions, and converts it into a format compatible with LeRobot for training RL policies.

Usage:
python semi_autonomous/aera_semi_autonomous/scripts/convert_data_to_lerobot.py --data_dir /path/to/your/rl_training_data

The resulting dataset will be saved to `rl_training_data_lerobot` by default.
"""

import json
import shutil
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import tyro
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main(data_dir: str, output_dir: Optional[str] = None):
    """
    Main function to convert trajectory data to LeRobot format.

    Args:
        data_dir: Path to the root directory containing episode data folders.
        output_dir: Path to save the converted dataset. Defaults to `{data_dir}_lerobot`.
    """
    if output_dir is None:
        output_dir = f"{data_dir}_lerobot"

    # Clean up any existing dataset in the output directory
    output_path = Path(output_dir)
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
    features = {
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
    }
    dataset = LeRobotDataset(
        repo_id=output_path.name,
        root=output_path.parent,
        image_writer_threads=10,
        image_writer_processes=5,
    )
    dataset.info["robot_type"] = "ar4_mk3"
    dataset.info["fps"] = fps
    dataset.info["features"] = features
    dataset.save_info()

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


if __name__ == "__main__":
    tyro.cli(main)
