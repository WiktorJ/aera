from gymnasium import register
from aera.autonomous.envs.ar4_mk3_config import Ar4Mk3EnvConfig
import os


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    current_dir = os.path.dirname(__file__)

    # --- Joint control envs ---
    model_path = os.path.join(
        current_dir, "..", "simulation", "mujoco", "ar4_mk3", "scene.xml"
    )
    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        config = Ar4Mk3EnvConfig(model_path=model_path, reward_type=reward_type)
        kwargs = {
            # "model_path": model_path,
            # "reward_type": reward_type,
            "config": config
        }
        register(
            id=f"Ar4Mk3PickAndPlace{suffix}Env-v1",
            entry_point="aera.autonomous.envs.ar4_mk3_pick_and_place:Ar4Mk3PickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

    # --- EEF control envs ---
    model_path_eef = os.path.join(
        current_dir, "..", "simulation", "mujoco", "ar4_mk3", "scene_eef.xml"
    )
    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "model_path": model_path_eef,
            "reward_type": reward_type,
            "use_eef_control": True,
        }
        register(
            id=f"Ar4Mk3PickAndPlaceEef{suffix}Env-v1",
            entry_point="aera.autonomous.envs.ar4_mk3_pick_and_place:Ar4Mk3PickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )


register_robotics_envs()
