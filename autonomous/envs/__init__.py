from gymnasium import register
import os


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""
    
    # Get the path to the MuJoCo scene file
    current_dir = os.path.dirname(__file__)
    model_path = os.path.join(current_dir, "..", "simulation", "mujoco", "ar4_mk3", "scene.xml")
    
    for reward_type in ["sparse", "dense"]:
        suffix = "Dense" if reward_type == "dense" else ""
        kwargs = {
            "model_path": model_path,
            "reward_type": reward_type,
        }
        register(
            id=f"Ar4Mk3PickAndPlace{suffix}Env-v1",
            entry_point="autonomous.envs.ar4_mk3_pick_and_place:Ar4Mk3PickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )


register_robotics_envs()
