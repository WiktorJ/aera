from gymnasium import register


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    for suffix in ["", "Dense"]:
        kwargs = {
            "reward_type": suffix,
        }
        register(
            id=f"Ar4Mk3PickAndPlace{suffix}Env-v1",
            entry_point="gymnasium_robotics.envs.fetch.pick_and_place:MujocoFetchPickAndPlaceEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )


register_robotics_envs()
