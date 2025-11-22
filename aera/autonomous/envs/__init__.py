from gymnasium import register


def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""
    register(
        id="Ar4Mk3PickAndPlaceEnv-v1",
        entry_point="aera.autonomous.envs.ar4_mk3_pick_and_place:Ar4Mk3PickAndPlaceEnv",
    )


register_robotics_envs()
