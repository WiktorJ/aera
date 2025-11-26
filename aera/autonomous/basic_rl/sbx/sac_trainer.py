import sys
from typing import Any, Dict, Tuple, Union

import gymnasium as gym
import mlflow
import numpy as np
from sbx import SAC
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv

import aera.autonomous.envs.ar4_mk3_pick_and_place
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3EnvConfig


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """

    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


loggers = Logger(
    folder=None,
    output_formats=[HumanOutputFormat(sys.stdout), MLflowOutputFormat()],
)

config = Ar4Mk3EnvConfig(
    use_eef_control=True,
    reward_type="dense",
)

env_name = "Ar4Mk3PickAndPlaceEnv-v1"
mlflow.set_experiment(f"{env_name}-SAC")
env = DummyVecEnv(
    [lambda: gym.make(env_name, render_mode="rgb_array", max_episode_steps=100)]
)
env = VecVideoRecorder(
    env,
    f"videos/{env_name}.mp4",
    record_video_trigger=lambda x: x == 0,
    video_length=100,
)

with mlflow.start_run():
    model = SAC("MultiInputPolicy", env, verbose=2)
    model.set_logger(loggers)
    model.learn(total_timesteps=10_000, progress_bar=True, log_interval=1)

vec_env = model.get_env()
if vec_env is None:
    raise RuntimeError("VecEnv is None")
obs = vec_env.reset()
for _ in range(1000):
    vec_env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

vec_env.close()
