import gymnasium as gym
import aera.autonomous.envs.ar4_mk3_pick_and_place
from aera.autonomous.envs.ar4_mk3_pick_and_place import Ar4Mk3EnvConfig

from sbx import SAC

config = Ar4Mk3EnvConfig(
    use_eef_control=True,
    reward_type="dense",
)

env = gym.make("Ar4Mk3PickAndPlaceEnv-v1", render_mode="human", max_episode_steps=100)

model = SAC("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000, progress_bar=True)

vec_env = model.get_env()
if vec_env is None:
    raise RuntimeError("VecEnv is None")
obs = vec_env.reset()
for _ in range(1000):
    vec_env.render()
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)

vec_env.close()
