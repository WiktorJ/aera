import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np

from aera.autonomous.basic_rl.reinforce import reinforce_config, reinforce_policy


class Trainer:
    def __init__(self, config: reinforce_config.Config) -> None:
        self.config = config
        self.env = gym.make(
            config.env_name,
            render_mode=config.env_render_mode,
            width=config.env_render_width,
            height=config.env_render_height,
            max_episode_steps=config.env_max_steps,
        )
        self.env.reset()
        self.eval_env = gym.make(
            config.env_name,
            render_mode=config.env_render_mode,
            width=config.env_render_width,
            height=config.env_render_height,
            max_episode_steps=config.env_max_steps,
        )
        self.eval_env.reset()

        self.policy = reinforce_policy.ReinforcePolicy()
        self.rng = jax.random.PRNGKey(seed=config.seed)
