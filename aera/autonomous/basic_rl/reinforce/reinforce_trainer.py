from typing import Callable

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax

from aera.autonomous.basic_rl.reinforce import reinforce_config, reinforce_policy


def _get_activation_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation function: {name}")


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

        action_shape = self.env.action_space.shape
        observation_shape = self.env.observation_space.shape
        optimizer = optax.adamw(config.policy_lr)

        self.policy = reinforce_policy.ReinforcePolicyState.create(
            hidden_dims=config.policy_hidden_dims,
            action_dim=action_shape[0],  # type: ignore
            obs_dim=observation_shape[0],  # type: ignore
            key=jax.random.PRNGKey(seed=config.seed),
            optimizer=optimizer,
            obs_dependent_std=config.policy_obs_dependent_std,
            tanh_squash_dist=config.policy_tanh_squash_dist,
            log_std_min=config.policy_log_std_min,
            log_std_max=config.policy_log_std_max,
            dropout_rate=config.policy_dropout_rate,
            temperature=config.policy_temperature,
            activation_fn=_get_activation_fn(config.policy_activation_fn),
        )
