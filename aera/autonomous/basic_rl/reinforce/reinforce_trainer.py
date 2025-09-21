from typing import Callable

import tqdm
import gymnasium as gym
import jax
import jax.numpy as jnp
import optax

from aera.autonomous.basic_rl.reinforce import reinforce_config, reinforce_policy
from aera.autonomous.basic_rl.reinforce.common import Batch


def _get_activation_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation function: {name}")


def _get_observation(observation: jnp.ndarray) -> jnp.ndarray:
    if hasattr(observation, "__getitem__") and "observation" in observation:
        return observation["observation"]
    return observation


class Trainer:
    def __init__(self, config: reinforce_config.Config) -> None:
        self.config = config
        self.env = gym.make(
            config.env_name,
            render_mode=config.env_render_mode,
            width=config.env_render_width,
            height=config.env_render_height,
            max_episode_steps=config.ep_len,
        )
        self.env.reset()
        self.eval_env = gym.make(
            config.env_name,
            render_mode=config.env_render_mode,
            width=config.env_render_width,
            height=config.env_render_height,
            max_episode_steps=config.ep_len,
        )
        self.eval_env.reset()

        action_shape = self.env.action_space.shape
        observation_shape = self.env.observation_space.shape
        optimizer = optax.adamw(config.policy_lr)
        self.key, policy_seed = jax.random.split(jax.random.PRNGKey(seed=config.seed))
        self.policy_state = reinforce_policy.ReinforcePolicyState.create(
            hidden_dims=config.policy_hidden_dims,
            action_dim=action_shape[0],  # type: ignore
            obs_dim=observation_shape[0],  # type: ignore
            key=policy_seed,
            optimizer=optimizer,
            obs_dependent_std=config.policy_obs_dependent_std,
            tanh_squash_dist=config.policy_tanh_squash_dist,
            log_std_min=config.policy_log_std_min,
            log_std_max=config.policy_log_std_max,
            dropout_rate=config.policy_dropout_rate,
            training_temperature=config.policy_temperature,
            activation_fn=_get_activation_fn(config.policy_activation_fn),
        )

    def train(self) -> None:
        observation, _ = self.env.reset()
        for i in tqdm.tqdm(range(self.config.max_steps), smoothing=0.01):
            observations = []
            actions = []
            masks = []
            rewards = []
            for j in range(self.config.batch_size):
                observation = _get_observation(observation)
                self.key, action_seed = jax.random.split(self.key)
                action, _ = reinforce_policy.sample_action(
                    observation,
                    self.policy_state,
                    action_seed,
                )
                new_observation, reward, done, truncated, info = self.env.step(action)

                observations.append(observation)
                actions.append(action)
                masks.append(not (done or truncated))
                rewards.append(reward)

                if done or truncated:
                    observation, _ = self.env.reset()
                else:
                    observation = new_observation

            batch = Batch(
                observations=jnp.array(observations),
                actions=jnp.array(actions),
                masks=jnp.array(masks).reshape(-1, 1),
                rewards=jnp.array(rewards).reshape(-1, 1),
            )

            # Calculate reward to go for each state, taking into account masks
            def reward_to_go_step(carry, xs):
                reward, mask = xs
                carry = reward + self.config.gamma * carry * mask
                return carry, carry

            # We process rewards and masks in reverse order
            _, reward_to_go_rev = jax.lax.scan(
                reward_to_go_step,
                0.0,
                (batch.rewards, batch.masks),
                reverse=True,
            )
            # And flip the result back
            advantate = jnp.flip(reward_to_go_rev, axis=0)

            # Normalize advantage
            advantate = advantate - advantate.mean()

            aux, self.policy_state = reinforce_policy.update_reinforce_policy(
                self.policy_state,
                batch,
                advantate,
            )

            if (
                self.config.eval_step_interval > 0
                and i % self.config.eval_step_interval == 0
            ):
                self.eval()

    def eval(self) -> tuple[dict, list]:
        episode_returns = []
        episode_lengths = []
        frames = []
        episode_infos = []

        for _ in range(self.config.eval_num_episodes):
            observation, _ = self.eval_env.reset()
            done, truncated = False, False
            total_reward = 0.0
            ep_len = 0
            info = {}
            while not (done or truncated):
                if self.config.eval_render:
                    frame = self.eval_env.render()
                    if frame is not None:
                        frames.append(frame)

                observation = _get_observation(observation)
                action, _ = reinforce_policy.sample_action(
                    observation, self.policy_state, temperature=0.0
                )
                observation, reward, done, truncated, info = self.eval_env.step(action)
                total_reward += reward  # type: ignore
                ep_len += 1

            episode_returns.append(total_reward)
            episode_lengths.append(ep_len)
            episode_infos.append(info)

        metrics = {
            "eval/avg_return": jnp.mean(jnp.array(episode_returns)),
            "eval/sum_return": jnp.sum(jnp.array(episode_returns)),
            "eval/avg_ep_len": jnp.mean(jnp.array(episode_lengths)),
        }

        if episode_infos:
            for key in episode_infos[0].keys():
                try:
                    values = [info[key] for info in episode_infos]
                    metrics[f"eval/{key}"] = jnp.mean(jnp.array(values))
                except (TypeError, KeyError, ValueError):
                    # This can happen if info dicts are not consistent or values are not numeric
                    pass

        return metrics, frames
