import argparse
import dataclasses
import enum
import os
import tempfile
from typing import Callable

import gymnasium as gym
import imageio
import jax
import jax.numpy as jnp
import mlflow
import optax
import tqdm

from aera.autonomous.basic_rl.reinforce import reinforce_config, reinforce_policy
from aera.autonomous.basic_rl.reinforce.common import (
    Batch,
    _init_network_params,
    _apply_dropout,
)


def _get_activation_fn(name: str) -> Callable[[jnp.ndarray], jnp.ndarray]:
    if name == "relu":
        return jax.nn.relu
    raise ValueError(f"Unknown activation function: {name}")


def _get_observation(observation: jnp.ndarray) -> jnp.ndarray:
    if hasattr(observation, "__getitem__") and "observation" in observation:
        return observation["observation"]
    return observation


def unscale_actions(scaled_action: jnp.ndarray, env) -> jnp.ndarray:
    action_space = (
        env.single_action_space
        if hasattr(env, "single_action_space")
        else env.action_space
    )
    low, high = action_space.low, action_space.high
    return low + (0.5 * (scaled_action + 1.0) * (high - low))


def _log_train_metrics(
    step: int,
    aux: dict,
    advantate: jnp.ndarray,
    infos: list,
    train_episode_returns: list,
    train_episode_lengths: list,
    profile: bool = False,
):
    metrics = {f"train/{k}": v for k, v in aux.items()}
    metrics["train/advantage_mean"] = jnp.mean(advantate)
    metrics["train/advantage_std"] = jnp.std(advantate)
    if infos:
        for key in infos[0].keys():
            try:
                values = [info[key] for info in infos]
                metrics[f"train/{key}"] = jnp.mean(jnp.array(values))
            except (TypeError, KeyError, ValueError):
                pass

    if train_episode_returns:
        metrics["train/avg_return"] = jnp.mean(jnp.array(train_episode_returns))
        metrics["train/avg_ep_len"] = jnp.mean(jnp.array(train_episode_lengths))
        train_episode_returns.clear()
        train_episode_lengths.clear()

    if not profile:
        mlflow.log_metrics({k: v.item() for k, v in metrics.items()}, step=step)


def _gather_batch(
    key: jnp.ndarray,
    policy_state: reinforce_policy.ReinforcePolicyState,
    env: gym.vector.VectorEnv,
    batch_size: int,
    num_envs: int,
    observation: jnp.ndarray,
    current_episode_return: jnp.ndarray,
    current_episode_length: jnp.ndarray,
    train_episode_returns: list,
    train_episode_lengths: list,
) -> tuple[jnp.ndarray, Batch, list, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    observations = []
    actions = []
    masks = []
    rewards = []
    infos = []
    num_steps = batch_size // num_envs
    for _ in range(num_steps):
        observation = _get_observation(observation)
        key, action_seed = jax.random.split(key)
        action, _ = reinforce_policy.sample_action(
            observation,
            policy_state,
            action_seed,
        )
        new_observation, reward, terminated, truncated, info = env.step(
            unscale_actions(action, env)
        )

        observations.append(observation)
        actions.append(action)
        masks.append(~(terminated | truncated))
        rewards.append(reward)
        infos.append(info)

        current_episode_return += reward
        current_episode_length += 1

        done = terminated | truncated
        if jnp.any(done):
            for i, d in enumerate(done):
                if d:
                    train_episode_returns.append(current_episode_return[i])
                    train_episode_lengths.append(current_episode_length[i])
        current_episode_return = current_episode_return * (1 - done)
        current_episode_length = current_episode_length * (1 - done)

        observation = new_observation

    batch = Batch(
        observations=jnp.concatenate(observations),
        actions=jnp.concatenate(actions),
        masks=jnp.array(masks).reshape(-1, 1),
        rewards=jnp.array(rewards).reshape(-1, 1),
    )
    return (
        key,
        batch,
        infos,
        observation,
        current_episode_return,
        current_episode_length,
    )


def eval_policy(
    step: int,
    config: reinforce_config.Config,
    eval_env: gym.Env,
    policy_state: reinforce_policy.ReinforcePolicyState,
) -> None:
    episode_returns = []
    episode_lengths = []
    frames = []
    episode_infos = []
    all_actions = []
    all_observations = []

    for episode_idx in range(config.eval_num_episodes):
        observation, _ = eval_env.reset()
        done, truncated = False, False
        total_reward = 0.0
        ep_len = 0
        info = {}
        episode_actions = []
        episode_observations = []

        while not (done or truncated):
            if config.eval_render:
                frame = eval_env.render()
                if frame is not None:
                    frames.append(frame)

            observation = _get_observation(observation)
            episode_observations.append(observation)

            action, _ = reinforce_policy.sample_action(
                observation, policy_state, temperature=0.0
            )
            episode_actions.append(action)

            observation, reward, done, truncated, info = eval_env.step(
                unscale_actions(action, eval_env)
            )
            total_reward += reward  # type: ignore
            ep_len += 1

        episode_returns.append(total_reward)
        episode_lengths.append(ep_len)
        episode_infos.append(info)
        all_actions.append(jnp.array(episode_actions))
        all_observations.append(jnp.array(episode_observations))

    metrics = {
        "eval/avg_return": jnp.mean(jnp.array(episode_returns)),
        "eval/sum_return": jnp.sum(jnp.array(episode_returns)),
        "eval/avg_ep_len": jnp.mean(jnp.array(episode_lengths)),
    }
    if all_actions:
        avg_action_per_dim = jnp.mean(jnp.concatenate(all_actions), axis=0)
        for i, avg_action in enumerate(avg_action_per_dim):
            metrics[f"eval/avg_action_dim_{i}"] = avg_action

    if episode_infos:
        for key in episode_infos[0].keys():
            try:
                values = [info[key] for info in episode_infos]
                metrics[f"eval/{key}"] = jnp.mean(jnp.array(values))
            except (TypeError, KeyError, ValueError):
                # This can happen if info dicts are not consistent or values are not numeric
                pass

    if not config.profile:
        mlflow.log_metrics({k: v.item() for k, v in metrics.items()}, step=step)

    if frames and not config.profile:
        with tempfile.TemporaryDirectory() as tmpdir:
            video_path = os.path.join(tmpdir, f"eval_video_step_{step}.mp4")
            imageio.mimsave(video_path, frames, fps=30)
            mlflow.log_artifact(video_path, artifact_path="videos")


@dataclasses.dataclass(frozen=True)
class ValueFunctionState:
    weights: list[tuple[jnp.ndarray, jnp.ndarray]]
    optimizer: optax.GradientTransformationExtraArgs
    opt_state: optax.OptState
    dropout_rate: float = 0.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu

    @staticmethod
    def create(
        hidden_dims: tuple[int, ...],
        obs_dim: int,
        key: jnp.ndarray,
        optimizer: optax.GradientTransformationExtraArgs,
        dropout_rate: float = 0.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    ):
        weights = _init_network_params(key, (obs_dim,) + hidden_dims + (1,))
        opt_state = optimizer.init(weights)
        return ValueFunctionState(
            weights=weights,
            optimizer=optimizer,
            opt_state=opt_state,
            dropout_rate=dropout_rate,
            activation_fn=activation_fn,
        )


def _value_fn(
    obs: jnp.ndarray,
    weights: list[tuple[jnp.ndarray, jnp.ndarray]],
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
    dropout_key=jax.random.PRNGKey(0),
    dropout_rate: float = 0.0,
    activate_final: bool = False,
) -> jnp.ndarray:
    x = obs
    dropout_keys = jax.random.split(dropout_key, len(weights) + int(activate_final))
    for i, (w, b) in enumerate(weights[:-1]):
        x = _apply_dropout(
            activation_fn(jnp.dot(x, w) + b), dropout_rate, dropout_keys[i]
        )
    x = jnp.dot(x, weights[-1][0]) + weights[-1][1]
    if activate_final:
        return _apply_dropout(activation_fn(x), dropout_rate, dropout_keys[-1])
    return x


def _update_value_function(
    obs: jnp.ndarray,
    targets: jnp.ndarray,
    state: ValueFunctionState,
    dropout_key: jnp.ndarray,
):
    def loss_fn(
        weights: list[tuple[jnp.ndarray, jnp.ndarray]],
    ):
        values = _value_fn(
            obs, weights, state.activation_fn, dropout_key, state.dropout_rate
        )
        loss = jnp.mean((targets - values) ** 2)
        return loss, {
            "value_loss": loss,
            "value_pred": values.mean(),
        }

    targets = jax.lax.stop_gradient(targets)
    grad, aux = jax.grad(loss_fn, has_aux=True)(state.weights)
    updates, opt_state = state.optimizer.update(grad, state.opt_state, state.weights)
    new_state = dataclasses.replace(
        state, weights=optax.apply_updates(state.weights, updates), opt_state=opt_state
    )
    return aux, new_state


def _calculate_gae(
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    values: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> jnp.ndarray:
    def gae_fn(gae, xs):
        delta, mask = xs
        gea = delta + gamma * gae_lambda * gae * mask
        return gea, gea

    deltas = jax.vmap(lambda r, v, nx, m: r + gamma * nx * m - v)(
        rewards, values[:-1], values[1:], masks
    )

    return jax.lax.scan(
        gae_fn, jnp.zeros(rewards.shape[1:]), (deltas, masks), reverse=True
    )[1]


def _calculate_advantage(
    use_gae: bool,
    reward_to_go: jnp.ndarray,
    values: jnp.ndarray,
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    next_value: jnp.ndarray,
    num_steps: int,
    num_envs: int,
    gamma: float,
    gae_lambda: float,
) -> jnp.ndarray:
    if use_gae:
        values_by_env = values.reshape((num_steps, num_envs, -1))
        advantage = _calculate_gae(
            rewards,
            masks,
            jnp.concatenate([values_by_env, next_value.reshape(1, -1, 1)]),
            gamma,
            gae_lambda,
        ).reshape((-1, 1))
    else:
        advantage = reward_to_go - values
    return (advantage - advantage.mean()) / (advantage.std() - 1e-8)


class Trainer:
    def __init__(self, config: reinforce_config.Config) -> None:
        self.config = config
        self.env = gym.make_vec(
            config.env_name,
            num_envs=config.num_envs,
            max_episode_steps=config.ep_len,
        )
        self.env.reset()
        try:
            self.eval_env = gym.make(
                config.env_name,
                render_mode=config.env_render_mode,
                width=config.env_render_width,
                height=config.env_render_height,
                max_episode_steps=config.ep_len,
            )
        except Exception as _:
            self.eval_env = gym.make(
                config.env_name,
                render_mode=config.env_render_mode,
                max_episode_steps=config.ep_len,
            )

        self.eval_env.reset()

        action_shape = self.env.single_action_space.shape
        observation_shape = self.env.single_observation_space.shape
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0), optax.adam(config.policy_lr)
        )
        self.key, policy_seed, value_seed = jax.random.split(
            jax.random.PRNGKey(seed=config.seed), 3
        )
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

        self.value_state = ValueFunctionState.create(
            hidden_dims=config.value_hidden_dims,
            obs_dim=observation_shape[0],  # type: ignore
            key=value_seed,
            optimizer=optax.adam(config.value_lr),
            dropout_rate=config.value_dropout_rate,
            activation_fn=_get_activation_fn(config.value_activation_fn),
        )

    def train(self) -> None:
        experiment = mlflow.set_experiment(self.config.env_name)
        print("Starting Experiment")
        print(experiment)
        with mlflow.start_run():
            mlflow.set_tags({"algorithm": "reinforce"})
            mlflow.log_params(dataclasses.asdict(self.config))
            observation, _ = self.env.reset()
            train_episode_returns = []
            train_episode_lengths = []
            current_episode_return = jnp.zeros(self.config.num_envs)
            current_episode_length = jnp.zeros(self.config.num_envs, dtype=jnp.int32)
            for i in tqdm.tqdm(
                range(self.config.max_steps), smoothing=0.1, disable=self.config.profile
            ):
                (
                    self.key,
                    batch,
                    infos,
                    observation,
                    current_episode_return,
                    current_episode_length,
                ) = _gather_batch(
                    self.key,
                    self.policy_state,
                    self.env,
                    self.config.batch_size,
                    self.config.num_envs,
                    observation,
                    current_episode_return,
                    current_episode_length,
                    train_episode_returns,
                    train_episode_lengths,
                )

                next_value = _value_fn(
                    _get_observation(observation),
                    self.value_state.weights,
                    self.value_state.activation_fn,
                )

                def reward_to_go_step(carry, xs):
                    reward, mask = xs
                    carry = reward + self.config.gamma * carry * mask
                    return carry, carry

                num_steps = self.config.batch_size // self.config.num_envs
                rewards = batch.rewards.reshape((num_steps, self.config.num_envs, -1))
                masks = batch.masks.reshape((num_steps, self.config.num_envs, -1))

                _, reward_to_go_by_env = jax.lax.scan(
                    reward_to_go_step,
                    next_value,
                    (rewards, masks),
                    reverse=True,
                )
                reward_to_go = reward_to_go_by_env.reshape((-1, 1))
                values = _value_fn(
                    batch.observations,
                    self.value_state.weights,
                    self.value_state.activation_fn,
                )
                values_by_env = values.reshape((num_steps, self.config.num_envs, -1))

                bootstrap_targets = jax.vmap(lambda r, v: r + self.config.gamma * v)(
                    rewards,
                    jnp.concatenate([values_by_env, next_value.reshape(1, -1, 1)])[1:],
                ).reshape((-1, 1))

                advantage = _calculate_advantage(
                    use_gae=self.config.use_gae,
                    reward_to_go=reward_to_go,
                    values=values,
                    rewards=rewards,
                    masks=masks,
                    next_value=next_value,
                    num_steps=num_steps,
                    num_envs=self.config.num_envs,
                    gamma=self.config.gamma,
                    gae_lambda=self.config.gae_lambda,
                )
                self.key, dropout_key_policy, dropout_key_value = jax.random.split(
                    self.key, 3
                )

                aux, self.policy_state = reinforce_policy.update_policy(
                    self.policy_state, batch, advantage, dropout_key_policy
                )
                val_aux, self.value_state = _update_value_function(
                    batch.observations,
                    # reward_to_go,
                    bootstrap_targets,
                    self.value_state,
                    dropout_key_value,
                )
                aux = aux | val_aux

                _log_train_metrics(
                    i,
                    aux,
                    advantage,
                    infos,
                    train_episode_returns,
                    train_episode_lengths,
                    self.config.profile,
                )

                if (
                    self.config.eval_step_interval > 0
                    and i % self.config.eval_step_interval == 0
                ):
                    eval_policy(i, self.config, self.eval_env, self.policy_state)


def main():
    """Creates a default config and runs the trainer."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable profiling mode (disables logging, etc.)",
    )
    args, _ = parser.parse_known_args()

    config = reinforce_config.Config()
    if args.profile:
        config.profile = True
        config.max_steps = 1000
        config.eval_step_interval = 0

    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
