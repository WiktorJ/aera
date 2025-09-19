from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import optax
import dataclasses

from aera.autonomous.basic_rl.reinforce.common import Batch


@dataclasses.dataclass(frozen=True)
class ReinforcePolicyState:
    hidden_dims: list[int]
    action_dim: int
    obs_dim: int
    key: jnp.ndarray
    oprimizer: optax.GradientTransformationExtraArgs
    trunk_weights: list[tuple[jnp.ndarray, jnp.ndarray]]
    mean_weights: tuple[jnp.ndarray, jnp.ndarray]
    log_std_weights: tuple[jnp.ndarray, jnp.ndarray]
    opt_state: optax.OptState
    obs_dependent_std: bool = False
    tanh_squash_dist: bool = False
    log_std_min: float = -20.0
    log_std_max: float = 2.0
    dropout_rate: float = 0.0
    temperature: float = 1.0
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu

    @staticmethod
    def create(
        hidden_dims: list[int],
        action_dim: int,
        obs_dim: int,
        key: jnp.ndarray,
        optimizer: optax.GradientTransformationExtraArgs,
        obs_dependent_std: bool = False,
        tanh_squash_dist: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    ):
        key, trunk_key, mean_key, log_std_key = jax.random.split(key, 4)

        trunk_weights = _init_network_params(trunk_key, [obs_dim] + hidden_dims)

        mean_weights = _init_network_params(mean_key, [hidden_dims[-1], action_dim])[0]

        log_std_weights = _init_network_params(
            log_std_key, [hidden_dims[-1], action_dim]
        )[0]

        opt_state = optimizer.init(
            (
                trunk_weights,
                mean_weights,
                log_std_weights,
            )
        )

        return ReinforcePolicyState(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            obs_dim=obs_dim,
            key=key,
            oprimizer=optimizer,
            trunk_weights=trunk_weights,
            mean_weights=mean_weights,
            log_std_weights=log_std_weights,
            opt_state=opt_state,
            obs_dependent_std=obs_dependent_std,
            tanh_squash_dist=tanh_squash_dist,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            dropout_rate=dropout_rate,
            temperature=temperature,
            activation_fn=activation_fn,
        )


def _random_layer_params(
    m: int, n: int, key: jnp.ndarray, scale: float = 1e-2
) -> tuple[jnp.ndarray, jnp.ndarray]:
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (m, n)), scale * jax.random.normal(
        b_key, (n,)
    )


def _init_network_params(
    key: jnp.ndarray, dims: Sequence[int]
) -> list[tuple[jnp.ndarray, jnp.ndarray]]:
    keys = jax.random.split(key, len(dims) - 1)
    layers = [
        _random_layer_params(m, n, k) for m, n, k in zip(dims[:-1], dims[1:], keys)
    ]
    return layers


def _sample_gaussian(
    key: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
    return mean + jnp.exp(log_std) * jax.random.normal(key, mean.shape)


def _gaussian_log_prob(
    value: jnp.ndarray, mean: jnp.ndarray, log_std: jnp.ndarray
) -> jnp.ndarray:
    variance = jnp.exp(2 * log_std) + 1e-6
    log2pi = jnp.log(2 * jnp.pi)
    log_prob = -0.5 * (((value - mean) ** 2) / variance) - log_std - 0.5 * log2pi
    return jnp.sum(log_prob)


def _tanh_jacobian_diag(x: jnp.ndarray) -> jnp.ndarray:
    return 1 - jnp.tanh(x) ** 2


def _tanh_squash_log_prob(
    value: jnp.ndarray, log_prob_base: jnp.ndarray
) -> jnp.ndarray:
    # The variable change formula states that
    # pY(y) = p_X(f^{-1}(y)) * det(d/dy f^{-1}(y))
    # in base variable x or p_Y(y) = p_X(x) * det(d/dx f(x)).
    # In this case f = tanh and f^{-1} = atanh.
    # Since we are interested in log_prob, the formula becomes
    # log(p_Y(y)) = log(p_X(x)) - log(det(d/dx f(x))) =
    # = log_probs_base - jacobian
    jacobian = _tanh_jacobian_diag(value)
    return log_prob_base - jnp.sum(jnp.log(jacobian))


def _tanh_squash(value: jnp.ndarray):
    return jnp.tanh(jnp.clip(value, -1.0 + 1e-6, 1.0 - 1e-6))


def _call_reinforce_policy(
    obs: jnp.ndarray,
    trunk_weights: list[tuple[jnp.ndarray, jnp.ndarray]],
    mean_weights: tuple[jnp.ndarray, jnp.ndarray],
    log_std_weights: tuple[jnp.ndarray, jnp.ndarray],
    obs_dependent_std: bool,
    tanh_squash_dist: bool,
    log_std_min: float,
    log_std_max: float,
    temperature: float,
    activation_fn: Callable[[jnp.ndarray], jnp.ndarray],
):
    for w, b in trunk_weights:
        obs = activation_fn(jnp.dot(obs, w) + b)
    x = jnp.tanh(obs)
    mean = jnp.dot(x, mean_weights[0]) + mean_weights[1]
    if obs_dependent_std:
        log_std = jnp.dot(x, log_std_weights[0]) + log_std_weights[1]
    else:
        log_std = jnp.zeros_like(mean)
    log_std = jnp.clip(log_std, log_std_min, log_std_max) * temperature
    if not tanh_squash_dist:
        mean = _tanh_squash(mean)
    if not tanh_squash_dist:
        return (
            lambda seed: _sample_gaussian(seed, mean, log_std),
            lambda sample: _gaussian_log_prob(sample, mean, log_std),
        )
    return (
        lambda seed: _tanh_squash(_sample_gaussian(seed, mean, log_std)),
        lambda sample: _tanh_squash_log_prob(
            sample, _gaussian_log_prob(sample, mean, log_std)
        ),
    )


def call_reinforce_policy(
    obs: jnp.ndarray,
    state: ReinforcePolicyState,
) -> tuple[jnp.ndarray, jnp.ndarray, ReinforcePolicyState]:
    action_fn, log_prob_fn = _call_reinforce_policy(
        obs,
        state.trunk_weights,
        state.mean_weights,
        state.log_std_weights,
        state.obs_dependent_std,
        state.tanh_squash_dist,
        state.log_std_min,
        state.log_std_max,
        state.temperature,
        state.activation_fn,
    )
    new_key, action_seed = jax.random.split(state.key)
    action = action_fn(action_seed)
    log_prob = log_prob_fn(action)
    state = dataclasses.replace(state, key=new_key)
    return action, log_prob, state


def update_reinforce_policy(
    state: ReinforcePolicyState,
    batch: Batch,
    advantate: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], ReinforcePolicyState]:
    def loss_fn(
        trunk_weights: list[tuple[jnp.ndarray, jnp.ndarray]],
        mean_weights: tuple[jnp.ndarray, jnp.ndarray],
        log_std_weights: tuple[jnp.ndarray, jnp.ndarray],
    ):
        _, log_prob_fn = _call_reinforce_policy(
            batch.observations,
            trunk_weights,
            mean_weights,
            log_std_weights,
            state.obs_dependent_std,
            state.tanh_squash_dist,
            state.log_std_min,
            state.log_std_max,
            state.temperature,
            state.activation_fn,
        )
        log_prob = log_prob_fn(batch.actions)
        loss = -(log_prob * advantate).mean()
        return loss, {
            "policy_loss": loss,
            "log_prob": log_prob.mean(),
        }

    grad, aux = jax.grad(loss_fn, has_aux=True, argnums=(0, 1, 2))(
        state.trunk_weights,
        state.mean_weights,
        state.log_std_weights,
    )
    updates, opt_state = state.oprimizer.update(grad, state.opt_state)
    trunk_weights, mean_weights, log_std_weights = optax.apply_updates(
        (state.trunk_weights, state.mean_weights, state.log_std_weights), updates
    )  # type: ignore
    new_state = dataclasses.replace(
        state,
        opt_state=opt_state,
        trunk_weights=trunk_weights,
        mean_weights=mean_weights,
        log_std_weights=log_std_weights,
    )
    return aux, new_state


# state = ReinforcePolicyState.create(
#     hidden_dims=[32, 32],
#     action_dim=1,
#     obs_dim=1,
#     key=jax.random.PRNGKey(0),
#     optimizer=optax.adam(1e-3),
#     obs_dependent_std=True,
# )
#
# ran_key1, ran_key2, ran_key3 = jax.random.split(jax.random.PRNGKey(0), 3)
#
# batch = Batch(
#     observations=jax.random.normal(ran_key1, (10, 1)),
#     actions=jax.random.normal(ran_key2, (10, 1)),
#     rewards=jnp.ones((10, 1)),
#     next_observations=jnp.ones((10, 1)),
#     masks=jnp.ones((10, 1)),
# )
# advantage = jax.random.normal(ran_key3, (10, 1)) * 0.1 + 0.5
#
# aux, state = update_reinforce_policy(state, batch, advantage)
