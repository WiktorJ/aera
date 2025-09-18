from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import optax
import dataclasses

from aera.autonomous.basic_rl.reinforce.common import Batch


def _random_layer_params(
    m: int, n: int, key: jnp.ndarray, scale: float = 1e-2
) -> tuple[jnp.ndarray, jnp.ndarray]:
    w_key, b_key = jax.random.split(key)
    return scale * jax.random.normal(w_key, (n, m)), scale * jax.random.normal(
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


def _tanh_squash(
    value: jnp.ndarray, log_prob_base: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    # The variable change formula states that
    # pY(y) = p_X(f^{-1}(y)) * det(d/dy f^{-1}(y))
    # in base variable x or p_Y(y) = p_X(x) * det(d/dx f(x)).
    # In this case f = tanh and f^{-1} = atanh.
    # Since we are interested in log_prob, the formula becomes
    # log(p_Y(y)) = log(p_X(x)) - log(det(d/dx f(x))) =
    # = log_probs_base - jacobian
    jacobian = _tanh_jacobian_diag(value)
    value = jnp.tanh(jnp.clip(value, -1.0 + 1e-6, 1.0 - 1e-6))
    return value, log_prob_base - jnp.sum(jnp.log(jacobian))


@dataclasses.dataclass(frozen=True)
class ReinforcePolicyState:
    hidden_dims: list[int]
    action_dim: int
    obs_dim: int
    key: jnp.ndarray
    oprimizer: optax.GradientTransformationExtraArgs
    trunk_weights: list[tuple[jnp.ndarray, jnp.ndarray]]
    mean_weights: jnp.ndarray
    mean_bias: jnp.ndarray
    log_std_weights: jnp.ndarray
    log_std_bias: jnp.ndarray
    opt_state: optax.OptState
    all_params: (
        list[tuple[jnp.ndarray, jnp.ndarray]]
        | list[tuple[tuple[jnp.ndarray, jnp.ndarray], tuple[jnp.ndarray, jnp.ndarray]]]
    )
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
        oprimizer: optax.GradientTransformationExtraArgs,
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

        mean_weights, mean_bias = _init_network_params(
            mean_key, [hidden_dims[-1], action_dim]
        )[0]

        log_std_weights, log_std_bias = _init_network_params(
            log_std_key, [hidden_dims[-1], action_dim]
        )[0]

        all_params = trunk_weights + [
            (mean_weights, mean_bias),
            (log_std_weights, log_std_bias),
        ]
        opt_state = oprimizer.init(all_params)

        return ReinforcePolicyState(
            hidden_dims=hidden_dims,
            action_dim=action_dim,
            obs_dim=obs_dim,
            key=key,
            oprimizer=oprimizer,
            trunk_weights=trunk_weights,
            mean_weights=mean_weights,
            mean_bias=mean_bias,
            log_std_weights=log_std_weights,
            log_std_bias=log_std_bias,
            opt_state=opt_state,
            all_params=all_params,
            obs_dependent_std=obs_dependent_std,
            tanh_squash_dist=tanh_squash_dist,
            log_std_min=log_std_min,
            log_std_max=log_std_max,
            dropout_rate=dropout_rate,
            temperature=temperature,
            activation_fn=activation_fn,
        )


def call_reinforce_policy(
    x: jnp.ndarray,
    state: ReinforcePolicyState,
) -> tuple[jnp.ndarray, jnp.ndarray, ReinforcePolicyState]:
    for w, b in state.trunk_weights:
        x = state.activation_fn(jnp.dot(x, w) + b)
    x = jnp.tanh(x)
    mean = jnp.dot(x, state.mean_weights) + state.mean_bias
    if state.obs_dependent_std:
        log_std = jnp.dot(x, state.log_std_weights) + state.log_std_bias
    else:
        log_std = jnp.zeros_like(mean)
    log_std = jnp.clip(log_std, state.log_std_min, state.log_std_max)
    if not state.tanh_squash_dist:
        mean = jnp.tanh(jnp.clip(mean, -1.0 + 1e-6, 1.0 - 1e-6))
    new_key, key_gaussian = jax.random.split(state.key)
    new_state = dataclasses.replace(state, key=new_key)  # type: ignore
    action = _sample_gaussian(key_gaussian, mean, log_std * state.temperature)
    log_prob = _gaussian_log_prob(action, mean, log_std)
    if not state.tanh_squash_dist:
        return action, log_prob, new_state
    return *_tanh_squash(action, log_prob), new_state


def update_reinforce_policy(
    state: ReinforcePolicyState,
    batch: Batch,
    baseline: jnp.ndarray,
) -> tuple[dict[str, jnp.ndarray], ReinforcePolicyState]:
    def loss_fn(log_prob: jnp.ndarray, baseline: jnp.ndarray):
        loss = -(log_prob * baseline).mean()
        return loss, {"policy_loss": loss, "log_prob": log_prob.mean()}

    _, log_prob, state = call_reinforce_policy(batch.observations, state)
    grad, info = jax.grad(loss_fn, has_aux=True)(log_prob, baseline)
    updates, opt_state = state.oprimizer.update(grad, state.opt_state)
    all_params = optax.apply_updates(state.all_params, updates)
    new_state = dataclasses.replace(state, opt_state=opt_state, all_params=all_params)  # type: ignore
    return info, new_state


ReinforcePolicyState.create(
    hidden_dims=[256, 256],
    action_dim=1,
    obs_dim=1,
    key=jax.random.PRNGKey(0),
    oprimizer=optax.adam(1e-3),
    obs_dependent_std=False,
    tanh_squash_dist=False,
    log_std_min=-20.0,
    log_std_max=2.0,
)
