from typing import Callable, Sequence

import jax
import jax.numpy as jnp


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
    value = jnp.tanh(jnp.clip(value, -1.0 + 1e-6, 1.0 - 1e-6))
    jacobian = _tanh_jacobian_diag(value)
    return value, log_prob_base - jnp.sum(jnp.log(jacobian))


class ReinforcePolicy:
    def __init__(
        self,
        hidden_dims: list[int],
        action_dim: int,
        obs_dim: int,
        key: jnp.ndarray,
        obs_dependent_std: bool = False,
        tanh_squash_dist: bool = False,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        dropout_rate: float = 0.0,
        temperature: float = 1.0,
        activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    ):
        self.hidden_dims = hidden_dims
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.obs_dependent_std = obs_dependent_std
        self.tanh_squash_dist = tanh_squash_dist
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.dropout_rate = dropout_rate
        self.activation_fn = activation_fn
        self.temperature = temperature

        self.key, trunk_key, mean_key, log_std_key = jax.random.split(key, 4)
        self.trunk_weights = _init_network_params(trunk_key, [obs_dim] + hidden_dims)
        self.mean_weights, self.mean_bias = _init_network_params(
            mean_key, [hidden_dims[-1], action_dim]
        )[0]
        self.log_std_weights, self.log_std_bias = _init_network_params(
            log_std_key, [hidden_dims[-1], action_dim]
        )[0]

    def __call__(self, x):
        for w, b in self.trunk_weights:
            x = self.activation_fn(jnp.dot(x, w) + b)
        x = jnp.tanh(x)
        mean = jnp.dot(x, self.mean_weights) + self.mean_bias
        if self.obs_dependent_std:
            log_std = jnp.dot(x, self.log_std_weights) + self.log_std_bias
        else:
            log_std = jnp.zeros_like(mean)
        log_std = jnp.clip(log_std, self.log_std_min, self.log_std_max)
        if not self.tanh_squash_dist:
            mean = jnp.tanh(jnp.clip(mean, -1.0 + 1e-6, 1.0 - 1e-6))
        self.key, key_gaussian = jax.random.split(self.key)
        action = _sample_gaussian(key_gaussian, mean, log_std * self.temperature)
        log_prob = _gaussian_log_prob(action, mean, log_std)
        if not self.tanh_squash_dist:
            return action, log_prob
        return _tanh_squash(action, log_prob)
