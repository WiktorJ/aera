import collections
import jax
import jax.numpy as jnp
from typing import Sequence

Batch = collections.namedtuple("Batch", ["observations", "actions", "rewards", "masks"])


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


def _apply_dropout(
    x: jnp.ndarray,
    dropout_rate: float,
    key: jnp.ndarray,
) -> jnp.ndarray:
    if dropout_rate <= 0.0:
        return x
    keep_prob = 1.0 - dropout_rate
    mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
    return jnp.where(mask, x / keep_prob, 0.0)
