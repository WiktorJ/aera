from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    seed: int = 44
    env_name: str = "InvertedDoublePendulum-v5"
    # env_name: str = "Hopper-v5"
    max_episodes: int = 1000
    env_render_width: int = 1920
    env_render_height: int = 1088
    env_render_mode: str = "rgb_array"
    # env_render_mode: Optional[str] = None
    env_max_steps: int = 1000
    ep_len: int = 512
    max_steps: int = 1001
    batch_size: int = 16384
    num_envs: int = 32
    eval_step_interval: int = 50
    eval_num_episodes: int = 1
    eval_render: bool = True

    gamma = 0.99
    policy_hidden_dims: tuple[int, ...] = (256, 256)
    policy_activation_fn: str = "relu"
    policy_lr: float = 3e-4
    policy_obs_dependent_std: bool = False
    policy_tanh_squash_dist: bool = True
    policy_log_std_min: float = -5.0
    policy_log_std_max: float = 2.0
    policy_dropout_rate: float = 0.0
    policy_temperature: float = 1.0
    profile: bool = False

    value_hidden_dims: tuple[int, ...] = (256, 256)
    value_activation_fn: str = "relu"
    value_lr: float = 3e-4
    value_dropout_rate: float = 0.0

    use_gae = True
    use_bootstrap_targets = True
    gae_lambda = 0.95
