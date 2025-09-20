from dataclasses import dataclass


@dataclass
class Config:
    seed: int = 42
    env_name: str = ""
    max_episodes: int = 1000
    env_render_width: int = 1920
    env_render_height: int = 1080
    env_render_mode: str = "rgb_array"
    env_max_steps: int = 1000

    policy_batch_size: int = 1024
    policy_num_epochs: int = 10
    policy_update_freq: int = 1

    policy_hidden_dims: list[int] = [128, 128]
    policy_activation_fn: str = "relu"
    policy_lr: float = 3e-4
    policy_obs_dependent_std: bool = True
    policy_tanh_squash_dist: bool = False
    policy_log_std_min: float = -20.0
    policy_log_std_max: float = 2.0
    policy_dropout_rate: float = 0.0
    policy_temperature: float = 1.0
