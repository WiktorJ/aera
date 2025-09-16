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
