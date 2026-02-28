from dataclasses import dataclass
import torch


@dataclass
class PPOConfig:
    lr: float = 3e-4  # Increased from 1e-4 for better learning
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    k_epochs: int = 10
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Lagrangian PPO fields
    enable_lagrangian: bool = False
    lambda_init: float = 0.0
    lambda_lr: float = 0.01
    lambda_max: float = 10.0
