from dataclasses import dataclass
import torch


@dataclass
class PPOConfig:
    lr: float = 1e-4
    gamma: float = 0.99
    lam: float = 0.95
    clip_epsilon: float = 0.2
    k_epochs: int = 10
    batch_size: int = 64
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
