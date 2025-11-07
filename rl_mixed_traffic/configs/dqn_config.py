from dataclasses import dataclass
import torch

@dataclass
class DQNConfig:
    gamma: float = 0.99
    lr: float =5e-4
    batch_size: int = 128
    buffer_size: int = 200_000
    start_learning_after: int = 10_000  
    train_freq: int = 1
    target_update_freq: int = 3_000   # steps
    tau: float = 0.01                  # hard update if 1.0; else soft update factor
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 500_000
    max_grad_norm: float = 10.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"