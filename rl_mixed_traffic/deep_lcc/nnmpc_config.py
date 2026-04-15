from dataclasses import dataclass

import torch


@dataclass
class NNMPCConfig:
    """Configuration for NNMPC training and inference."""

    # Network architecture
    hidden_dims: tuple[int, ...] = (256, 128)

    # Training
    lr: float = 5e-4
    batch_size: int = 256
    num_epochs: int = 200
    val_split: float = 0.1
    weight_decay: float = 1e-5
    max_grad_norm: float = 10.0
    patience: int = 20  # early stopping patience (epochs)

    # Output bounds (acceleration limits)
    accel_min: float = -5.0
    accel_max: float = 3.0

    # Paths
    dataset_path: str = "deep_lcc_dataset/dataset.npz"
    model_path: str = "deep_lcc_results/nnmpc.pth"

    # Device
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
