import torch
from torch import nn
import torch.nn.functional as F


class NNMPCNetwork(nn.Module):
    """MLP that approximates the DeeP-LCC QP solver.

    Maps (uini, yini, eini) → u_opt with Tanh output scaling
    to guarantee acceleration bounds.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: tuple[int, ...] = (256, 128),
        accel_min: float = -5.0,
        accel_max: float = 2.0,
    ):
        super().__init__()

        self.accel_min = accel_min
        self.accel_max = accel_max

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            prev_dim = h
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Tanh output in [-1, 1] → scale to [accel_min, accel_max]
        t = self.net(x)
        return (t + 1.0) / 2.0 * (self.accel_max - self.accel_min) + self.accel_min
