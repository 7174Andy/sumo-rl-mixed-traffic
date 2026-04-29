"""NNMPC-shaped actor-critic network for RLMPC PPO training.

Actor architecture matches NNMPCNetwork (260 → 256 → 128 → 2) so warm-starting
from nnmpc.pth is a direct state_dict copy. The actor produces the *pre-tanh*
mean of a diagonal Gaussian; the post-tanh squashed action is in [-1, 1] and
the env scales it to the per-mode action box.

Implements the same get_action_and_value signature as
rl_mixed_traffic/ppo/network.py:ActorCriticNetwork so PPOAgent works unchanged.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch
from torch import nn


def _ortho_init(layer: nn.Linear, gain: float = math.sqrt(2.0)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class NNMPCActorCritic(nn.Module):
    """Actor-Critic with NNMPC-shaped actor body and an independent critic head."""

    def __init__(
        self,
        obs_dim: int = 260,
        action_dim: int = 2,
        hidden_dims: tuple[int, ...] = (256, 128),
        log_std_init: float = math.log(0.5),
        final_layer_gain: float = 1.0,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dims = tuple(hidden_dims)

        # --- Actor body: same shape as NNMPCNetwork's nn.Sequential pre-tanh ---
        actor_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            actor_layers.append(_ortho_init(nn.Linear(prev, h)))
            actor_layers.append(nn.ReLU())
            prev = h
        actor_final = _ortho_init(nn.Linear(prev, action_dim), gain=final_layer_gain)
        actor_layers.append(actor_final)
        self.actor_body = nn.Sequential(*actor_layers)

        # --- Critic head: independent MLP, same hidden dims ---
        critic_layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden_dims:
            critic_layers.append(_ortho_init(nn.Linear(prev, h)))
            critic_layers.append(nn.ReLU())
            prev = h
        critic_layers.append(_ortho_init(nn.Linear(prev, 1), gain=1.0))
        self.critic = nn.Sequential(*critic_layers)

        # State-independent learnable log-std
        self.actor_log_std = nn.Parameter(
            torch.full((1, action_dim), float(log_std_init))
        )

    # ----- Public API matching ActorCriticNetwork -----

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        actor_pre_tanh = self.actor_body(state)
        value = self.critic(state)
        return actor_pre_tanh, value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        actor_pre_tanh, value = self.forward(state)
        log_std = torch.clamp(self.actor_log_std, min=-2.0, max=0.5)
        std = torch.exp(log_std.expand_as(actor_pre_tanh))
        normal = torch.distributions.Normal(actor_pre_tanh, std)

        if action is None:
            raw = normal.rsample()
            action_squashed = torch.tanh(raw)
        else:
            action_squashed = action
            raw = torch.atanh(torch.clamp(action_squashed, -0.999, 0.999))

        log_prob = normal.log_prob(raw).sum(dim=-1)
        # tanh-squash log-prob correction
        log_prob = log_prob - torch.log(
            1.0 - action_squashed.pow(2) + 1e-6
        ).sum(dim=-1)
        entropy = normal.entropy().sum(dim=-1)

        return action_squashed, log_prob, entropy, value

    # ----- Warm-start from a trained NNMPCNetwork checkpoint -----

    def warm_start_from_nnmpc(self, ckpt_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Copy NNMPCNetwork weights into self.actor_body.

        NNMPCNetwork.net is nn.Sequential of:
            [Linear, ReLU, Linear, ReLU, Linear, Tanh]
        We map the three Linear layers into self.actor_body's three Linear layers.

        Returns:
            (input_mean, input_std) from the checkpoint, for env-side normalization.
        """
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["model_state_dict"]

        # NNMPCNetwork.net keys look like 'net.0.weight', 'net.0.bias',
        # 'net.2.weight', 'net.2.bias', 'net.4.weight', 'net.4.bias'
        # Our actor_body (same shape) has keys '0.weight', '0.bias', etc.
        nnmpc_indices = [0, 2, 4]
        actor_indices = [0, 2, 4]
        for nnmpc_i, actor_i in zip(nnmpc_indices, actor_indices):
            w_key = f"net.{nnmpc_i}.weight"
            b_key = f"net.{nnmpc_i}.bias"
            if w_key not in sd or b_key not in sd:
                raise KeyError(
                    f"NNMPC checkpoint missing keys {w_key!r}/{b_key!r}"
                )
            target = self.actor_body[actor_i]
            assert isinstance(target, nn.Linear), (
                f"actor_body[{actor_i}] is {type(target).__name__}, expected nn.Linear"
            )
            if target.weight.shape != sd[w_key].shape:
                raise ValueError(
                    f"Shape mismatch at layer {actor_i}: "
                    f"actor {tuple(target.weight.shape)} vs nnmpc {tuple(sd[w_key].shape)}"
                )
            with torch.no_grad():
                target.weight.copy_(sd[w_key])
                target.bias.copy_(sd[b_key])

        return ckpt["input_mean"], ckpt["input_std"]
