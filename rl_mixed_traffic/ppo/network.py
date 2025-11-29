import torch
from torch import nn
import torch.nn.functional as F
from typing import Tuple
import numpy as np


class ActorCriticNetwork(nn.Module):
    """Actor-Critic network for PPO with shared feature extraction.

    Supports both continuous (Gaussian policy) and discrete (Categorical policy) action spaces.

    Architecture:
    - Shared layers: state -> hidden1 -> hidden2
    - Actor head (continuous): hidden2 -> action_mean, log_std (for Gaussian policy)
    - Actor head (discrete): hidden2 -> action_logits (for Categorical policy)
    - Critic head: hidden2 -> state_value (V(s))

    Args:
        state_dim: Dimension of observation space
        action_dim: Dimension of action space (continuous) or number of actions (discrete)
        hidden_dim: Number of hidden units in each layer (default: 256)
        continuous: If True, use Gaussian policy for continuous actions.
                   If False, use Categorical policy for discrete actions (default: True)

    Example (continuous):
        >>> net = ActorCriticNetwork(state_dim=8, action_dim=1, continuous=True)
        >>> state = torch.randn(32, 8)  # batch of 32 states
        >>> action_mean, values = net(state)

    Example (discrete):
        >>> net = ActorCriticNetwork(state_dim=8, action_dim=21, continuous=False)
        >>> state = torch.randn(32, 8)
        >>> action_logits, values = net(state)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        continuous: bool = True,
    ):
        super(ActorCriticNetwork, self).__init__()

        self.continuous = continuous
        self.action_dim = action_dim

        # Shared feature extraction layers
        self.shared1 = nn.Linear(state_dim, hidden_dim)
        self.shared2 = nn.Linear(hidden_dim, hidden_dim)

        if continuous:
            # Continuous action space: Gaussian policy
            # Output mean and learn a separate log_std parameter
            self.actor_mean = nn.Linear(hidden_dim, action_dim)
            # Log standard deviation as learnable parameter (state-independent)
            self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))
        else:
            # Discrete action space: Categorical policy
            self.actor_head = nn.Linear(hidden_dim, action_dim)

        # Critic head (value function): outputs state value V(s)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.

        Args:
            state: State tensor of shape [batch_size, state_dim]

        Returns:
            Tuple of (actor_output, state_values):
            - If continuous: actor_output is action_mean [batch_size, action_dim]
            - If discrete: actor_output is action_logits [batch_size, action_dim]
            - state_values: [batch_size, 1] estimated state values V(s)
        """
        # Shared feature extraction
        x = F.relu(self.shared1(state))
        x = F.relu(self.shared2(x))

        if self.continuous:
            # Continuous: output action mean
            action_mean = self.actor_mean(x)
            actor_output = action_mean
        else:
            # Discrete: output action logits
            action_logits = self.actor_head(x)
            actor_output = action_logits

        # Critic: output state value
        state_value = self.critic_head(x)

        return actor_output, state_value

    def get_action_and_value(
        self, state: torch.Tensor, action: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, entropy, and value for given state.

        This is a convenience method for PPO training that combines forward pass
        with action sampling and log probability computation.

        Args:
            state: State tensor [batch_size, state_dim]
            action: Optional action tensor.
                   - For continuous: [batch_size, action_dim]
                   - For discrete: [batch_size]
                   If None, samples new actions.

        Returns:
            Tuple of (action, log_prob, entropy, value):
            - action: sampled or provided actions
            - log_prob: [batch_size] log probability of taken actions
            - entropy: [batch_size] policy entropy for each state
            - value: [batch_size, 1] state values V(s)
        """
        actor_output, value = self.forward(state)

        if self.continuous:
            # Continuous action space: Gaussian distribution
            action_mean = actor_output
            action_std = torch.exp(self.actor_log_std.expand_as(action_mean))
            probs = torch.distributions.Normal(action_mean, action_std)

            if action is None:
                action = probs.sample()

            # Sum log probs across action dimensions for multi-dimensional actions
            log_prob = probs.log_prob(action).sum(dim=-1)
            entropy = probs.entropy().sum(dim=-1)
        else:
            # Discrete action space: Categorical distribution
            action_logits = actor_output
            probs = torch.distributions.Categorical(logits=action_logits)

            if action is None:
                action = probs.sample()

            log_prob = probs.log_prob(action)
            entropy = probs.entropy()

        return action, log_prob, entropy, value