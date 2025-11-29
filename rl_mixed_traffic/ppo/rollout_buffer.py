import numpy as np
import torch
from typing import Tuple, List


class RolloutBuffer:
    """Buffer for storing trajectories during PPO rollout collection.

    Stores states, actions, rewards, values, and log probabilities collected
    during environment interaction. Supports GAE (Generalized Advantage Estimation)
    computation for advantage calculation.

    Args:
        buffer_size: Maximum number of transitions to store
        obs_dim: Dimension of observation space
        device: Device to store tensors on ('cpu' or 'cuda')

    Example:
        >>> buffer = RolloutBuffer(buffer_size=2048, obs_dim=8, device='cpu')
        >>> # During rollout
        >>> buffer.add(state, action, reward, value, log_prob)
        >>> # After rollout
        >>> batch = buffer.get()
        >>> buffer.clear()
    """

    def __init__(self, buffer_size: int, obs_dim: int, device: str = "cpu"):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.device = device
        self.clear()

    def clear(self) -> None:
        """Reset the buffer to empty state."""
        self.states: List[np.ndarray] = []
        self.actions: List[int] = []
        self.rewards: List[float] = []
        self.values: List[float] = []
        self.log_probs: List[float] = []
        self.dones: List[bool] = []
        self.ptr = 0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add a single transition to the buffer.

        Args:
            state: Observation from environment
            action: Action taken
            reward: Reward received
            value: Value estimate V(s) from critic
            log_prob: Log probability of action under current policy
            done: Whether episode terminated
        """
        if self.ptr >= self.buffer_size:
            raise ValueError(
                f"Buffer overflow: trying to add to full buffer (size={self.buffer_size})"
            )

        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.log_probs.append(log_prob)
        self.dones.append(done)
        self.ptr += 1

    def compute_gae(
        self, last_value: float, gamma: float = 0.99, lam: float = 0.95
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation (GAE) and returns.

        GAE formula:
            δ_t = r_t + γV(s_{t+1}) - V(s_t)
            A_t = δ_t + (γλ)δ_{t+1} + (γλ)²δ_{t+2} + ...

        Returns are computed as: R_t = A_t + V(s_t)

        Args:
            last_value: Value estimate for the state after the last transition
                       (V(s_T) for final state, or 0 if episode terminated)
            gamma: Discount factor (default: 0.99)
            lam: GAE lambda parameter for bias-variance tradeoff (default: 0.95)

        Returns:
            Tuple of (advantages, returns) as numpy arrays
        """
        n = len(self.rewards)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        # Convert lists to arrays for easier manipulation
        rewards = np.array(self.rewards, dtype=np.float32)
        values = np.array(self.values, dtype=np.float32)
        dones = np.array(self.dones, dtype=np.float32)

        # Compute advantages using GAE
        gae = 0.0
        for t in reversed(range(n)):
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # TD error: δ_t = r_t + γV(s_{t+1})(1-done) - V(s_t)
            delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]

            # GAE: A_t = δ_t + (γλ)δ_{t+1} + ...
            gae = delta + gamma * lam * (1 - dones[t]) * gae
            advantages[t] = gae

        # Returns: R_t = A_t + V(s_t)
        returns = advantages + values

        return advantages, returns

    def get(self) -> Tuple[torch.Tensor, ...]:
        """Get all stored transitions as tensors.

        Returns:
            Tuple of (states, actions, log_probs, advantages, returns) as tensors
            ready for PPO training.

        Note:
            This method does NOT clear the buffer. Call clear() explicitly after use.
        """
        states = torch.as_tensor(
            np.array(self.states), dtype=torch.float32, device=self.device
        )
        actions = torch.as_tensor(
            np.array(self.actions), dtype=torch.int64, device=self.device
        )
        log_probs = torch.as_tensor(
            np.array(self.log_probs), dtype=torch.float32, device=self.device
        )

        # Note: advantages and returns are computed via compute_gae()
        # and should be added to the buffer before calling get()
        return states, actions, log_probs

    def __len__(self) -> int:
        """Return the current number of transitions in buffer."""
        return self.ptr

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return self.ptr >= self.buffer_size
