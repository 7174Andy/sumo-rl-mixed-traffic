import random
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path

from rl_mixed_traffic.agents.base_agent import BaseAgent
from rl_mixed_traffic.configs.dqn_config import DQNConfig
from rl_mixed_traffic.dqn.replay_mem import ReplayMemory
from rl_mixed_traffic.dqn.network import DQNNetwork


class DQNAgent(BaseAgent):
    """Deep Q-Network (DQN) agent with experience replay and target network.

    Implements Double DQN with:
    - Experience replay buffer for sample efficiency
    - Separate target network for training stability
    - Epsilon-greedy exploration with linear decay
    - Gradient clipping for training stability
    - Optional soft/hard target network updates

    Args:
        obs_dim: Dimension of observation space
        n_actions: Number of discrete actions
        config: DQNConfig object containing all hyperparameters

    Example:
        >>> config = DQNConfig(lr=1e-3, gamma=0.99)
        >>> agent = DQNAgent(obs_dim=8, n_actions=21, config=config)
        >>> state = np.random.rand(8)
        >>> action = agent.act(state)
        >>> # Training loop
        >>> loss = agent.update(state, action, reward=1.0, next_state, done=False)
    """

    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig):
        self.config = config
        self.obs_dim = obs_dim
        self.n_actions = n_actions
        self.device = torch.device(config.device)

        # Q-networks
        self.q = DQNNetwork(obs_dim, n_actions).to(self.device)
        self.q_target = DQNNetwork(obs_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())

        # Optimizer and loss
        self.opt = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.loss_fn = nn.SmoothL1Loss()

        # Replay buffer
        self.buffer = ReplayMemory(config.buffer_size)

        # Training state
        self.steps = 0

        # ε-greedy schedule
        self.eps_start = config.epsilon_start
        self.eps_end = config.epsilon_end
        self.eps_decay_steps = config.epsilon_decay_steps

    def epsilon(self) -> float:
        """Compute the current epsilon for ε-greedy action selection.

        Returns:
            Current epsilon value linearly interpolated between eps_start and eps_end
        """
        fraction = min(float(self.steps) / self.eps_decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, state: np.ndarray, eval_mode: bool = False) -> int:
        """Select an action using ε-greedy policy.

        Args:
            state: Current observation (numpy array)
            eval_mode: If True, always exploit (no exploration). Default: False

        Returns:
            Selected action index

        Note:
            This method increments self.steps for epsilon decay scheduling.
        """
        self.steps += 1

        # Exploration vs exploitation
        if (not eval_mode) and (random.random() < self.epsilon()):
            return random.randrange(self.n_actions)

        # Forward pass through Q-network
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)  # [1, n_actions]
        return int(q.argmax(dim=1).item())

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> Optional[float]:
        """Store transition and perform learning update (if conditions are met).

        This is a unified interface that:
        1. Stores the transition in replay buffer
        2. Performs a learning step if conditions are met

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated

        Returns:
            Loss value if learning occurred, None otherwise

        Note:
            Learning only occurs when:
            - Buffer has at least start_learning_after samples
            - Current step is a multiple of train_freq
        """
        # Store transition
        self.store_transition(state, action, reward, next_state, done)

        # Perform learning
        return self.learn()

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store a transition in the replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.buffer.append((state, action, reward, next_state, done))

    def learn(self) -> Optional[float]:
        """Perform one step of learning from replay buffer.

        Samples a batch from replay buffer, computes Double DQN loss,
        and updates the Q-network. Also updates target network periodically.

        Returns:
            Loss value if learning occurred, None if conditions not met

        Note:
            Uses Double DQN: action selection from online network,
            value estimation from target network.
        """
        # Check if we should learn
        if len(self.buffer) < self.config.start_learning_after:
            return None
        if self.steps % self.config.train_freq != 0:
            return None

        # Sample batch
        batch = self.buffer.sample(self.config.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert to tensors
        states = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Compute current Q values
        q_values = self.q(states).gather(1, actions)  # [batch_size, 1]

        # Double DQN target: a* = argmax_a Q_online(ns,a); y = r + γ(1-d) Q_target(ns,a*)
        with torch.no_grad():
            a_star = self.q(next_states).argmax(dim=1, keepdim=True)  # [batch_size, 1]
            q_next = self.q_target(next_states).gather(1, a_star)  # [batch_size, 1]
            targets = rewards + self.config.gamma * (1 - dones) * q_next  # [batch_size, 1]

        # Compute loss
        loss = self.loss_fn(q_values, targets)

        # Optimize the model
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.config.max_grad_norm)
        self.opt.step()

        # Update target network
        if self.steps % self.config.target_update_freq == 0:
            if self.config.tau == 1.0:
                # Hard update
                self.q_target.load_state_dict(self.q.state_dict())
            else:
                # Soft update: θ_target = (1-τ)θ_target + τθ_online
                with torch.no_grad():
                    for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                        p_t.data.mul_(1.0 - self.config.tau).add_(self.config.tau * p.data)

        return loss.item()

    def save(self, path: str) -> None:
        """Save the agent's networks, optimizer, and training state.

        Args:
            path: Path to save the checkpoint

        Note:
            Saves: Q-network, target network, optimizer state, training steps, and config
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "q": self.q.state_dict(),
            "q_target": self.q_target.state_dict(),
            "optimizer": self.opt.state_dict(),
            "steps": self.steps,
            "config": self.get_config(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """Load the agent's networks, optimizer, and training state.

        Args:
            path: Path to the checkpoint file
            map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')

        Note:
            If map_location is not specified, uses the agent's current device.
        """
        device = map_location or self.device
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Load networks
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])

        # Load optimizer if available
        if "optimizer" in ckpt:
            self.opt.load_state_dict(ckpt["optimizer"])

        # Load training state
        self.steps = ckpt.get("steps", 0)

    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration as a dictionary.

        Returns:
            Dictionary containing all agent hyperparameters and state
        """
        return {
            "agent_type": "DQNAgent",
            "obs_dim": self.obs_dim,
            "n_actions": self.n_actions,
            "lr": self.config.lr,
            "gamma": self.config.gamma,
            "batch_size": self.config.batch_size,
            "buffer_size": self.config.buffer_size,
            "start_learning_after": self.config.start_learning_after,
            "train_freq": self.config.train_freq,
            "target_update_freq": self.config.target_update_freq,
            "tau": self.config.tau,
            "epsilon_start": self.eps_start,
            "epsilon_end": self.eps_end,
            "epsilon_decay_steps": self.eps_decay_steps,
            "max_grad_norm": self.config.max_grad_norm,
            "device": str(self.device),
            "steps": self.steps,
            "buffer_size_current": len(self.buffer),
        }
