import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
from pathlib import Path

from rl_mixed_traffic.agents.base_agent import BaseAgent
from rl_mixed_traffic.configs.ppo_config import PPOConfig
from rl_mixed_traffic.ppo.network import ActorCriticNetwork
from rl_mixed_traffic.ppo.rollout_buffer import RolloutBuffer


class PPOAgent(BaseAgent):
    """Proximal Policy Optimization (PPO) agent with clipped surrogate objective.

    Implements PPO with:
    - Actor-Critic architecture with shared feature extraction
    - Clipped surrogate objective for stable policy updates
    - Generalized Advantage Estimation (GAE) for advantage computation
    - Multiple epochs of minibatch updates per rollout
    - Entropy regularization for exploration
    - Value function loss with optional clipping

    Args:
        obs_dim: Dimension of observation space
        n_actions: Number of discrete actions
        config: PPOConfig object containing all hyperparameters

    Example:
        >>> config = PPOConfig(lr=3e-4, gamma=0.99, clip_epsilon=0.2)
        >>> agent = PPOAgent(obs_dim=8, n_actions=21, config=config)
        >>> # Collect rollout
        >>> for step in range(2048):
        >>>     action = agent.act(state)
        >>>     next_state, reward, done, _ = env.step(action)
        >>>     agent.store_transition(state, action, reward, value, log_prob, done)
        >>> # Update policy
        >>> losses = agent.learn(last_value=0.0)
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        config: PPOConfig,
        continuous: bool = True,
    ):
        self.config = config
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.continuous = continuous
        self.device = torch.device(config.device)

        # Actor-Critic network
        self.network = ActorCriticNetwork(
            obs_dim, action_dim, continuous=continuous
        ).to(self.device)

        # Optimizer
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=config.lr)

        # Rollout buffer
        self.buffer = RolloutBuffer(
            buffer_size=10000, obs_dim=obs_dim, device=self.device
        )

        # Training state
        self.total_steps = 0
        self.update_count = 0

    @torch.no_grad()
    def act(self, state: np.ndarray, eval_mode: bool = False):
        """Select an action using the current policy.

        Args:
            state: Current observation (numpy array)
            eval_mode: If True, select deterministically (mean). If False, sample.

        Returns:
            - For continuous: action as numpy array [action_dim]
            - For discrete: action index as int

        Note:
            During training (eval_mode=False), actions are sampled from the policy.
            During evaluation (eval_mode=True):
            - Continuous: returns the mean action
            - Discrete: returns the argmax action
        """
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        if self.continuous:
            action_mean, _ = self.network(state_tensor)

            if eval_mode:
                # Deterministic: use mean action
                action = action_mean.squeeze(0).cpu().numpy()
            else:
                # Stochastic: sample from Gaussian policy
                action_std = torch.exp(
                    self.network.actor_log_std.expand_as(action_mean)
                )
                probs = torch.distributions.Normal(action_mean, action_std)
                action = probs.sample().squeeze(0).cpu().numpy()

            return action
        else:
            logits, _ = self.network(state_tensor)

            if eval_mode:
                # Deterministic: select action with highest probability
                action = logits.argmax(dim=-1).item()
            else:
                # Stochastic: sample from policy distribution
                probs = torch.distributions.Categorical(logits=logits)
                action = probs.sample().item()

            return int(action)

    @torch.no_grad()
    def get_action_and_value(self, state: np.ndarray):
        """Get action, value, and log probability for a given state.

        This method is used during rollout collection to get all necessary
        information for PPO training in a single forward pass.

        Args:
            state: Current observation (numpy array)

        Returns:
            Tuple of (action, value, log_prob):
            - action: For continuous: numpy array [action_dim], For discrete: int
            - value: Value estimate V(s) as float
            - log_prob: Log probability of sampled action as float
        """
        state_tensor = torch.as_tensor(
            state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        action, log_prob, _, value = self.network.get_action_and_value(state_tensor)

        if self.continuous:
            return (
                action.squeeze(0).cpu().numpy(),
                float(value.item()),
                float(log_prob.item()),
            )
        else:
            return (
                int(action.item()),
                float(value.item()),
                float(log_prob.item()),
            )

    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Store a transition in the rollout buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            value: Value estimate V(s) from critic
            log_prob: Log probability of action under current policy
            done: Whether episode terminated
        """
        self.buffer.add(state, action, reward, value, log_prob, done)
        self.total_steps += 1

    def learn(self, last_value: float = 0.0) -> Dict[str, float]:
        """Update policy using PPO algorithm on collected rollout data.

        Performs multiple epochs of minibatch gradient descent on the rollout
        buffer, optimizing the clipped PPO objective.

        Args:
            last_value: Value estimate for the state after the last transition.
                       Use V(s_final) if episode didn't terminate, 0.0 otherwise.

        Returns:
            Dictionary containing training metrics:
            - policy_loss: Mean policy (actor) loss
            - value_loss: Mean value (critic) loss
            - entropy: Mean policy entropy
            - clipfrac: Fraction of ratios that were clipped

        Note:
            This method clears the rollout buffer after training.
        """
        if len(self.buffer) == 0:
            return {}

        # Compute advantages and returns using GAE
        advantages, returns = self.buffer.compute_gae(
            last_value=last_value,
            gamma=self.config.gamma,
            lam=self.config.lam,
        )

        # Get all buffer data
        states, actions, old_log_probs = self.buffer.get()

        # Convert advantages and returns to tensors
        advantages = torch.as_tensor(advantages, dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(returns, dtype=torch.float32, device=self.device)

        # Normalize advantages (improves training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Training metrics
        policy_losses = []
        value_losses = []
        entropies = []
        clipfracs = []

        # Multiple epochs of updates (rollout data reused)
        for epoch in range(self.config.k_epochs):
            # Generate random minibatch indices
            indices = torch.randperm(len(states), device=self.device)

            # Process minibatches
            for start in range(0, len(states), self.config.batch_size):
                end = start + self.config.batch_size
                mb_indices = indices[start:end]

                # Get minibatch data
                mb_states = states[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]

                # Forward pass through network
                _, new_log_probs, entropy, values = self.network.get_action_and_value(
                    mb_states, mb_actions
                )

                # Compute ratio: π_new(a|s) / π_old(a|s)
                log_ratio = new_log_probs - mb_old_log_probs
                ratio = torch.exp(log_ratio)

                # PPO clipped surrogate objective
                # L^CLIP = E[min(r_t * A_t, clip(r_t, 1-ε, 1+ε) * A_t)]
                policy_loss_1 = ratio * mb_advantages
                policy_loss_2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_epsilon,
                        1.0 + self.config.clip_epsilon,
                    )
                    * mb_advantages
                )
                policy_loss = -torch.min(policy_loss_1, policy_loss_2).mean()

                # Value loss (MSE between predicted and actual returns)
                values = values.squeeze(-1)
                value_loss = nn.functional.mse_loss(values, mb_returns)

                # Entropy bonus (encourages exploration)
                entropy_loss = -entropy.mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * value_loss
                    + self.config.entropy_coef * entropy_loss
                )

                # Optimize
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
                self.optimizer.step()

                # Track metrics
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropies.append(entropy.mean().item())

                # Track clip fraction (diagnostic)
                with torch.no_grad():
                    clipfrac = (
                        (torch.abs(ratio - 1.0) > self.config.clip_epsilon)
                        .float()
                        .mean()
                        .item()
                    )
                    clipfracs.append(clipfrac)

        # Clear buffer after training
        self.buffer.clear()
        self.update_count += 1

        # Return training metrics
        return {
            "policy_loss": np.mean(policy_losses),
            "value_loss": np.mean(value_losses),
            "entropy": np.mean(entropies),
            "clipfrac": np.mean(clipfracs),
        }

    def update(self, *args: Any, **kwargs: Any) -> Optional[Dict[str, float]]:
        """Update method for BaseAgent interface compatibility.

        For PPO, prefer using store_transition() during rollout and learn()
        after collecting a full batch of experiences.

        This method is here for interface compatibility but delegates to learn().
        """
        # PPO doesn't update after every step like DQN
        # This is mainly for interface compatibility
        if "last_value" in kwargs:
            return self.learn(last_value=kwargs["last_value"])
        return None

    def save(self, path: str) -> None:
        """Save the agent's network, optimizer, and training state.

        Args:
            path: Path to save the checkpoint

        Note:
            Saves: actor-critic network, optimizer state, step count, and config
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "config": self.get_config(),
        }
        torch.save(checkpoint, path)

    def load(self, path: str, map_location: Optional[str] = None) -> None:
        """Load the agent's network, optimizer, and training state.

        Args:
            path: Path to the checkpoint file
            map_location: Device to map tensors to (e.g., 'cpu', 'cuda:0')

        Note:
            If map_location is not specified, uses the agent's current device.
        """
        device = map_location or self.device
        ckpt = torch.load(path, map_location=device, weights_only=False)

        # Load network
        self.network.load_state_dict(ckpt["network"])

        # Load optimizer if available
        if "optimizer" in ckpt:
            self.optimizer.load_state_dict(ckpt["optimizer"])

        # Load training state
        self.total_steps = ckpt.get("total_steps", 0)
        self.update_count = ckpt.get("update_count", 0)

    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration as a dictionary.

        Returns:
            Dictionary containing all agent hyperparameters and state
        """
        return {
            "agent_type": "PPOAgent",
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "continuous": self.continuous,
            "lr": self.config.lr,
            "gamma": self.config.gamma,
            "lam": self.config.lam,
            "clip_epsilon": self.config.clip_epsilon,
            "k_epochs": self.config.k_epochs,
            "batch_size": self.config.batch_size,
            "entropy_coef": self.config.entropy_coef,
            "value_coef": self.config.value_coef,
            "device": str(self.device),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
        }