import random
import numpy as np
import torch
import torch.nn as nn
from rl_mixed_traffic.configs.dqn_config import DQNConfig
from rl_mixed_traffic.dqn.reply_mem import ReplayMemory
from rl_mixed_traffic.dqn.network import DQNNetwork


class DQNAgent:
    def __init__(self, obs_dim: int, n_actions: int, config: DQNConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.q = DQNNetwork(obs_dim, n_actions).to(self.device)
        self.q_target = DQNNetwork(obs_dim, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=config.lr)
        self.buffer = ReplayMemory(config.buffer_size)
        self.steps = 0
        self.n_actions = n_actions
        self.loss_fn = nn.SmoothL1Loss()

        # ε-greedy schedule
        self.eps_start = config.epsilon_start
        self.eps_end = config.epsilon_end
        self.eps_decay_steps = config.epsilon_decay_steps

    def epsilon(self) -> float:
        """Compute the current epsilon for ε-greedy action selection."""
        fraction = min(float(self.steps) / self.eps_decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    @torch.no_grad()
    def act(self, state: np.ndarray, eval_mode: bool=False) -> int:
        """Select an action using ε-greedy policy.

        Args:
            state (np.ndarray): The current state.
            explore (bool): Whether to use exploration (ε-greedy) or not.

        Returns:
            int: The selected action.
        """
        self.steps += 1
        if (not eval_mode) and (random.random() < self.epsilon()):
            return random.randrange(self.n_actions)
        x = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.q(x)  # [1, n_actions]
        return int(q.argmax(dim=1).item())

    def remember(self, state, action, reward, next_state, done) -> None:
        self.buffer.append((state, action, reward, next_state, done))
    
    def learn(self):
        # warm up and train frequency
        if len(self.buffer) < self.config.start_learning_after:
            return None
        if self.steps % self.config.train_freq != 0:
            return None
        
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
                # Soft update
                with torch.no_grad():
                    for p_t, p in zip(self.q_target.parameters(), self.q.parameters()):
                        p_t.data.mul_(1.0 - self.config.tau).add_(self.config.tau * p.data)

        return loss.item()
    

    def save(self, path: str):
        """Saves the model checkpoints.

        Args:
            path (str): The path to save the model checkpoints.
        """
        torch.save({"q": self.q.state_dict(),
                    "q_target": self.q_target.state_dict(),
                    "steps": self.steps}, path)

    def load(self, path: str, map_location=None):
        """Loads the model checkpoints.

        Args:
            path (str): The path to load the model checkpoints from.
            map_location: The device to map the loaded tensors to.
        """
        ckpt = torch.load(path, map_location=map_location or self.device)
        self.q.load_state_dict(ckpt["q"])
        self.q_target.load_state_dict(ckpt["q_target"])
        self.steps = ckpt.get("steps", 0)
