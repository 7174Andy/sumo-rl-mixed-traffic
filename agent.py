from collections import defaultdict
import numpy as np
from typing import Tuple

class QLearningAgent:
    def __init__(self, action_space: int, alpha: float=0.2, gamma: float = 0.98, eps_start: float = 1.0, eps_end: float = 0.1, eps_decay_steps: int = 15_000):
        self.action_space = action_space
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.eps_start = eps_start  # Exploration rate
        self.eps_end = eps_end  # Minimum exploration rate
        self.step_count = 0  # Step counter for epsilon decay
        self.eps_decay_steps = eps_decay_steps  # Steps to decay exploration rate
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))
        
    def epsilon(self) -> float:
        """Calculate the current epsilon value based on decay schedule."""
        frac = min(1.0, self.step_count / max(1, self.eps_decay_steps))
        return float(self.eps_start + frac * (self.eps_end - self.eps_start))

    def act(self, state: Tuple[int, int, int]) -> int:
        """Select an action using epsilon-greedy policy."""
        self.step_count += 1
        if np.random.rand() < self.epsilon():
            return np.random.randint(self.action_space)  # Explore
        else:
            q = self.q_table[state]
            return int(np.random.choice(np.flatnonzero(q == q.max())))  # Exploit
    
    def update(self, state: Tuple[int, int, int], action: int, reward: float, next_state: Tuple[int, int, int], done: bool):
        """Update the Q-value for the given state and action."""
        current_q = self.q_table[state][action]
        max_next_q = 0 if done else reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state][action] = (1 - self.alpha) * current_q + self.alpha * max_next_q