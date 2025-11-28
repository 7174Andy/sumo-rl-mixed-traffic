from collections import defaultdict
import numpy as np
import pickle
from typing import Tuple, Union, Dict, Any
from pathlib import Path

from rl_mixed_traffic.agents.base_agent import BaseAgent
from rl_mixed_traffic.configs.q_config import QLearningConfig


class QLearningAgent(BaseAgent):
    """Tabular Q-Learning agent with epsilon-greedy exploration.

    Implements the classic Q-learning algorithm using a Q-table stored as a defaultdict.
    States are expected to be tuples of integers (discretized observations).

    Args:
        action_space: Number of discrete actions available
        alpha: Learning rate (default: 0.2)
        gamma: Discount factor (default: 0.98)
        eps_start: Initial exploration rate (default: 1.0)
        eps_end: Final exploration rate (default: 0.1)
        eps_decay_steps: Steps over which to decay epsilon (default: 15,000)
        config: Optional QLearningConfig object (overrides individual parameters)

    Example:
        >>> config = QLearningConfig(alpha=0.1, gamma=0.99)
        >>> agent = QLearningAgent(action_space=7, config=config)
        >>> state = (0, 1, 2)  # Discretized state tuple
        >>> action = agent.act(state)
        >>> agent.update(state, action, reward=1.0, next_state=(0, 1, 3), done=False)
    """

    def __init__(
        self,
        action_space: int,
        alpha: float = 0.2,
        gamma: float = 0.98,
        eps_start: float = 1.0,
        eps_end: float = 0.1,
        eps_decay_steps: int = 15_000,
        config: QLearningConfig | None = None,
    ):
        if config is not None:
            # Use config if provided
            self.alpha = config.alpha
            self.gamma = config.gamma
            self.eps_start = config.eps_start
            self.eps_end = config.eps_end
            self.eps_decay_steps = config.eps_decay_steps
            self.action_space = action_space  # Override action_space from env
        else:
            # Use individual parameters
            self.alpha = alpha
            self.gamma = gamma
            self.eps_start = eps_start
            self.eps_end = eps_end
            self.eps_decay_steps = eps_decay_steps
            self.action_space = action_space

        self.step_count = 0  # Step counter for epsilon decay
        self.q_table = defaultdict(lambda: np.zeros(self.action_space))

    def epsilon(self) -> float:
        """Calculate the current epsilon value based on decay schedule.

        Returns:
            Current epsilon value linearly interpolated between eps_start and eps_end
        """
        frac = min(1.0, self.step_count / max(1, self.eps_decay_steps))
        return float(self.eps_start + frac * (self.eps_end - self.eps_start))

    def act(self, state: Union[Tuple, np.ndarray], eval_mode: bool = False) -> int:
        """Select an action using epsilon-greedy policy.

        Args:
            state: Current state (tuple of ints for Q-learning, or array that will be converted)
            eval_mode: If True, always exploit (no exploration). Default: False

        Returns:
            Selected action index

        Note:
            This method increments step_count for epsilon decay scheduling.
        """
        # Convert state to tuple if needed (for compatibility with BaseAgent signature)
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist() if state.ndim > 0 else [int(state)])
        elif not isinstance(state, tuple):
            state = tuple(state)

        self.step_count += 1

        # Exploration vs exploitation
        if (not eval_mode) and (np.random.random() < self.epsilon()):
            return int(np.random.randint(self.action_space))  # Explore
        else:
            q = self.q_table[state]
            # Break ties randomly
            return int(np.random.choice(np.flatnonzero(q == q.max())))  # Exploit

    def update(
        self,
        state: Union[Tuple, np.ndarray],
        action: int,
        reward: float,
        next_state: Union[Tuple, np.ndarray],
        done: bool,
    ) -> None:
        """Update the Q-value for the given state and action using the Q-learning update rule.

        Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether the episode terminated

        Note:
            Uses standard Q-learning (off-policy TD(0)) with linear interpolation.
        """
        # Convert states to tuples if needed
        if isinstance(state, np.ndarray):
            state = tuple(state.tolist() if state.ndim > 0 else [int(state)])
        elif not isinstance(state, tuple):
            state = tuple(state)

        if isinstance(next_state, np.ndarray):
            next_state = tuple(next_state.tolist() if next_state.ndim > 0 else [int(next_state)])
        elif not isinstance(next_state, tuple):
            next_state = tuple(next_state)

        current_q = self.q_table[state][action]

        # Fixed TD target calculation (bug fix from refactor plan)
        # Old (incorrect): max_next_q = 0 if done else reward + gamma * max(Q(s'))
        # New (correct): max_next_q = reward + (0 if done else gamma * max(Q(s')))
        if done:
            td_target = reward
        else:
            td_target = reward + self.gamma * np.max(self.q_table[next_state])

        # Q-learning update: Q(s,a) ← (1-α)Q(s,a) + α * td_target
        self.q_table[state][action] = (1 - self.alpha) * current_q + self.alpha * td_target

    def save(self, path: str) -> None:
        """Save the Q-table and agent state to a pickle file.

        Args:
            path: Path to save the agent (will create parent directories if needed)

        Note:
            Saves both the Q-table (as a plain dict) and agent metadata (step_count, config).
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        # Convert defaultdict to plain dict for pickling
        q_table_dict = {k: np.array(v, dtype=np.float32) for k, v in self.q_table.items()}

        state_dict = {
            "q_table": q_table_dict,
            "step_count": self.step_count,
            "config": self.get_config(),
        }

        with open(path, "wb") as f:
            pickle.dump(state_dict, f)

    def load(self, path: str, map_location=None) -> None:
        """Load the Q-table and agent state from a pickle file.

        Args:
            path: Path to the saved agent file
            map_location: Not used (for compatibility with BaseAgent interface)

        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the loaded Q-table has incompatible action space
        """
        with open(path, "rb") as f:
            state_dict = pickle.load(f)

        # Load Q-table
        if "q_table" in state_dict:
            loaded_q_table = state_dict["q_table"]
            # Validate action space compatibility
            if loaded_q_table:
                sample_q_values = next(iter(loaded_q_table.values()))
                if len(sample_q_values) != self.action_space:
                    raise ValueError(
                        f"Loaded Q-table has action space {len(sample_q_values)}, "
                        f"but agent expects {self.action_space}"
                    )

            # Convert back to defaultdict
            self.q_table = defaultdict(lambda: np.zeros(self.action_space))
            for state, q_values in loaded_q_table.items():
                self.q_table[state] = np.array(q_values, dtype=np.float32)

        # Load step count if available
        if "step_count" in state_dict:
            self.step_count = state_dict["step_count"]

        # Optionally update config from saved state
        if "config" in state_dict:
            saved_config = state_dict["config"]
            # Update hyperparameters if they were saved
            if "alpha" in saved_config:
                self.alpha = saved_config["alpha"]
            if "gamma" in saved_config:
                self.gamma = saved_config["gamma"]

    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration as a dictionary.

        Returns:
            Dictionary containing all agent hyperparameters and state
        """
        return {
            "agent_type": "QLearningAgent",
            "action_space": self.action_space,
            "alpha": self.alpha,
            "gamma": self.gamma,
            "eps_start": self.eps_start,
            "eps_end": self.eps_end,
            "eps_decay_steps": self.eps_decay_steps,
            "step_count": self.step_count,
            "q_table_size": len(self.q_table),
        }
