from dataclasses import dataclass


@dataclass
class QLearningConfig:
    """Configuration for Q-Learning agent."""

    # Learning parameters
    alpha: float = 0.2  # Learning rate
    gamma: float = 0.98  # Discount factor

    # Exploration parameters
    eps_start: float = 1.0  # Initial exploration rate
    eps_end: float = 0.1  # Final exploration rate
    eps_decay_steps: int = 15_000  # Steps over which to decay epsilon

    # Action space
    action_space: int = 7  # Number of discrete actions (will be set by agent)

    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {self.alpha}")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {self.gamma}")
        if not 0.0 <= self.eps_start <= 1.0:
            raise ValueError(f"eps_start must be in [0, 1], got {self.eps_start}")
        if not 0.0 <= self.eps_end <= 1.0:
            raise ValueError(f"eps_end must be in [0, 1], got {self.eps_end}")
        if self.eps_decay_steps <= 0:
            raise ValueError(f"eps_decay_steps must be positive, got {self.eps_decay_steps}")
        if self.action_space <= 0:
            raise ValueError(f"action_space must be positive, got {self.action_space}")
