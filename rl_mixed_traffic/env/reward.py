from dataclasses import dataclass

@dataclass
class RewardConfig:
    """Configuration for reward calculation in mixed traffic RL environment."""

    # TTC Penalty Parameters
    ttc_threshold: float = 0.6
    ttc_weight: float  = 0.15
    ttc_penalty_base: float = -0.02

    # Headway Distance
    gap_threshold: float = 15.0  # meters
    gap_penalty: float = -1.0
    gap_weight: float = 1.0

    # Jerk Penalty
    jerk_weight: float = 0.2