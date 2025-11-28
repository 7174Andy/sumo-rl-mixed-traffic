from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict, Tuple

class BaseAgent(ABC):
    """Abstract base class for all RL agents."""

    @abstractmethod
    def act(self, state:np.ndarray, eval_mode: bool = False) -> int:
        """Select an action given current state."""
        pass

    @abstractmethod
    def update(self, *args: Any, **kwargs: Any) -> None:
        """Update the agent's knowledge based on experience."""
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save the agent's parameters to a file."""
        pass

    @abstractmethod
    def load(self, path: str, map_location=None) -> None:
        """Load the agent's parameters from a file."""
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """Get the agent's configuration as a dictionary."""
        pass