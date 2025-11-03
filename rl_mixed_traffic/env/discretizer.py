import gymnasium as gym
import numpy as np
from dataclasses import dataclass

class DiscretizeActionWrapper(gym.ActionWrapper):
    """
    A Gym Action Wrapper that discretizes a continuous action space into a discrete one.

    Parameters:
    - env: The original Gym environment with a continuous action space.
    - n_bins: The number of discrete bins to create for each dimension of the action space.
    """

    def __init__(self, env: gym.Env, n_bins: int):
        super().__init__(env)
        assert isinstance(env.action_space, gym.spaces.Box) and env.action_space.shape == (1,)
        low, high = float(env.action_space.low[0]), float(env.action_space.high[0])
        self.actions = np.linspace(low, high, n_bins, dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.actions))

    def action(self, action_idx: int) -> np.ndarray:
        a = np.array([self.actions[int(action_idx)]], dtype=np.float32)
        return a
    
    def __getattr__(self, name):
        return getattr(self.env, name)
    

@dataclass
class DiscretizerConfig:
    # number of bins per dimension; match your obs shape (2 * num_vehicles)
    bins_per_dim: int = 8
    clip_min: float = 0.0
    clip_max: float = 1.0
    # optional nonuniform binning: e.g., more resolution near low gaps/speeds
    use_position_heavy_tail: bool = True

class StateDiscretizer:
    """
    Turn continuous obs (normalized in [0,1]) into a tuple of bin indices.
    Works with shape (2*N,).
    """
    def __init__(self, obs_dim: int, cfg: DiscretizerConfig = DiscretizerConfig()):
        self.obs_dim = obs_dim
        self.cfg = cfg
        self.edges = self._make_edges()

    def _make_edges(self):
        edges = []
        # half dims are speeds, half are positions by your design
        half = self.obs_dim // 2
        for i in range(self.obs_dim):
            if i < half:
                # speeds: uniform bins
                e = np.linspace(self.cfg.clip_min, self.cfg.clip_max, self.cfg.bins_per_dim + 1)
            else:
                # positions: optionally emphasize small headways (more bins near 0)
                if self.cfg.use_position_heavy_tail:
                    # quadratic spacing toward 0
                    u = np.linspace(0.0, 1.0, self.cfg.bins_per_dim + 1)
                    e = (u**2) * (self.cfg.clip_max - self.cfg.clip_min) + self.cfg.clip_min
                else:
                    e = np.linspace(self.cfg.clip_min, self.cfg.clip_max, self.cfg.bins_per_dim + 1)
            edges.append(e)
        return edges

    def __call__(self, obs: np.ndarray) -> tuple[int, ...]:
        x = np.asarray(obs, dtype=np.float32)
        assert x.shape == (self.obs_dim,), f"Expected obs_dim={self.obs_dim}, got {x.shape}"
        x = np.clip(x, self.cfg.clip_min, self.cfg.clip_max)
        idxs = []
        for i, e in enumerate(self.edges):
            # np.digitize returns 1..len(e)-1 for in-range; convert to 0..bins-1
            b = int(np.digitize(x[i], e[1:-1], right=False))
            idxs.append(b)
        return tuple(idxs)