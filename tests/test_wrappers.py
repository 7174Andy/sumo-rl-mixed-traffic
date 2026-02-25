"""Tests for FourToFiveTupleWrapper."""

import gymnasium as gym
import numpy as np

from rl_mixed_traffic.env.wrappers import FourToFiveTupleWrapper


class FakeEnv4Tuple(gym.Env):
    """Minimal env that returns old-style 4-tuples from step()."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        obs = np.zeros(4, dtype=np.float32)
        reward = 1.0
        done = False
        info = {"key": "val"}
        return obs, reward, done, info


class FakeEnv5Tuple(gym.Env):
    """Env that already returns 5-tuples."""

    def __init__(self):
        self.observation_space = gym.spaces.Box(low=-1, high=1, shape=(4,))
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(4, dtype=np.float32), 1.0, False, False, {}


class TestFourToFiveTupleWrapper:
    def test_converts_4_tuple_to_5_tuple(self):
        env = FourToFiveTupleWrapper(FakeEnv4Tuple())
        result = env.step(np.array([0.0]))
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert reward == 1.0
        assert terminated is False
        assert truncated is False
        assert info == {"key": "val"}

    def test_passes_through_5_tuple(self):
        env = FourToFiveTupleWrapper(FakeEnv5Tuple())
        result = env.step(np.array([0.0]))
        assert len(result) == 5

    def test_reset_returns_tuple(self):
        env = FourToFiveTupleWrapper(FakeEnv4Tuple())
        result = env.reset()
        assert len(result) == 2
        obs, info = result
        assert obs.shape == (4,)

    def test_works_with_clip_action(self):
        """Wrapper chain: FourToFiveTuple -> ClipAction should work."""
        env = FourToFiveTupleWrapper(FakeEnv4Tuple())
        env = gym.wrappers.ClipAction(env)
        obs, info = env.reset()
        # Action outside bounds should be clipped
        result = env.step(np.array([5.0]))
        assert len(result) == 5

    def test_works_with_normalize_reward(self):
        """Full training wrapper chain should work."""
        env = FourToFiveTupleWrapper(FakeEnv4Tuple())
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeReward(env, gamma=0.99)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        obs, info = env.reset()
        result = env.step(np.array([0.5]))
        assert len(result) == 5
        _, reward, _, _, _ = result
        assert -10 <= reward <= 10
