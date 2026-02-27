"""Tests for DiscretizeActionWrapper and StateDiscretizer.

No SUMO needed — tests pure discretization math.
"""

import pytest
import numpy as np
import gymnasium as gym
import types
import sys
import os

os.environ.setdefault("SUMO_HOME", "/tmp/fake_sumo")
sys.modules.setdefault(
    "traci",
    types.ModuleType("traci"),
)

from rl_mixed_traffic.env.discretizer import (
    DiscretizeActionWrapper,
    StateDiscretizer,
    DiscretizerConfig,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeBoxEnv(gym.Env):
    """Minimal env with a Box(1,) action space for testing the wrapper."""

    def __init__(self, low=-3.0, high=3.0):
        self.action_space = gym.spaces.Box(
            low=np.float32(low), high=np.float32(high), shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))
        self.last_action = None

    def step(self, action):
        self.last_action = action
        return np.zeros(4, dtype=np.float32), 0.0, False, False, {}

    def reset(self, **kwargs):
        return np.zeros(4, dtype=np.float32), {}


# ---------------------------------------------------------------------------
# DiscretizeActionWrapper
# ---------------------------------------------------------------------------
class TestDiscretizeActionWrapper:
    def test_action_space_is_discrete(self):
        env = DiscretizeActionWrapper(_FakeBoxEnv(), n_bins=5)
        assert isinstance(env.action_space, gym.spaces.Discrete)
        assert env.action_space.n == 5

    def test_bins_span_action_range(self):
        env = DiscretizeActionWrapper(_FakeBoxEnv(low=-3.0, high=3.0), n_bins=7)
        assert env.actions[0] == pytest.approx(-3.0)
        assert env.actions[-1] == pytest.approx(3.0)

    def test_bins_evenly_spaced(self):
        env = DiscretizeActionWrapper(_FakeBoxEnv(low=-2.0, high=2.0), n_bins=5)
        diffs = np.diff(env.actions)
        assert np.allclose(diffs, diffs[0]), "Bins should be evenly spaced"

    def test_action_maps_index_to_continuous(self):
        env = DiscretizeActionWrapper(_FakeBoxEnv(low=-3.0, high=3.0), n_bins=7)
        # Index 0 -> -3.0, index 6 -> 3.0
        a0 = env.action(0)
        a6 = env.action(6)
        assert a0[0] == pytest.approx(-3.0)
        assert a6[0] == pytest.approx(3.0)

    def test_action_returns_numpy_array(self):
        env = DiscretizeActionWrapper(_FakeBoxEnv(), n_bins=5)
        a = env.action(2)
        assert isinstance(a, np.ndarray)
        assert a.shape == (1,)
        assert a.dtype == np.float32

    def test_middle_bin_is_zero_for_symmetric_range(self):
        env = DiscretizeActionWrapper(_FakeBoxEnv(low=-3.0, high=3.0), n_bins=7)
        mid = env.action(3)
        assert mid[0] == pytest.approx(0.0)

    def test_step_passes_continuous_to_inner_env(self):
        inner = _FakeBoxEnv(low=-3.0, high=3.0)
        env = DiscretizeActionWrapper(inner, n_bins=7)
        env.step(0)  # discrete action 0
        assert inner.last_action[0] == pytest.approx(-3.0)

    def test_n_bins_one(self):
        """Single bin should map to the midpoint of the range."""
        env = DiscretizeActionWrapper(_FakeBoxEnv(low=-2.0, high=2.0), n_bins=1)
        assert env.action_space.n == 1
        # linspace(-2, 2, 1) = [0.0] — the midpoint (np.linspace with n=1 returns start)
        a = env.action(0)
        assert a[0] == pytest.approx(-2.0)


# ---------------------------------------------------------------------------
# StateDiscretizer
# ---------------------------------------------------------------------------
class TestStateDiscretizer:
    def test_output_is_tuple_of_ints(self):
        sd = StateDiscretizer(obs_dim=4)
        obs = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float32)
        result = sd(obs)
        assert isinstance(result, tuple)
        assert all(isinstance(x, int) for x in result)
        assert len(result) == 4

    def test_output_length_matches_obs_dim(self):
        sd = StateDiscretizer(obs_dim=6)
        obs = np.random.rand(6).astype(np.float32)
        assert len(sd(obs)) == 6

    def test_zeros_map_to_first_bin(self):
        sd = StateDiscretizer(obs_dim=4, cfg=DiscretizerConfig(bins_per_dim=8))
        obs = np.zeros(4, dtype=np.float32)
        result = sd(obs)
        assert all(b == 0 for b in result)

    def test_ones_map_to_last_bin(self):
        sd = StateDiscretizer(obs_dim=4, cfg=DiscretizerConfig(bins_per_dim=8))
        obs = np.ones(4, dtype=np.float32)
        result = sd(obs)
        assert all(b == 7 for b in result), f"Expected all 7, got {result}"

    def test_values_clipped_to_range(self):
        """Values outside [clip_min, clip_max] should be clipped."""
        sd = StateDiscretizer(obs_dim=2, cfg=DiscretizerConfig(bins_per_dim=4))
        obs = np.array([-0.5, 1.5], dtype=np.float32)
        result = sd(obs)
        # Should not crash; clipped values land in first/last bin
        assert result[0] == 0
        assert result[1] == 3

    def test_wrong_obs_dim_raises(self):
        sd = StateDiscretizer(obs_dim=4)
        with pytest.raises(AssertionError, match="Expected obs_dim=4"):
            sd(np.zeros(6, dtype=np.float32))

    def test_midpoint_maps_to_middle_bin(self):
        sd = StateDiscretizer(obs_dim=2, cfg=DiscretizerConfig(bins_per_dim=4))
        obs = np.array([0.5, 0.5], dtype=np.float32)
        result = sd(obs)
        # With 4 bins, edges are [0, 0.25, 0.5, 0.75, 1.0]
        # 0.5 falls at the boundary between bin 1 and bin 2
        assert result[0] in (1, 2)
        assert result[1] in (1, 2)

    def test_position_heavy_tail_mode(self):
        """Position dims should use quadratic spacing when enabled."""
        cfg = DiscretizerConfig(bins_per_dim=4, use_position_heavy_tail=True)
        sd = StateDiscretizer(obs_dim=4, cfg=cfg)  # dim 0,1 = speed; 2,3 = position

        # Verify that position edges are non-uniform (quadratic)
        speed_edges = sd.edges[0]
        pos_edges = sd.edges[2]

        speed_diffs = np.diff(speed_edges)
        pos_diffs = np.diff(pos_edges)

        # Speed edges should be uniform
        assert np.allclose(speed_diffs, speed_diffs[0], atol=1e-6)
        # Position edges should NOT be uniform (quadratic spacing)
        assert not np.allclose(pos_diffs, pos_diffs[0], atol=1e-6)

    def test_uniform_mode_all_edges_equal(self):
        """Without heavy tail, speed and position edges are identical."""
        cfg = DiscretizerConfig(bins_per_dim=4, use_position_heavy_tail=False)
        sd = StateDiscretizer(obs_dim=4, cfg=cfg)
        np.testing.assert_array_almost_equal(sd.edges[0], sd.edges[2])

    def test_deterministic(self):
        """Same input should always produce same output."""
        sd = StateDiscretizer(obs_dim=4)
        obs = np.array([0.1, 0.4, 0.7, 0.9], dtype=np.float32)
        assert sd(obs) == sd(obs)
