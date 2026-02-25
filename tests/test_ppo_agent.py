"""Tests for PPOAgent with CleanRL-aligned action semantics."""

import numpy as np
import pytest

from rl_mixed_traffic.agents.ppo_agent import PPOAgent
from rl_mixed_traffic.configs.ppo_config import PPOConfig


@pytest.fixture
def agent():
    cfg = PPOConfig(device="cpu")
    return PPOAgent(obs_dim=8, action_dim=1, config=cfg, continuous=True)


@pytest.fixture
def discrete_agent():
    cfg = PPOConfig(device="cpu")
    return PPOAgent(obs_dim=8, action_dim=5, config=cfg, continuous=False)


class TestActContinuous:
    def test_act_eval_returns_mean(self, agent):
        """Eval mode should return the raw mean (no tanh)."""
        state = np.zeros(8, dtype=np.float32)
        action = agent.act(state, eval_mode=True)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)

    def test_act_eval_is_deterministic(self, agent):
        state = np.random.randn(8).astype(np.float32)
        a1 = agent.act(state, eval_mode=True)
        a2 = agent.act(state, eval_mode=True)
        np.testing.assert_array_equal(a1, a2)

    def test_act_train_is_stochastic(self, agent):
        state = np.random.randn(8).astype(np.float32)
        actions = [agent.act(state, eval_mode=False) for _ in range(20)]
        # At least some actions should differ
        unique = len(set(a.item() for a in actions))
        assert unique > 1, "Training actions should be stochastic"

    def test_act_train_unbounded(self, agent):
        """Raw Gaussian samples should not be clamped to [-1, 1]."""
        agent.network.actor_log_std.data.fill_(2.0)
        state = np.random.randn(8).astype(np.float32)
        max_abs = max(abs(agent.act(state).item()) for _ in range(200))
        assert max_abs > 1.0, "Actions should be unbounded (no tanh)"


class TestActDiscrete:
    def test_act_returns_int(self, discrete_agent):
        state = np.random.randn(8).astype(np.float32)
        action = discrete_agent.act(state)
        assert isinstance(action, int)
        assert 0 <= action < 5

    def test_act_eval_is_deterministic(self, discrete_agent):
        state = np.random.randn(8).astype(np.float32)
        a1 = discrete_agent.act(state, eval_mode=True)
        a2 = discrete_agent.act(state, eval_mode=True)
        assert a1 == a2


class TestGetActionAndValue:
    def test_continuous_returns(self, agent):
        state = np.random.randn(8).astype(np.float32)
        action, value, log_prob = agent.get_action_and_value(state)
        assert isinstance(action, np.ndarray)
        assert action.shape == (1,)
        assert isinstance(value, float)
        assert isinstance(log_prob, float)

    def test_discrete_returns(self, discrete_agent):
        state = np.random.randn(8).astype(np.float32)
        action, value, log_prob = discrete_agent.get_action_and_value(state)
        assert isinstance(action, int)
        assert isinstance(value, float)
        assert isinstance(log_prob, float)


class TestStoreAndLearn:
    def test_store_and_learn_cycle(self, agent):
        """Full rollout -> learn cycle should run without errors."""
        state = np.random.randn(8).astype(np.float32)
        for _ in range(64):
            action, value, log_prob = agent.get_action_and_value(state)
            reward = np.random.randn()
            done = False
            agent.store_transition(state, action, reward, value, log_prob, done)
            state = np.random.randn(8).astype(np.float32)

        metrics = agent.learn(last_value=0.0)
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert "clipfrac" in metrics

    def test_learn_clears_buffer(self, agent):
        state = np.random.randn(8).astype(np.float32)
        for _ in range(16):
            action, value, log_prob = agent.get_action_and_value(state)
            agent.store_transition(state, action, 0.0, value, log_prob, False)
        agent.learn(last_value=0.0)
        assert len(agent.buffer) == 0


class TestSaveLoad:
    def test_save_load_roundtrip(self, agent, tmp_path):
        state = np.random.randn(8).astype(np.float32)
        action_before = agent.act(state, eval_mode=True)

        path = str(tmp_path / "test_agent.pth")
        agent.save(path)

        cfg = PPOConfig(device="cpu")
        agent2 = PPOAgent(obs_dim=8, action_dim=1, config=cfg, continuous=True)
        agent2.load(path)

        action_after = agent2.act(state, eval_mode=True)
        np.testing.assert_array_almost_equal(action_before, action_after)
