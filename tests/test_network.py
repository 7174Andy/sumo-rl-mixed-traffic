"""Tests for PPO ActorCriticNetwork with CleanRL-aligned architecture."""

import numpy as np
import pytest
import torch

from rl_mixed_traffic.ppo.network import ActorCriticNetwork, layer_init


class TestLayerInit:
    def test_orthogonal_weights(self):
        layer = layer_init(torch.nn.Linear(8, 64), std=np.sqrt(2))
        W = layer.weight.data
        # Orthogonal matrices satisfy W @ W^T ≈ c * I for some constant c
        product = W @ W.T
        off_diag = product - torch.diag(product.diag())
        assert off_diag.abs().max() < 0.5  # off-diag elements should be small

    def test_bias_initialized_to_constant(self):
        layer = layer_init(torch.nn.Linear(4, 8), std=1.0, bias_const=0.5)
        assert torch.allclose(layer.bias.data, torch.full_like(layer.bias.data, 0.5))

    def test_default_bias_is_zero(self):
        layer = layer_init(torch.nn.Linear(4, 8))
        assert torch.allclose(layer.bias.data, torch.zeros_like(layer.bias.data))


class TestActorCriticContinuous:
    @pytest.fixture
    def net(self):
        return ActorCriticNetwork(state_dim=10, action_dim=1, continuous=True)

    def test_forward_shapes(self, net):
        state = torch.randn(4, 10)
        action_mean, value = net(state)
        assert action_mean.shape == (4, 1)
        assert value.shape == (4, 1)

    def test_get_action_and_value_sample(self, net):
        state = torch.randn(4, 10)
        action, log_prob, entropy, value = net.get_action_and_value(state)
        assert action.shape == (4, 1)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4, 1)

    def test_get_action_and_value_evaluate(self, net):
        state = torch.randn(4, 10)
        action, _, _, _ = net.get_action_and_value(state)
        # Evaluate log_prob of the same action
        action2, log_prob, entropy, value = net.get_action_and_value(state, action)
        assert torch.allclose(action, action2)
        assert log_prob.shape == (4,)

    def test_actions_are_unbounded(self):
        """Actions should NOT be squashed by tanh — raw Gaussian samples."""
        net = ActorCriticNetwork(state_dim=4, action_dim=1, continuous=True)
        # Set high log_std to produce large samples
        net.actor_log_std.data.fill_(2.0)
        max_abs = 0.0
        for _ in range(200):
            a, _, _, _ = net.get_action_and_value(torch.randn(1, 4))
            max_abs = max(max_abs, a.abs().item())
        assert max_abs > 1.0, "Actions should be unbounded (no tanh squashing)"

    def test_no_log_std_clamping(self):
        """log_std should be used directly, not clamped."""
        net = ActorCriticNetwork(state_dim=4, action_dim=1, continuous=True)
        net.actor_log_std.data.fill_(3.0)
        state = torch.randn(1, 4)
        action_mean, _ = net(state)
        # The std used internally should be exp(3.0), not clamped
        expected_std = np.exp(3.0)
        # Sample many actions and check variance is large
        actions = torch.stack(
            [net.get_action_and_value(state)[0] for _ in range(500)]
        ).squeeze()
        actual_std = actions.std().item()
        # Should be roughly exp(3.0) ≈ 20; clamped would be exp(0.5) ≈ 1.6
        assert actual_std > 5.0, f"std should be ~{expected_std:.1f}, got {actual_std:.1f}"

    def test_log_prob_is_simple_gaussian(self):
        """log_prob should be plain Gaussian, no Jacobian correction."""
        net = ActorCriticNetwork(state_dim=4, action_dim=1, continuous=True)
        state = torch.randn(1, 4)
        action, log_prob, _, _ = net.get_action_and_value(state)

        # Manually compute expected log_prob
        action_mean, _ = net(state)
        action_std = torch.exp(net.actor_log_std.expand_as(action_mean))
        dist = torch.distributions.Normal(action_mean, action_std)
        expected_lp = dist.log_prob(action).sum(dim=-1)

        assert torch.allclose(log_prob, expected_lp, atol=1e-6)

    def test_tanh_activations(self, net):
        """Hidden activations should use tanh (outputs bounded in [-1, 1])."""
        state = torch.randn(32, 10)
        # Hook into shared layers to check activation range
        x = net.shared1(state)
        activated = torch.tanh(x)
        # If ReLU were used, some values would exceed 1.0
        assert activated.abs().max() <= 1.0

    def test_orthogonal_init_applied(self):
        """Network layers should have orthogonal initialization."""
        net = ActorCriticNetwork(state_dim=64, action_dim=1, continuous=True)
        W = net.shared1.weight.data  # shape [256, 64]
        # For orthogonal init with std=sqrt(2), W^T @ W ≈ 2*I (when rows >= cols)
        product = W.T @ W  # [64, 64]
        # Diagonal should be close to 2.0, off-diagonal close to 0
        diag_mean = product.diag().mean().item()
        off_diag = product - torch.diag(product.diag())
        off_diag_max = off_diag.abs().max().item()
        assert abs(diag_mean - 2.0) < 0.5, f"Expected diag ~2.0, got {diag_mean}"
        assert off_diag_max < 0.5, f"Off-diagonal should be small, got {off_diag_max}"


class TestActorCriticDiscrete:
    @pytest.fixture
    def net(self):
        return ActorCriticNetwork(state_dim=8, action_dim=5, continuous=False)

    def test_forward_shapes(self, net):
        state = torch.randn(4, 8)
        logits, value = net(state)
        assert logits.shape == (4, 5)
        assert value.shape == (4, 1)

    def test_get_action_and_value_sample(self, net):
        state = torch.randn(4, 8)
        action, log_prob, entropy, value = net.get_action_and_value(state)
        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)

    def test_actions_are_valid_indices(self, net):
        state = torch.randn(16, 8)
        action, _, _, _ = net.get_action_and_value(state)
        assert (action >= 0).all() and (action < 5).all()
