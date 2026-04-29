"""Tests for NNMPCActorCritic — warm-start fidelity, residual zero-init,
and distribution math."""

import math
import numpy as np
import pytest
import torch

from rl_mixed_traffic.deep_lcc.nnmpc_actor_critic import NNMPCActorCritic
from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork


@pytest.fixture
def fake_nnmpc_ckpt(tmp_path):
    """Build a small NNMPCNetwork checkpoint to use for warm-start tests."""
    nnmpc = NNMPCNetwork(input_dim=260, output_dim=2, hidden_dims=(256, 128))
    ckpt = {
        "model_state_dict": nnmpc.state_dict(),
        "input_mean": np.zeros(260, dtype=np.float32),
        "input_std": np.ones(260, dtype=np.float32),
        "input_dim": 260,
        "output_dim": 2,
        "config": {
            "hidden_dims": (256, 128),
            "accel_min": -5.0,
            "accel_max": 3.0,
        },
    }
    p = tmp_path / "fake_nnmpc.pth"
    torch.save(ckpt, p)
    return p, nnmpc


class TestForwardShape:
    def test_actor_output_and_value_shapes(self):
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        x = torch.zeros(4, 260)
        actor_out, value = net(x)
        assert actor_out.shape == (4, 2)
        assert value.shape == (4, 1)


class TestWarmStart:
    def test_warm_start_replicates_nnmpc_pre_tanh_output(self, fake_nnmpc_ckpt):
        ckpt_path, nnmpc = fake_nnmpc_ckpt
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        net.warm_start_from_nnmpc(str(ckpt_path))

        x = torch.randn(8, 260)

        # NNMPC output: net(x) returns the post-tanh, scaled action in [-5, 3].
        nn_out = nnmpc(x)

        # NNMPCActorCritic actor body produces the pre-tanh logits;
        # tanh + scale to [-5, 3] should reproduce NNMPC's output.
        with torch.no_grad():
            pre_tanh, _ = net(x)
        accel_min, accel_max = -5.0, 3.0
        scaled = (torch.tanh(pre_tanh) + 1.0) / 2.0 * (accel_max - accel_min) + accel_min

        assert torch.allclose(scaled, nn_out, atol=1e-5)


class TestResidualInit:
    def test_residual_init_outputs_near_zero(self):
        net = NNMPCActorCritic(
            obs_dim=260, action_dim=2,
            log_std_init=math.log(0.5),
            final_layer_gain=0.01,
        )
        x = torch.randn(16, 260)
        with torch.no_grad():
            pre_tanh, _ = net(x)
        # Pre-tanh logits should be close to zero so tanh(.) ≈ 0.
        assert pre_tanh.abs().mean().item() < 0.1


class TestGetActionAndValue:
    def test_signature_and_log_prob_shape(self):
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        x = torch.zeros(5, 260)
        action, log_prob, entropy, value = net.get_action_and_value(x)
        assert action.shape == (5, 2)
        # Action sampled then tanh-squashed to [-1, 1]
        assert (action >= -1.0).all() and (action <= 1.0).all()
        assert log_prob.shape == (5,)
        assert entropy.shape == (5,)
        assert value.shape == (5, 1)

    def test_log_prob_with_provided_action(self):
        torch.manual_seed(0)
        net = NNMPCActorCritic(obs_dim=260, action_dim=2)
        x = torch.zeros(3, 260)
        action = torch.tensor([[0.0, 0.0], [0.5, -0.5], [-0.9, 0.9]])
        _, log_prob, entropy, value = net.get_action_and_value(x, action=action)
        assert log_prob.shape == (3,)
        assert torch.isfinite(log_prob).all()
        assert torch.isfinite(entropy).all()


class TestLogStdInit:
    def test_warm_start_uses_lower_std(self):
        net_warm = NNMPCActorCritic(obs_dim=260, action_dim=2,
                                     log_std_init=math.log(0.1))
        net_resid = NNMPCActorCritic(obs_dim=260, action_dim=2,
                                      log_std_init=math.log(0.5))
        assert net_warm.actor_log_std.mean().item() < net_resid.actor_log_std.mean().item()
