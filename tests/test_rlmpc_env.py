"""Tests for PlatoonNNMPCEnv pieces that don't require SUMO.

Mocks traci where needed; pure-logic tests for ObservationBuilder.
"""

import os
import sys
import types

import numpy as np
import pytest

os.environ.setdefault("SUMO_HOME", "/tmp/fake_sumo")
_fake_traci = types.ModuleType("traci")
_fake_traci.vehicle = types.SimpleNamespace(
    getIDList=lambda: [],
    getSpeed=lambda vid: 0.0,
    getDistance=lambda vid: 0.0,
    getLength=lambda vid: 5.0,
    getLeader=lambda vid: None,
    getAcceleration=lambda vid: 0.0,
    setSpeed=lambda vid, speed: None,
    setSpeedMode=lambda vid, mode: None,
    setMaxSpeed=lambda vid, mx: None,
    setTau=lambda vid, t: None,
    setAccel=lambda vid, a: None,
    setDecel=lambda vid, d: None,
    setMinGap=lambda vid, g: None,
    setImperfection=lambda vid, s: None,
)
sys.modules.setdefault("traci", _fake_traci)
_fake_sumolib = types.ModuleType("sumolib")
_fake_sumolib.checkBinary = lambda x: x
sys.modules.setdefault("sumolib", _fake_sumolib)


from rl_mixed_traffic.deep_lcc.rlmpc_env import ObservationBuilder


class TestObservationBuilder:
    def test_empty_buffers_have_correct_shapes(self):
        b = ObservationBuilder(T_ini=20, n_followers=8, m_ctr=2)
        # Before any push, buffers are zero-padded
        obs = b.build_obs()
        assert obs.shape == (260,)
        assert obs.dtype == np.float32
        assert (obs == 0.0).all()

    def test_uini_yini_eini_layout(self):
        b = ObservationBuilder(T_ini=2, n_followers=3, m_ctr=1)
        # T_ini=2, n_followers=3, m_ctr=1 → p_ctr = 4
        # uini: 2*1 = 2, yini: 2*4 = 8, eini: 2*1 = 2 → total 12

        # Push step 0
        b.push_step(
            uini_step=np.array([1.0]),
            yini_step=np.array([10.0, 11.0, 12.0, 100.0]),
            eini_step=np.array([0.5]),
        )
        # Push step 1
        b.push_step(
            uini_step=np.array([2.0]),
            yini_step=np.array([20.0, 21.0, 22.0, 200.0]),
            eini_step=np.array([0.6]),
        )

        obs = b.build_obs()
        assert obs.shape == (12,)
        # uini: [1.0, 2.0]
        np.testing.assert_array_equal(obs[:2], [1.0, 2.0])
        # yini: [10, 11, 12, 100, 20, 21, 22, 200]
        np.testing.assert_array_equal(obs[2:10], [10, 11, 12, 100, 20, 21, 22, 200])
        # eini: [0.5, 0.6] — use almost_equal because 0.6 is not exactly representable
        # in float32 (the dtype build_obs returns).
        np.testing.assert_array_almost_equal(obs[10:12], [0.5, 0.6])

    def test_normalize_uses_provided_stats(self):
        b = ObservationBuilder(T_ini=2, n_followers=3, m_ctr=1)
        b.set_normalization(
            mean=np.full(12, 2.0, dtype=np.float32),
            std=np.full(12, 4.0, dtype=np.float32),
        )
        b.push_step(
            uini_step=np.array([10.0]),
            yini_step=np.array([2.0, 2.0, 2.0, 2.0]),
            eini_step=np.array([6.0]),
        )
        b.push_step(
            uini_step=np.array([10.0]),
            yini_step=np.array([2.0, 2.0, 2.0, 2.0]),
            eini_step=np.array([6.0]),
        )
        # mean=2, std=4 → uini values (10) → (10-2)/4 = 2; yini → 0; eini → 1
        norm_obs = b.build_normalized_obs()
        np.testing.assert_array_almost_equal(norm_obs[:2], [2.0, 2.0])
        np.testing.assert_array_almost_equal(norm_obs[2:10], np.zeros(8))
        np.testing.assert_array_almost_equal(norm_obs[10:12], [1.0, 1.0])

    def test_overflow_drops_oldest(self):
        b = ObservationBuilder(T_ini=2, n_followers=1, m_ctr=1)
        # Push 3 steps; only the last 2 survive.
        for k in range(3):
            b.push_step(
                uini_step=np.array([float(k)]),
                yini_step=np.array([float(k), float(k)]),
                eini_step=np.array([float(k)]),
            )
        obs = b.build_obs()
        # uini: [1.0, 2.0]
        np.testing.assert_array_equal(obs[:2], [1.0, 2.0])


# ----------------------------------------------------------------------
# PlatoonNNMPCEnv tests (mocked traci)
# ----------------------------------------------------------------------

from unittest.mock import patch

from rl_mixed_traffic.deep_lcc.rlmpc_config import RLMPCConfig
from rl_mixed_traffic.deep_lcc.rlmpc_env import PlatoonNNMPCEnv


@pytest.fixture
def stub_nnmpc_ckpt(tmp_path):
    """Save a small NNMPC checkpoint to disk for env construction."""
    import torch
    from rl_mixed_traffic.deep_lcc.nnmpc_network import NNMPCNetwork

    nnmpc = NNMPCNetwork(input_dim=260, output_dim=2, hidden_dims=(256, 128))
    p = tmp_path / "nnmpc.pth"
    torch.save({
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
    }, p)
    return str(p)


class TestPlatoonNNMPCEnvConstruction:
    def test_warm_start_action_space(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        assert env.action_space.shape == (2,)
        assert env.action_space.high.tolist() == [3.0, 3.0]
        assert env.action_space.low.tolist() == [-5.0, -5.0]

    def test_residual_action_space(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="residual", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        assert env.action_space.shape == (2,)
        assert env.action_space.high.tolist() == [2.0, 2.0]
        assert env.action_space.low.tolist() == [-2.0, -2.0]

    def test_observation_space_shape(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        assert env.observation_space.shape == (260,)

    def test_compose_action_residual_mode_clips_total(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="residual", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)

        u_total = env._compose_action(
            policy_action=np.array([2.0, 2.0]),
            u_nnmpc=np.array([3.0, 3.0]),
        )
        np.testing.assert_array_equal(u_total, [3.0, 3.0])

        u_total = env._compose_action(
            policy_action=np.array([-1.0, -1.0]),
            u_nnmpc=np.array([-5.0, -5.0]),
        )
        np.testing.assert_array_equal(u_total, [-5.0, -5.0])

        u_total = env._compose_action(
            policy_action=np.array([0.5, -0.5]),
            u_nnmpc=np.array([1.0, -1.0]),
        )
        np.testing.assert_array_almost_equal(u_total, [1.5, -1.5])

    def test_compose_action_warm_start_uses_policy_directly(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        u_total = env._compose_action(
            policy_action=np.array([2.5, -3.0]),
            u_nnmpc=np.zeros(2),
        )
        np.testing.assert_array_almost_equal(u_total, [2.5, -3.0])


class TestRewardAugmentation:
    def test_collision_overrides_to_minus_one(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt)
        env = PlatoonNNMPCEnv(cfg)
        r = env._augment_reward(r_base=0.5, violation=0.2, collision=True)
        assert r == -1.0

    def test_lagrangian_penalty_subtracts(self, stub_nnmpc_ckpt):
        cfg = RLMPCConfig(mode="warm_start", nnmpc_path=stub_nnmpc_ckpt,
                          lambda_violation=2.0)
        env = PlatoonNNMPCEnv(cfg)
        r = env._augment_reward(r_base=0.5, violation=0.1, collision=False)
        assert abs(r - (0.5 - 2.0 * 0.1)) < 1e-6
