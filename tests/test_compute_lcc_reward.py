"""Unit tests for non-negative DeeP-LCC reward functions.

The reward is r = max(J_max - J, 0) / J_max  where J >= 0 is the raw cost.
At equilibrium (J=0) reward is 1.0; at worst case (J=J_max) reward is 0.0.

Mocks traci calls to test the reward math in isolation (no SUMO needed).
"""

import pytest
import numpy as np
from unittest.mock import patch

from rl_mixed_traffic.env.ring_env import RingRoadEnv
from rl_mixed_traffic.configs.sumo_config import SumoConfig


# ---------------------------------------------------------------------------
# Fake traci state
# ---------------------------------------------------------------------------
_vehicle_data = {}  # vid -> {speed, distance, length}
_leader_data = {}   # vid -> (leader_id, gap) or None
_active_ids = []


def _setup_vehicles(vehicles: dict, leaders: dict):
    """Populate fake traci state."""
    global _active_ids
    _vehicle_data.clear()
    _leader_data.clear()
    _vehicle_data.update(vehicles)
    _leader_data.update(leaders)
    _active_ids = list(vehicles.keys())


@pytest.fixture(autouse=True)
def _mock_traci():
    """Patch all traci.vehicle calls used by the reward methods."""
    with (
        patch("traci.vehicle.getIDList", side_effect=lambda: list(_active_ids)),
        patch("traci.vehicle.getSpeed", side_effect=lambda vid: _vehicle_data[vid]["speed"]),
        patch("traci.vehicle.getDistance", side_effect=lambda vid: _vehicle_data[vid]["distance"]),
        patch("traci.vehicle.getLength", side_effect=lambda vid: _vehicle_data[vid]["length"]),
        patch("traci.vehicle.getLeader", side_effect=lambda vid: _leader_data.get(vid)),
    ):
        yield


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _make_env(**overrides):
    """Create a RingRoadEnv without connecting to SUMO."""
    defaults = dict(
        v_star=15.0,
        s_star=20.0,
        weight_v=0.8,
        weight_s=0.7,
        weight_u=0.1,
        num_vehicles=4,
        num_agents=1,
    )
    defaults.update(overrides)

    cfg = SumoConfig(sumocfg_path="configs/ring/simulation.sumocfg", use_gui=False)
    env = RingRoadEnv(sumo_config=cfg, gui=False, **defaults)
    env.ring_length = 230.0
    return env


def _compute_s_star(v_star, v_max=30.0):
    """Replicate the OVM-derived equilibrium spacing from the env."""
    v_ratio = max(0.0, min(v_star / v_max, 1.0))
    return np.arccos(1 - v_ratio * 2) / np.pi * (35 - 5) + 5


# ---------------------------------------------------------------------------
# Tests: compute_lcc_reward (single-agent)
# ---------------------------------------------------------------------------
class TestSingleAgentLCCReward:

    def test_agent_absent_returns_zero(self):
        env = _make_env()
        _setup_vehicles(
            {"car0": {"speed": 15.0, "distance": 0.0, "length": 5.0}},
            {},
        )
        assert env.compute_lcc_reward() == 0.0

    def test_at_equilibrium_reward_is_one(self):
        """When all vehicles at v_star and gap == s_star, reward == 1.0."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car3": {"speed": 15.0, "distance": 150.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        assert reward == pytest.approx(1.0, abs=1e-6)

    def test_velocity_error_reduces_reward(self):
        """Vehicles deviating from v_star produce reward < 1."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 10.0, "distance": 100.0, "length": 5.0},
                "car3": {"speed": 20.0, "distance": 150.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        # J_vel = 0.8*(25+25) = 40.0
        expected = max(env.J_max - 40.0, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)
        assert 0.0 < reward < 1.0

    def test_spacing_error_reduces_reward(self):
        """Gap deviating from s_star reduces reward."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 0.0

        gap = s_star + 5.0
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", gap)},
        )

        reward = env.compute_lcc_reward()
        # J_spacing = 0.7 * 25 = 17.5
        expected = max(env.J_max - 17.5, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_control_penalty_reduces_reward(self):
        """Non-zero acceleration reduces reward."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 2.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        # J_control = 0.1 * 4 = 0.4
        expected = max(env.J_max - 0.4, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_combined_cost(self):
        """All three cost terms contribute together."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 1.0

        gap = s_star + 3.0
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 12.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", gap)},
        )

        reward = env.compute_lcc_reward()
        # J_vel = 0.8*9 = 7.2, J_spacing = 0.7*9 = 6.3, J_ctrl = 0.1*1 = 0.1
        J = 7.2 + 6.3 + 0.1
        expected = max(env.J_max - J, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_spacing_error_clipped(self):
        """Spacing error is clipped to [-20, 20] before squaring."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 0.0

        gap = s_star + 50.0  # clipped to +20
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", gap)},
        )

        reward = env.compute_lcc_reward()
        # J_spacing = 0.7 * 400 = 280
        expected = max(env.J_max - 280.0, 0.0) / env.J_max
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_reward_is_always_nonnegative(self):
        """Non-negative reward: always in [0, 1]."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 1.5

        _setup_vehicles(
            {
                "car0": {"speed": 10.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 18.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 12.0, "distance": 100.0, "length": 5.0},
            },
            {"car1": ("car0", s_star + 8.0)},
        )

        reward = env.compute_lcc_reward()
        assert 0.0 <= reward <= 1.0

    def test_reward_bounded_above_by_one(self):
        """Reward never exceeds 1.0."""
        env = _make_env()
        s_star = _compute_s_star(env.v_star)
        env.prev_accel = 0.0

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_lcc_reward()
        assert reward <= 1.0


# ---------------------------------------------------------------------------
# Tests: compute_multi_agent_lcc_reward
# ---------------------------------------------------------------------------
class TestMultiAgentLCCReward:

    def test_at_equilibrium_reward_is_one(self):
        env = _make_env(num_agents=2)
        s_star = _compute_s_star(env.v_star)
        env.prev_accels = {"car1": 0.0, "car2": 0.0}

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 100.0, "length": 5.0},
                "car3": {"speed": 15.0, "distance": 150.0, "length": 5.0},
            },
            {
                "car1": ("car0", s_star),
                "car2": ("car1", s_star),
            },
        )

        reward = env.compute_multi_agent_lcc_reward()
        assert reward == pytest.approx(1.0, abs=1e-6)

    def test_sums_spacing_across_agents(self):
        """Spacing error summed across all CAVs."""
        env = _make_env(num_agents=2)
        s_star = _compute_s_star(env.v_star)
        env.prev_accels = {"car1": 0.0, "car2": 0.0}

        gap1 = s_star + 4.0
        gap2 = s_star + 3.0
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 100.0, "length": 5.0},
            },
            {
                "car1": ("car0", gap1),
                "car2": ("car1", gap2),
            },
        )

        reward = env.compute_multi_agent_lcc_reward()
        # J_spacing = 0.7 * (16 + 9) = 17.5
        expected = max(env.J_max_multi - 17.5, 0.0) / env.J_max_multi
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_sums_control_across_agents(self):
        """Control penalty summed across all CAVs."""
        env = _make_env(num_agents=2)
        s_star = _compute_s_star(env.v_star)
        env.prev_accels = {"car1": 2.0, "car2": 1.0}

        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
                "car2": {"speed": 15.0, "distance": 100.0, "length": 5.0},
            },
            {
                "car1": ("car0", s_star),
                "car2": ("car1", s_star),
            },
        )

        reward = env.compute_multi_agent_lcc_reward()
        # J_control = 0.1 * (4 + 1) = 0.5
        expected = max(env.J_max_multi - 0.5, 0.0) / env.J_max_multi
        assert reward == pytest.approx(expected, abs=1e-6)

    def test_inactive_agent_skipped(self):
        """Agent not in active_ids doesn't contribute to reward."""
        env = _make_env(num_agents=2)
        s_star = _compute_s_star(env.v_star)
        env.prev_accels = {"car1": 0.0, "car2": 3.0}

        # car2 absent
        _setup_vehicles(
            {
                "car0": {"speed": 15.0, "distance": 0.0, "length": 5.0},
                "car1": {"speed": 15.0, "distance": 50.0, "length": 5.0},
            },
            {"car1": ("car0", s_star)},
        )

        reward = env.compute_multi_agent_lcc_reward()
        assert reward == pytest.approx(1.0, abs=1e-6)
